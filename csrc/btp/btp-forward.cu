// Copyright 2025 Radical Numerics Inc.
//
// This source code is licensed under the Apache License, Version 2.0, found in the
// LICENSE file in the root directory of this source tree.

/******************************************************************************************
 * Unfused Stable LogSum (BTP) Semi-Separable FORWARD 
 *
 * Materialization: Stable segsum (unfused stable logsum) matching the Python
 * reference construct_L_logsumexp_stable(log_a) = exp(segsum_stable(log_a)).
 * - Row-wise triangular cumsum of log_a in the lower triangle; diag=1; upper=0;
 *   FP32 accumulation with cast to ElementT at store; avoids (P[row]-P[col])
 * subtraction.
 ******************************************************************************************/

#pragma once

// Hopper warning only when compiling device code.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
#warning "BTP DSMEM implementation is optimized for sm_90 (Hopper) or higher."
#endif

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/warp/mma_tensor_op.h>
#include <cutlass/gemm/warp/mma_tensor_op_policy.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>

#include "btp_common.h"

namespace sss {
using namespace cute;

template <typename ElementT>
struct WarpSmemFwd {
  // WMMA output C: [BL, DH] row-major (16B aligned)
  alignas(16) float C_tile[BL * DH];

  // Carry row (k=2): length DH
  alignas(16) float carry_row[DH];

  union {
    struct {
      // WMMA inputs (phase 1)
      alignas(16) ElementT A_tile[BL * BL];
      alignas(16) ElementT B_tile[BL * DH];
    } wmma_inputs;

    // Phase 2: transposed/padded C [DH, BL+1]
    alignas(16) float C_tile_Transposed[DH * (BL + 1)];
  };
};

// DSMEM-visible comms
template <int DH_VAL>
struct alignas(32) ClusterCommsFwd {
  // mbarrier must be 16B aligned for use with async operations.
  alignas(16) cute::uint64_t ingress_barrier_words[2];
  alignas(32) float ingress_carry[DH_VAL];
  static_assert(sizeof(ingress_barrier_words) == 16, "must be 16 bytes for alignment");
};

template <typename ElementT, int WPB>
struct BlockSmemFwd {
  WarpSmemFwd<ElementT> warp_storage[WPB];
  ClusterCommsFwd<DH> cluster_comms;
};

static_assert((sizeof(float) * DH) % 16 == 0,
              "Carry size must be a multiple of 16 bytes for cp.async.bulk alignment");

// Kernel

template <typename ElementT, typename ElementO, int WarpsPerBlock>
__global__ void btp_forward_kernel(ElementT const* __restrict__ coeff,
                                   ElementT const* __restrict__ x, ElementO* __restrict__ y,
                                   float* __restrict__ checkpoint_carry, int H, int L, int k,
                                   int cluster_size_param) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
  return;  // DSMEM path requires Hopper+.
#endif
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  // Cluster group & rank using CuTe
  const uint32_t cluster_rank = cute::block_rank_in_cluster();
  const uint32_t cluster_size = static_cast<uint32_t>(cluster_size_param);

  // Ensure full cluster residency before DSMEM mapping/traffic
  cute::cluster_sync();

  const int block_idx_x = blockIdx.x;
  const int tile_glob = block_idx_x * WarpsPerBlock + warp_id;
  const int tile_start = tile_glob * BL;

  // Dynamic SMEM base
  extern __shared__ char smem[];
  using SmemLayout = BlockSmemFwd<ElementT, WarpsPerBlock>;
  SmemLayout* block_smem = reinterpret_cast<SmemLayout*>(smem);
  auto* local_comms = &block_smem->cluster_comms;
  cute::uint64_t* local_barrier = &local_comms->ingress_barrier_words[0];

  // Barrier phase: Initial phase is 0. We wait for this phase to complete.
  const int wait_phase = 0;

  // DSMEM Initialization (k=2)
  if (k == 2) {
    // Initialize using a single elected thread (Thread 0)
    if (threadIdx.x == 0) {
      // Determine expected arrivals. Rank > 0 expects 1 arrival (from the
      // single cp.async.bulk).
      const int tiles_per_seq_init = (L + BL - 1) / BL;
      const int cta_first_tile_init = blockIdx.x * WarpsPerBlock;
      int cta_tiles_init = tiles_per_seq_init - cta_first_tile_init;
      cta_tiles_init = (cta_tiles_init < 0) ? 0 : ((cta_tiles_init > WarpsPerBlock) ? WarpsPerBlock : cta_tiles_init);
      const bool has_prev_cta = (cta_first_tile_init > 0);
      const bool cta_expects_arrival = (cluster_rank > 0) && (cta_tiles_init > 0) && has_prev_cta;
      const uint32_t expected_arrivals = cta_expects_arrival ? 1u : 0u;
      // 1. Initialize the barrier.
      cute::initialize_barrier(*local_barrier, expected_arrivals);

      // 2. Program expect_tx immediately if arrivals > 0.
      if (expected_arrivals > 0) {
        constexpr unsigned BYTES = sizeof(float) * DH;
        // Program the expected bytes for the current incomplete phase (Phase
        // 0).
        cute::set_barrier_transaction_bytes(*local_barrier, BYTES);
      }
    }
    __syncthreads();

    // CRITICAL: Ensure the initialized AND programmed barrier is visible
    // cluster-wide before any remote producer attempts to arrive on it.
    cute::cluster_sync();
  }

  // Determine active warps in this CTA
  const int tiles_per_seq = (L + BL - 1) / BL;
  const int cta_first_tile = block_idx_x * WarpsPerBlock;
  int cta_tiles = tiles_per_seq - cta_first_tile;
  cta_tiles = (cta_tiles < 0) ? 0 : ((cta_tiles > WarpsPerBlock) ? WarpsPerBlock : cta_tiles);
  const int last_active_warp = cta_tiles - 1;
  const bool tile_valid = (warp_id <= last_active_warp);

  // cute::cluster_sync() at the end to prevent cluster-level deadlocks.

  const int bh_idx = blockIdx.y;
  const long bh_idx_ll = static_cast<long>(bh_idx);
  const long coeff_off = bh_idx_ll * L + tile_start;
  const long x_off = bh_idx_ll * (long)DH * L + tile_start;
  const long BH_off_CP = bh_idx_ll * (L / BL) * (long)DH;

  // Warp-local views
  auto* warp_smem_struct = &block_smem->warp_storage[warp_id];
  float* C_tile = warp_smem_struct->C_tile;
  float* carry = (k == 2) ? warp_smem_struct->carry_row : nullptr;
  ElementT* A_tile = warp_smem_struct->wmma_inputs.A_tile;
  ElementT* B_tile = warp_smem_struct->wmma_inputs.B_tile;

  // 1-4) Computation: log(a), Materialize A, Load B, WMMA
  // (Implementation optimized in v1.3)

  // 1) log(a_t) and prefix sum
  float log_a_lane = 0.0f;
  if (tile_valid && lane < BL && tile_start + lane < L) {
    float a = static_cast<float>(coeff[coeff_off + lane]);
    // Use fast logf() intrinsic for improved performance.
    log_a_lane = (a > 1e-9f) ? __logf(a) : -INFINITY;
  }

  // We no longer need the full log_a_regs broadcast.
  const float log_a0 = __shfl_sync(0xffffffffu, log_a_lane, 0);

  // Calculate prefix sum: P[i] = sum_{j=1}^{i} log_a[j]. P[0]=0.
  // Input to scan: (0, log_a[1], log_a[2], ...)
  float scan_in = (lane > 0) ? log_a_lane : 0.0f;
  float log_prefix_sum = scan_in;
#pragma unroll
  for (int stride = 1; stride < BL; stride <<= 1) {
    float t = __shfl_up_sync(0xffffffffu, log_prefix_sum, stride);
    if (lane >= stride) log_prefix_sum += t;
  }

  // Step 2.
  float log_prefix_sum_regs[BL];
#pragma unroll
  for (int i = 0; i < BL; ++i) {
    log_prefix_sum_regs[i] = __shfl_sync(0xffffffffu, log_prefix_sum, i);
  }

  // build.
  float log_a_regs[BL];
#pragma unroll
  for (int i = 0; i < BL; ++i) {
    log_a_regs[i] = __shfl_sync(0xffffffffu, log_a_lane, i);
  }

  // 2) Materialize A using the stable segsum trick (no P[row]-P[col]
  // subtraction)
  //     S[r,c] = sum_{i=c+1}^r log a[i]   (r>=c), then A[r,c] = exp(S[r,c])
  //     -> exactly matches construct_L_logsumexp_stable(log_a) in the
  //     reference.
  if (tile_valid) {
    // Each lane < BL writes a whole row 'r' of A.
    if (lane < BL) {
      const int r = lane;

// Above diagonal: zero
#pragma unroll
      for (int c = r + 1; c < BL; ++c) {
        A_tile[r * BL + c] = ElementT(0.0f);
      }

      // Diagonal: one
      A_tile[r * BL + r] = ElementT(1.0f);

      // Strictly lower triangle: stable cumulative sum from right to left
      // S[r, r] = 0 (already handled by diag=1)
      float s = 0.0f;
#pragma unroll
      for (int c = r - 1; c >= 0; --c) {
        // add log a[c+1]; never subtract two large prefixes
        s += log_a_regs[c + 1];

        float v = __expf(s);
        // Guard: if exp overflows to NaN due to +/-Inf arithmetic upstream,
        // zero it.
        if (isnan(v)) v = 0.0f;

        A_tile[r * BL + c] = static_cast<ElementT>(v);
      }
    }
  }

  // 3) Load B
  if (tile_valid) {
    for (int idx = lane; idx < DH * BL; idx += 32) {
      int t = idx % BL;
      int c = idx / BL;
      B_tile[idx] = (tile_start + t < L) ? x[x_off + (long)c * L + t] : ElementT(0.0f);
    }
  }

  __syncwarp();

  // 4) WMMA: C = A * B
  if (tile_valid) {
    // Assumes WarpMma is defined in btp_common.h
    using Mma = WarpMma<ElementT>;
    typename Mma::IteratorA itA({A_tile, BL}, lane);
    typename Mma::IteratorB itB({B_tile, BL}, lane);
    typename Mma::IteratorC itC({C_tile, DH}, lane);

    typename Mma::FragmentA fA;
    typename Mma::FragmentB fB;
    typename Mma::FragmentC fC{};

    Mma mma;
    itA.load(fA);
    itB.load(fB);
    mma(fC, fA, fB, fC);
    itC.store(fC);
  }

  // 5) k==2: rank-1 correction using carry_{i-1} (CuTe DSMEM Transfer)
  if (k == 2) {
    // 5.0 Synchronization and Producer (CTA level)

    // Sync 1 & 2: Ensure Step 4 (WMMA) is complete and results are visible
    // across all warps before T0 proceeds to issue the copy or intra-CTA
    // One is sufficient.
    __syncthreads();


    // Determine if this CTA needs to send data to the next one.
    const bool subsequent_tile_exists = (cta_first_tile + cta_tiles) < tiles_per_seq;

    // We only send if we are not the last rank in the cluster, the next tile
    // exists, and this CTA actually processed data (cta_tiles > 0).
    if (cluster_rank < cluster_size - 1 && subsequent_tile_exists && cta_tiles > 0) {
      // Only T0 issues the async copy.
      if (threadIdx.x == 0) {
        const uint32_t dst_rank = cluster_rank + 1;


        // Source (Local SMEM - last row of C_tile from the last active warp)
        auto* producer_warp_smem = &block_smem->warp_storage[last_active_warp];
        const float* src_ptr = producer_warp_smem->C_tile + (BL - 1) * DH;
        uint32_t src_smem_addr = cute::cast_smem_ptr_to_uint(src_ptr);

        // Destination (Remote DSMEM - ingress_carry)
        uint32_t local_ingress_carry_addr =
            cute::cast_smem_ptr_to_uint(&local_comms->ingress_carry[0]);
        uint32_t dst_dsmem_addr = cute::set_block_rank(local_ingress_carry_addr, dst_rank);

        // Remote Barrier
        uint32_t local_barrier_addr = cute::cast_smem_ptr_to_uint(local_barrier);
        uint32_t remote_mbarrier_addr = cute::set_block_rank(local_barrier_addr, dst_rank);

        constexpr unsigned BYTES = sizeof(float) * DH;

        // Transfer data and arrive on the remote barrier.
        cp_async_bulk_dsmem(dst_dsmem_addr, src_smem_addr, BYTES, remote_mbarrier_addr);
      }
    }

    // 5.1 Consumer Synchronization (CTA level)

    // Sync 3: Ensure the send (if any) is initiated before we start waiting.
    __syncthreads();

    // If this CTA expects data precisely, wait then acquire DSMEM visibility.
    {
      const int tiles_per_seq_wait = (L + BL - 1) / BL;
      const int cta_first_tile_wait = blockIdx.x * WarpsPerBlock;
      int cta_tiles_wait = tiles_per_seq_wait - cta_first_tile_wait;
      cta_tiles_wait = (cta_tiles_wait < 0) ? 0 : ((cta_tiles_wait > WarpsPerBlock) ? WarpsPerBlock : cta_tiles_wait);
      const bool has_prev_cta = (cta_first_tile_wait > 0);
      const bool cta_expects_arrival = (cluster_rank > 0) && (cta_tiles_wait > 0) && has_prev_cta;
      if (cta_expects_arrival) {
        // Wait for the completion of the initial phase (Phase 0).
        cute::wait_barrier(*local_barrier, wait_phase);
        // Acquire visibility before reading DSMEM.
        cute::cluster_sync();
      }
    }

    // 5.2 Consumer Data Loading (Warp level)

    const bool has_prev_tile = (tile_glob > 0);
    const bool prev_tile_in_same_cta = (warp_id > 0);

    if (tile_valid && has_prev_tile) {
      const float* src_carry_ptr = nullptr;

      if (prev_tile_in_same_cta) {
        // Intra-CTA: read previous warp’s last row.
        // Synchronized by Sync 1&2 in 5.0.
        auto* prev_ws = warp_smem_struct - 1;
        src_carry_ptr = prev_ws->C_tile + (BL - 1) * DH;
      } else {
        // Inter-CTA (Warp 0)
        if (cluster_rank > 0) {
          // Inter-CTA, Intra-Cluster. Data is ready due to wait_barrier()
          // in 5.1.
          src_carry_ptr = local_comms->ingress_carry;
        }
        // else: Inter-cluster boundary (Rank 0, but tile_glob > 0): segmented.
        // src_carry_ptr remains null.
      }

      // Cooperative load from the determined source.
      for (int i = lane; i < DH; i += 32) {
        // If src_carry_ptr is null (segmented boundary), load 0.0f.
        carry[i] = (src_carry_ptr != nullptr) ? src_carry_ptr[i] : 0.0f;
      }

      // Checkpoint carry for backward pass
      const long cp_off = BH_off_CP + (static_cast<long>(tile_glob) - 1) * DH;
      for (int i = lane; i < DH; i += 32) {
        checkpoint_carry[cp_off + i] = carry[i];
      }
    }

    // Ensure 'carry' buffer is fully loaded by all threads in the warp before
    // applying correction
    __syncwarp();

    // 5.3 Apply Correction (Warp level) (Optimized in v1.3)

    // Apply C += V @ carry^T (stable)
    if (tile_valid && has_prev_tile) {
      // log_a0 was calculated efficiently in Step 1.

      // of the loop.

      // Cooperatively calculate V_row[lane] using the per-lane log_prefix_sum
      // result.
      float V_row_lane = 0.0f;  // Initialize for lanes >= BL.

      if (lane < BL) {
        // log_P_row for this lane is readily available in log_prefix_sum.
        float log_P_lane = log_prefix_sum;
        float log_V_lane = log_P_lane + log_a0;
        V_row_lane = __expf(log_V_lane);  // Use fast expf() intrinsic.

        // Handle NaN (Inf + -Inf) - Stability check remains exactly the same.
        if (isnan(V_row_lane)) {
          if (isinf(log_P_lane) && isinf(log_a0) && (log_P_lane * log_a0 < 0)) {
            V_row_lane = 0.0f;
          }
        }
      }

      // Broadcast the calculated V_row values to all threads.
      float V_rows[BL];
#pragma unroll
      for (int i = 0; i < BL; ++i) {
        V_rows[i] = __shfl_sync(0xffffffffu, V_row_lane, i);
      }

      // Apply the correction using the pre-calculated V_rows.
      for (int idx = lane; idx < DH * BL; idx += 32) {
        int row = idx / DH;
        int col = idx % DH;

        // Use the pre-calculated V_row (removes divergent shuffle and redundant
        // expf).
        float V_row = V_rows[row];

        C_tile[idx] += safe_multiply(V_row, carry[col]);
      }
    }
  }

  // 5b) Transpose + pad
  __syncwarp();
  constexpr int BL_P = BL + 1;
  float* C_T = warp_smem_struct->C_tile_Transposed;

  if (tile_valid) {
    for (int idx = lane; idx < DH * BL; idx += 32) {
      int r = idx / DH;
      int c = idx % DH;
      C_T[c * BL_P + r] = C_tile[idx];
    }
  }
  __syncwarp();

  // 6) Store to global
  const long y_off = x_off;
  if (tile_valid) {
    for (int idx = lane; idx < DH * BL; idx += 32) {
      int c = idx / BL;
      int r = idx % BL;
      if (tile_start + r < L) {
        float v = C_T[c * BL_P + r];
        y[y_off + (long)c * L + r] = static_cast<ElementO>(v);
      }
    }
  }

  // Finalization: No CTA exits while others may still access its DSMEM.
  // All CTAs must converge here.
  cute::cluster_sync();
}

// Host launcher (Updated to use CUTLASS Cluster Launch API)
// (Launcher implementation remains unchanged)

template <typename ElementT, typename ElementO, int WarpsPerBlock>
void launch_block_two_pass(ElementT const* coeff, ElementT const* x, ElementO* y,
                           float* checkpoint_carry, int B, int H, int L, int k,
                           cudaStream_t stream = 0) {
  if (L <= 0 || B * H <= 0) return;

  constexpr int WPB = WarpsPerBlock;
  constexpr int threads_per_block = 32 * WPB;

  const int tiles_per_seq = (L + BL - 1) / BL;
  const int total_blocks_x = (tiles_per_seq + WPB - 1) / WPB;

  // Determine Cluster dimensions
  const int MAX_CLUSTER = 16;
  int cluster_x = std::min(total_blocks_x, MAX_CLUSTER);

  // Prefer portable powers-of-two when possible, ensuring cluster_x divides
  // total_blocks_x This ensures segmentation boundaries align with cluster
  // boundaries.
  const int candidates[] = {16, 8, 4, 2, 1};
  for (int c : candidates) {
    if (c <= cluster_x && (total_blocks_x % c == 0)) {
      cluster_x = c;
      break;
    }
  }

  // Grid configuration for cutlass::launch_kernel_on_cluster
  dim3 grid_dims(total_blocks_x, B * H, 1);
  dim3 cluster_dims(cluster_x, 1, 1);
  dim3 block_dims(threads_per_block, 1, 1);

  constexpr int smem_bytes = int(sizeof(BlockSmemFwd<ElementT, WPB>));
  static_assert(sizeof(BlockSmemFwd<ElementT, WPB>) % 16 == 0, "SMEM alignment");
  static_assert(32 * WPB <= 1024, "Block size must be <= 1024 threads");

  const void* kernel_ptr = (const void*)btp_forward_kernel<ElementT, ElementO, WPB>;

  // Allow non-portable cluster sizes (mirrors the demo's permissive launch)
  cudaError_t err1 =
      cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (err1 != cudaSuccess) {
    printf(
        "[BTP_FORWARD LAUNCH ERROR] Failed to set non-portable cluster "
        "size: %s\n",
        cudaGetErrorString(err1));
    return;
  }

  // Opt in to the dynamic smem we will request at launch
  cudaError_t err2 =
      cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  if (err2 != cudaSuccess) {
    printf(
        "[BTP_FORWARD LAUNCH ERROR] Failed to set dynamic shared memory "
        "size (%d bytes): %s\n",
        smem_bytes, cudaGetErrorString(err2));
    return;
  }

  // Launch using the demo's helper
  cutlass::launch_kernel_on_cluster({grid_dims, block_dims, cluster_dims, smem_bytes, stream},
                                    kernel_ptr,
                                    // Kernel arguments:
                                    coeff, x, y, checkpoint_carry, H, L, k, int(cluster_dims.x));
}

// Explicit instantiations
#define INSTANTIATE_BTP_FWD(T, O, WPB)                                                          \
  template void launch_block_two_pass<T, O, WPB>(T const*, T const*, O*, float*, int, int, int, \
                                                 int, cudaStream_t);

#define INSTANTIATE_ALL_WPB(T, O) \
  INSTANTIATE_BTP_FWD(T, O, 4)    \
  INSTANTIATE_BTP_FWD(T, O, 8)    \
  INSTANTIATE_BTP_FWD(T, O, 16)   \
  INSTANTIATE_BTP_FWD(T, O, 32)

// Use CUTLASS definitions for types
using half_t = cutlass::half_t;
using bf16_t = cutlass::bfloat16_t;

INSTANTIATE_ALL_WPB(half_t, float)
INSTANTIATE_ALL_WPB(half_t, half_t)
INSTANTIATE_ALL_WPB(bf16_t, float)
INSTANTIATE_ALL_WPB(bf16_t, bf16_t)

#undef INSTANTIATE_BTP_FWD
#undef INSTANTIATE_ALL_WPB

}  // namespace sss
