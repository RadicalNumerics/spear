// Copyright 2025 Radical Numerics Inc.
//
// This source code is licensed under the Apache License, Version 2.0,
// found in the LICENSE file in the root directory of this source tree.

/******************************************************************************************
 * Unfused Stable LogSum (BTP) Semi-Separable BACKWARD 
 *
 * Materialization (A^T): Stable segsum (unfused stable logsum) as in the Python
 * reference construct_L_logsumexp_stable(log_a) = exp(segsum_stable(log_a)).
 * - Column-wise triangular cumsum of log_a in the upper triangle; diag=1;
 * below=0; FP32 accumulation with cast to ElementT at store; avoids
 * (P[col]-P[row]) subtraction.
 ******************************************************************************************/

#pragma once

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
#warning "BTP DSMEM implementation (k=2) is optimized for sm_90 (Hopper) or higher."
#endif

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/warp/mma_tensor_op.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>

#include "btp_common.h"

namespace sss {

using namespace cute;

//  Constants
#ifndef BL
#define BL 16
#endif
#ifndef DH
#define DH 16
#endif

//  Shared Memory Layout (DSMEM Enabled - Optimized Structure)

// Warp Private Storage
template <typename ElementT, int DH_VAL, int BLK>
struct BackwardWarpSmem {
  // Padded dimension to avoid bank conflicts.
  static constexpr int BLK_P = BLK + 1;

  // Stores log(P[r]).
  float log_prefix[BLK];

  // Buffer for Carry_{i-1} (Used in Part V)
  float Carry_im1[DH_VAL];

  // dC buffer [DH, BLK+1] RM (FP32). Padded.
  float DC_tile_RM_F32[DH_VAL * BLK_P];

  // X tile [DH, BLK+1] RM. Padded.
  alignas(16) ElementT X_tile_RM[DH_VAL * BLK_P];

  // WMMA Output buffer (dX^T). Reused for P, U, and SuffixMaxS.
  alignas(16) float DX_tile[BLK * DH_VAL];

  // [DSMEM] Buffer for the dCarry produced by this warp (outgoing to the left).
  alignas(16) float outgoing_dCarry[DH_VAL];

  // Union for memory reuse
  union {
    // WMMA Inputs (Used for dX WMMA)
    struct {
      alignas(16) ElementT A_tile[BLK * BLK];
      // B_tile (dC) is unpadded [DH, BLK] RM.
      alignas(16) ElementT B_tile[BLK * DH_VAL];
    } wmma_inputs;

    // Transposed DX [DH, BLK+1] RM.
    alignas(16) float DX_tile_Transposed[DH_VAL * (BLK + 1)];
  };
};

// DSMEM-visible comms (Inter-CTA)
template <int DH_VAL>
struct alignas(64) ClusterCommsBwd {
  // Barrier for incoming data from the RIGHT (CTA R+1)
  alignas(16) cute::uint64_t ingress_barrier;
  // Buffer for incoming data from the RIGHT (CTA R+1)
  alignas(16) float ingress_dCarry[DH_VAL];
};

// Final Block Layout
template <typename ElementT, int DH_VAL, int BLK, int WARPS_PER_BLOCK>
struct BackwardSmemLayout {
  BackwardWarpSmem<ElementT, DH_VAL, BLK> warp_storage[WARPS_PER_BLOCK];
  ClusterCommsBwd<DH_VAL> cluster_comms;
};

static_assert((sizeof(float) * DH) % 16 == 0,
              "Carry size must be a multiple of 16 bytes for alignment");

//  Helper Functions and Types


// Warp-synchronous prefix sum (Inclusive) for log space.
__device__ __forceinline__ float warp_prefix_sum(float val, int lane) {
#pragma unroll
  for (int offset = 1; offset < BL; offset <<= 1) {
    float other = __shfl_up_sync(0xffffffff, val, offset);
    if (lane >= offset) val += other;
  }
  return val;
}

// Warp-synchronous inclusive suffix sum (Used in Part V)
__device__ __forceinline__ float warp_suffix_sum_inclusive(float val, int lane) {
#pragma unroll
  for (int offset = 1; offset < BL; offset <<= 1) {
    float other = __shfl_down_sync(0xffffffff, val, offset);
    if (lane + offset < BL) {
      val += other;
    }
  }
  return val;
}

// Helper: Suffix Max (Needed for stable Part A)
__device__ __forceinline__ float warp_suffix_max_32(float val) {
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    float other = __shfl_down_sync(0xffffffff, val, offset);
    val = fmaxf(val, other);
  }
  return val;
}

// Helpers for Packed Types (PackedType<T>)
template <typename T>
struct PackedType;

template <>
struct PackedType<cutlass::half_t> {
  using type = __half2;
  static __device__ __forceinline__ type pack_and_convert(float a, float b) {
    return __floats2half2_rn(a, b);
  }
};

template <>
struct PackedType<cutlass::bfloat16_t> {
  using type = __nv_bfloat162;
  static __device__ __forceinline__ type pack_and_convert(float a, float b) {
    return __floats2bfloat162_rn(a, b);
  }
};

//  Main Backward Kernel (DSMEM Implementation, Optimized & Fixed)
template <typename ElementT, typename ElementO, int DH_VAL = 16, int WARPS_PER_BLOCK>
__global__ void btp_backward_kernel(const ElementT* __restrict__ g_coeff,
                                    const ElementT* __restrict__ g_x,
                                    const ElementO* __restrict__ g_dy,
                                    const float* __restrict__ g_checkpoint_carry,
                                    ElementT* __restrict__ g_dx, float* __restrict__ g_dA, int L,
                                    int H, int k, int cluster_size_param) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
  if (k == 2) return;  // DSMEM path requires Hopper+.
#endif

  // Constants (BLK=16, DH=16)
  constexpr int BLK = BL;

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  const uint32_t cluster_rank = cute::block_rank_in_cluster();
  const uint32_t cluster_size = static_cast<uint32_t>(cluster_size_param);

  // Ensure full cluster residency
  cute::cluster_sync();

  const int block_idx_x = blockIdx.x;
  const int tile_glob = block_idx_x * WARPS_PER_BLOCK + warp_id;
  const int l0 = tile_glob * BLK;

  const int bh_idx = blockIdx.y;

  // Calculate activity status
  const int tiles_per_seq = (L + BLK - 1) / BLK;
  const int cta_first_tile = block_idx_x * WARPS_PER_BLOCK;
  int cta_tiles = tiles_per_seq - cta_first_tile;
  cta_tiles = (cta_tiles < 0) ? 0 : ((cta_tiles > WARPS_PER_BLOCK) ? WARPS_PER_BLOCK : cta_tiles);
  const bool tile_valid = (warp_id < cta_tiles);

  extern __shared__ uint8_t smem_base[];
  using SmemLayout = BackwardSmemLayout<ElementT, DH_VAL, BLK, WARPS_PER_BLOCK>;
  SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_base);

  // Define the padded stride constant.
  using WarpSmemType = BackwardWarpSmem<ElementT, DH_VAL, BLK>;
  constexpr int BLK_P = WarpSmemType::BLK_P;

  // [DSMEM] SMEM pointers
  auto* warp_smem = &smem->warp_storage[warp_id];
  auto* local_comms = &smem->cluster_comms;
  cute::uint64_t* local_barrier = &local_comms->ingress_barrier;

  // Pointers for WMMA inputs (used for dX)
  ElementT* A_tile = warp_smem->wmma_inputs.A_tile;
  ElementT* B_tile = warp_smem->wmma_inputs.B_tile;

  // Pointers for P and U storage (reusing DX_tile buffer).
  float* P_smem = warp_smem->DX_tile;
  float* U_smem = warp_smem->DX_tile + BLK;

  // DSMEM Initialization (k=2)
  const int wait_phase = 0;
  bool expects_arrival = false;

  if (k == 2) {
    if (threadIdx.x == 0) {
      uint32_t expected_arrivals = 0u;
      // Check if the next CTA (to the right) exists globally and is in the
      // cluster.
      if (cluster_rank < cluster_size - 1) {
        const int next_cta_start_tile = (block_idx_x + 1) * WARPS_PER_BLOCK;
        if (next_cta_start_tile < tiles_per_seq) {
          expected_arrivals = 1u;
        }
      }
      cute::initialize_barrier(*local_barrier, expected_arrivals);
      if (expected_arrivals > 0) {
        constexpr unsigned BYTES = sizeof(float) * DH_VAL;
        cute::set_barrier_transaction_bytes(*local_barrier, BYTES);
      }
    }

    // Determine if this CTA expects arrival
    if (cluster_rank < cluster_size - 1) {
      const int next_cta_start_tile = (block_idx_x + 1) * WARPS_PER_BLOCK;
      if (next_cta_start_tile < tiles_per_seq) {
        expects_arrival = true;
      }
    }

    __syncthreads();
    cute::cluster_sync();
  }

  const long long bh_idx_ll = static_cast<long long>(bh_idx);
  const long long coeff_offset = bh_idx_ll * L + l0;
  const long long x_offset = bh_idx_ll * DH_VAL * L + l0;
  // Assumes L is multiple of BLK for checkpoint offset.
  const long long checkpoint_offset =
      bh_idx_ll * (L / BLK) * DH_VAL + (static_cast<long long>(tile_glob - 1) * DH_VAL);

  const ElementT* coeff_tile_ptr = g_coeff + coeff_offset;
  const ElementT* x_tile_ptr = g_x + x_offset;
  const ElementO* dy_tile_ptr = g_dy + x_offset;
  const float* checkpoint_ptr = (tile_glob > 0 && g_checkpoint_carry != nullptr)
                                    ? (g_checkpoint_carry + checkpoint_offset)
                                    : nullptr;

  ElementT* dx_tile_ptr = g_dx + x_offset;
  float* dA_tile_ptr = g_dA + coeff_offset;


  // Load Coefficients and calculate log(a_t).
  float log_a_val_lane = 0.0f;
  float inv_a_val_lane = 0.0f;

  if (tile_valid && lane < BLK && l0 + lane < L) {
    float a_val_lane = (float)coeff_tile_ptr[lane];
    if (a_val_lane > 1e-9f) {
      log_a_val_lane = __logf(a_val_lane);
      inv_a_val_lane = __expf(-log_a_val_lane);  // Calculate here
    } else {
      log_a_val_lane = -INFINITY;
      inv_a_val_lane = INFINITY;  // 1/0 = Inf
    }
  }

  // Register to accumulate dA
  float dA_tile_reg = 0.0f;

  // Load X tile into SMEM
  if (tile_valid) {
    // Use padded stride BLK_P for SMEM writes.
    for (int i = lane; i < DH_VAL * BLK; i += 32) {
      int c = i / BLK;
      int r = i % BLK;
      int smem_idx = c * BLK_P + r;

      if (l0 + r < L) {
        warp_smem->X_tile_RM[smem_idx] = x_tile_ptr[static_cast<long long>(c) * L + r];
      } else {
        warp_smem->X_tile_RM[smem_idx] = ElementT(0.0f);
      }
    }
  }


  // 1.1: Compute log-space prefix products log(P_i) (Warp collective)
  float scan_input_log = log_a_val_lane;
  float log_prefix_val = warp_prefix_sum(scan_input_log, lane);

  if (tile_valid && lane < BLK) {
    warp_smem->log_prefix[lane] = log_prefix_val;
  }
  __syncwarp();

  // 1.2 Load dY tile and upcast to fp32 -> dC_i
  if (tile_valid) {
    // Use padded stride BLK_P for SMEM writes.
    for (int i = lane; i < DH_VAL * BLK; i += 32) {
      int c = i / BLK;
      int r = i % BLK;
      int smem_idx = c * BLK_P + r;

      if (l0 + r < L) {
        warp_smem->DC_tile_RM_F32[smem_idx] =
            static_cast<float>(dy_tile_ptr[static_cast<long long>(c) * L + r]);
      } else {
        warp_smem->DC_tile_RM_F32[smem_idx] = 0.0f;
      }
    }
  }
  __syncwarp();

  // Phases 1.3 and 1.4 handle k=2 dependencies.
  if (k == 2) {
    // 1.3 Compute outgoing dCarry

    // Step 1.3.9: Materialize P stably.
    if (tile_valid) {
      for (int r = lane; r < BLK; r += 32) {
        float log_P_r = warp_smem->log_prefix[r];
        float v_r = __expf(log_P_r);

        // Handle potential NaN (Inf + -Inf -> 0).
        if (isnan(v_r) && isnan(log_P_r)) {
          v_r = 0.0f;
        }
        P_smem[r] = v_r;
      }
    }
    __syncwarp();  // Ensure P is visible.

    // Step 1.3.2: Calculate dCarry = dC @ P.
    if (tile_valid) {
      // Distribute the calculation of the DH elements across the warp.
      if (lane < DH_VAL) {
        int c = lane;
        float sum = 0.0f;
#pragma unroll
        for (int r = 0; r < BLK; r++) {
          float P_r = P_smem[r];
          // Access pattern for dC (c*BLK_P + r). Conflict-free.
          float dC_cr = warp_smem->DC_tile_RM_F32[c * BLK_P + r];
          sum += safe_multiply(P_r, dC_cr);
        }
        warp_smem->outgoing_dCarry[c] = sum;
      }
    }

    // 1.4 Compute dCoeff (Part V) - Depends on checkpointed forward carry.
    // tile_glob > 0 is checked by checkpoint_ptr != nullptr.
    if (tile_valid && checkpoint_ptr != nullptr) {
      // 1.4.1 Load Carry_{i-1}
      for (int i = lane; i < DH_VAL; i += 32) {
        warp_smem->Carry_im1[i] = checkpoint_ptr[i];
      }
      __syncwarp();

      // 1.4.2 Calculate dV[r] and U[r]
      if (lane < BLK) {
        int r = lane;
        // dV[r] = dC[r] @ Carry_{i-1}
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < DH_VAL; c++) {
          // Access pattern for dC: c * BLK_P + r. Conflict-free.
          sum += safe_multiply(warp_smem->DC_tile_RM_F32[c * BLK_P + r], warp_smem->Carry_im1[c]);
        }

        // U[r] = dV[r] * P[r]. (P_smem was materialized in 1.3.9)
        float P_r = P_smem[r];
        U_smem[r] = safe_multiply(sum, P_r);
      }
      __syncwarp();

      // 1.4.3 Calculate dA from U (Parallel Suffix Sum) (Warp collective)
      float U_t = (lane < BLK) ? U_smem[lane] : 0.0f;
      float SuffixSum_t = warp_suffix_sum_inclusive(U_t, lane);

      if (lane < BLK) {
        // Calculate contribution = SuffixSum_t / a_t.
        float contribution;
        if (SuffixSum_t == 0.0f) {
          contribution = 0.0f;  // Handle 0/0=0
        } else {
          contribution = safe_multiply(SuffixSum_t, inv_a_val_lane);
        }

        dA_tile_reg += contribution;
      }
    }
  }


  if (k == 2) {
    // 2.1 Synchronization (Ensure Phase 1.3 outgoing_dCarry is ready across all
    // warps)
    __syncthreads();

    // Sync 2 (Ordering fence - ensure SMEM writes are visible before T0 issues
    // cp.async)
    __syncthreads();

    // 2.2 Producer Logic (Inter-CTA, T0 handles communication)
    // Producer is the CTA on the RIGHT (Rank R), sending to the LEFT (Rank
    // R-1).
    if (cluster_rank > 0 && cta_tiles > 0) {
      if (threadIdx.x == 0) {
        const uint32_t dst_rank = cluster_rank - 1;

        // Source: Warp 0's outgoing_dCarry (the dCarry for the entire CTA).
        auto* producer_warp_smem = &smem->warp_storage[0];
        const float* src_ptr = producer_warp_smem->outgoing_dCarry;
        uint32_t src_smem_addr = cute::cast_smem_ptr_to_uint(src_ptr);

        // Destination: Previous CTA's ingress_dCarry.
        uint32_t local_ingress_addr = cute::cast_smem_ptr_to_uint(&local_comms->ingress_dCarry[0]);
        uint32_t dst_dsmem_addr = cute::set_block_rank(local_ingress_addr, dst_rank);

        // Remote Barrier: Previous CTA's ingress_barrier.
        uint32_t local_barrier_addr = cute::cast_smem_ptr_to_uint(local_barrier);
        uint32_t remote_mbarrier_addr = cute::set_block_rank(local_barrier_addr, dst_rank);

        constexpr unsigned BYTES = sizeof(float) * DH_VAL;

        // Transfer data and arrive on the remote barrier.
        cp_async_bulk_dsmem(dst_dsmem_addr, src_smem_addr, BYTES, remote_mbarrier_addr);
      }
    }

    // 2.3 Consumer Synchronization (Inter-CTA)

    // Sync 3: Ensure send is initiated before wait (R5).
    __syncthreads();

    // If this CTA expects data (determined during initialization), wait
    // collectively (R3).
    if (expects_arrival) {
      cute::wait_barrier(*local_barrier, wait_phase);
    }
  }


  // 3.1 Update dC_i (Apply incoming dCarry from the RIGHT)
  // This phase runs concurrently with 3.2 (ILP).
  if (k == 2) {
    if (tile_valid) {
      const bool has_next_tile_globally = (tile_glob + 1 < tiles_per_seq);

      const bool next_tile_in_same_cta =
          (warp_id < WARPS_PER_BLOCK - 1) && (warp_id + 1 < cta_tiles);

      const float* src_dCarry_ptr = nullptr;

      if (has_next_tile_globally) {
        if (next_tile_in_same_cta) {
          // Intra-CTA: read next warp's outgoing_dCarry.
          auto* next_ws = warp_smem + 1;
          src_dCarry_ptr = next_ws->outgoing_dCarry;
        } else if (expects_arrival) {
          // Inter-CTA: Read from the ingress buffer (data ready due to
          // wait_barrier).
          src_dCarry_ptr = local_comms->ingress_dCarry;
        }
        // else: Inter-cluster boundary. src_dCarry_ptr remains null
        // (segmented).
      }

      // Apply the correction if we found a source.
      if (src_dCarry_ptr != nullptr) {
        for (int c = lane; c < DH_VAL; c += 32) {
          float dCarry_in = src_dCarry_ptr[c];
          // Apply to the last element of the tile (r = BLK-1)
          // Use padded stride BLK_P.
          warp_smem->DC_tile_RM_F32[c * BLK_P + (BLK - 1)] += dCarry_in;
        }
      }
    }
  }

  // Guard subsequent computations with tile_valid.
  if (tile_valid) {

    // 1. Cooperatively load log_prefix from SMEM (calculated in Phase 1.1).
    float my_log_prefix = 0.0f;
    if (lane < BLK) {
      my_log_prefix = warp_smem->log_prefix[lane];
    }

    // 2. Broadcast to all threads in the warp.
    float log_prefix_regs[BLK];
#pragma unroll
    for (int i = 0; i < BLK; ++i) {
      log_prefix_regs[i] = __shfl_sync(0xffffffff, my_log_prefix, i);
    }

    // 3.2 Materialize A^T using the stable segsum trick
    // For A^T, we need upper-triangular values:
    //   S^T[row, col] = sum_{i=row+1}^{col} log a[i] (col >= row)
    //   A^T[row, col] = exp(S^T[row, col]); diag=1; below-diag=0.
    {
      float log_a_regs[BLK];
#pragma unroll
      for (int i = 0; i < BLK; ++i) {
        log_a_regs[i] = __shfl_sync(0xffffffff, log_a_val_lane, i);
      }

      // Each lane < BLK writes a full column 'c' of A^T.
      if (lane < BLK) {
        const int c = lane;

// Below diagonal: zero
#pragma unroll
        for (int r = c + 1; r < BLK; ++r) {
          A_tile[r * BLK + c] = ElementT(0.0f);
        }

        // Diagonal: one
        A_tile[c * BLK + c] = ElementT(1.0f);

        // Strictly above diagonal: cumulative sum upward (decreasing row)
        float s = 0.0f;
#pragma unroll
        for (int r = c - 1; r >= 0; --r) {
          // add log a[r+1]; no prefix subtraction
          s += log_a_regs[r + 1];

          float v = __expf(s);
          if (isnan(v)) v = 0.0f;

          A_tile[r * BLK + c] = static_cast<ElementT>(v);
        }
      }
    }
  }

  // Synchronization required before 3.3 and 3.4.
  __syncwarp();

  if (tile_valid) {
    // 3.3 Prepare dC for dX WMMA
    // Copy from padded [DH, BLK_P] RM (FP32) to unpadded [DH, BLK] RM (B_tile),
    // and downcast.
    using PackedElementT = typename PackedType<ElementT>::type;
    PackedElementT* dst_ptr = reinterpret_cast<PackedElementT*>(B_tile);
    const float* src_ptr_base = warp_smem->DC_tile_RM_F32;

    constexpr int NUM_PACKED = (DH_VAL * BLK) / 2;

    // Efficient packed copy.
    for (int i = lane; i < NUM_PACKED; i += 32) {
      // i = c * (BLK/2) + r_base/2.
      int c = i / (BLK / 2);
      int r_base = (i % (BLK / 2)) * 2;

      // Calculate source index in the padded layout.
      int src_idx = c * BLK_P + r_base;

      // Read the two consecutive elements.
      float f1 = src_ptr_base[src_idx];
      float f2 = src_ptr_base[src_idx + 1];

      // Pack and write to the unpadded destination.
      dst_ptr[i] = PackedType<ElementT>::pack_and_convert(f1, f2);
    }

    __syncwarp();

    // 3.4 WMMA: dX^T = A^T @ dC^T. (A=RM, B=RM -> C=RM)
    // Assuming WarpMma supports RM@RM configuration.
    using Mma = WarpMma<ElementT>;
    typename Mma::IteratorA itA({A_tile, BLK}, lane);
    // B_tile layout is [DH, BLK] RM, stride is BLK.
    typename Mma::IteratorB itB({B_tile, BLK}, lane);
    // C_tile (DX_tile) layout is [BLK, DH] RM, stride is DH_VAL.
    typename Mma::IteratorC itC({warp_smem->DX_tile, DH_VAL}, lane);

    typename Mma::FragmentA fragA;
    typename Mma::FragmentB fragB;
    typename Mma::FragmentC fragC{};

    itA.load(fragA);
    itB.load(fragB);
    Mma mma;
    mma(fragC, fragA, fragB, fragC);
    itC.store(fragC);

    // 3.4b: Shared Memory Transpose
    __syncwarp();

    float* DX_T = warp_smem->DX_tile_Transposed;

    // Read [BLK, DH] RM from DX_tile, Write [DH, BLK+1] RM to DX_T.
    for (int i = lane; i < DH_VAL * BLK; i += 32) {
      int r = i / DH_VAL;
      int c = i % DH_VAL;
      float val = warp_smem->DX_tile[i];

      int write_idx = c * BLK_P + r;
      DX_T[write_idx] = val;
    }
    __syncwarp();

    // 3.5 Store dX
    for (int i = lane; i < DH_VAL * BLK; i += 32) {
      int c = i / BLK;
      int r = i % BLK;

      if (l0 + r < L) {
        float val = DX_T[c * BLK_P + r];
        dx_tile_ptr[static_cast<long long>(c) * L + r] = static_cast<ElementT>(val);
      }
    }

    // 3.6 Compute dCoeff (Part A) - Stable Iterative Approach (O(BLK^2*DH))
    // and sync.

    __syncwarp();  // Ensure previous SMEM usage (DX_tile reuse) is finished.

    // 3.6.0: Pre-calculate Suffix Maximums of s[r] (log_prefix). (Warp
    // collective) log_prefix_val holds the lane's value calculated in
    // Phase 1.1.
    float s_val_lane = (lane < BLK) ? log_prefix_val : -INFINITY;
    float suffix_max_s = warp_suffix_max_32(s_val_lane);

    // Store SuffixMaxS[t]. Reuse the DX_tile buffer.
    float* SuffixMaxS_smem = warp_smem->DX_tile;
    if (lane < BLK) {
      SuffixMaxS_smem[lane] = suffix_max_s;
    }
    __syncwarp();

    // Initialize scaled PSS (Prefix Sum of X) in registers.

    float scaled_prefix_sum_X_lane = 0.0f;

    // c_lane (0 to 15) is the index within DH handled by this thread.
    const int c_lane = lane % 16;

    // Initialize scaling factor M for PSS (Uniform across warp)
    float M = -INFINITY;

// Iterate t from 1 to BLK-1.
#pragma unroll
    for (int t = 1; t < BLK; ++t) {
      int j = t - 1;
      // W_j = -log_prefix[j]. Read from SMEM.
      float W_j = -warp_smem->log_prefix[j];

      // Step 3.6.1: Robust update of PSS and M. (Uniform calculation)
      float M_new = fmaxf(M, W_j);

      float scale_PSS;
      if (isinf(M) && M < 0) {
        scale_PSS = 0.0f;
      } else {
        scale_PSS = __expf(M - M_new);
      }

      float scale_Xj;
      if (isinf(M_new) && M_new < 0) {
        scale_Xj = 0.0f;
      } else {
        scale_Xj = __expf(W_j - M_new);
      }

      // Update PSS (Optimized utilization)
      // Assuming DH_VAL<=16.
      if (c_lane < DH_VAL) {
        // Load X[c_lane, j]. Broadcast load.
        // X_tile_RM is padded (BLK_P).
        float X_cj = (float)warp_smem->X_tile_RM[c_lane * BLK_P + j];

        // Update PSS value in the lane's register.
        scaled_prefix_sum_X_lane =
            scaled_prefix_sum_X_lane * scale_PSS + safe_multiply(X_cj, scale_Xj);
      }

      M = M_new;

      // Step 3.6.2: Calculate Stable_dLogA[t]. (Optimized Utilization)
      float SuffixMaxS_t = SuffixMaxS_smem[t];
      float grad_t_scaled = 0.0f;

// Iterate over r >= t
// simultaneously.
#pragma unroll
      for (int r_base = t; r_base < BLK; r_base += 2) {
        // Determine 'r' for this thread. Lanes 0-15 handle r_base, 16-31 handle
        // r_base+1.
        int r_lane;
        if (lane & 16) {
          r_lane = r_base + 1;
        } else {
          r_lane = r_base;
        }

        bool active = (r_lane < BLK);

        // Calculate scaled_exp_r (Uniform within half-warp).
        float scaled_exp_r = 0.0f;
        if (active) {
          // S_r = log_prefix[r_lane]. Read from SMEM.
          float S_r = warp_smem->log_prefix[r_lane];
          scaled_exp_r = __expf(S_r - SuffixMaxS_t);
        }

        // Calculate (dC_r @ PSS) - Dot product using half-warp.
        float inner_sum = 0.0f;

        // Use c_lane (lane % 16).
        if (active && c_lane < DH_VAL) {
          // Load dC[c_lane, r_lane]. Conflict-free access (padded layout).
          float dC_cr = warp_smem->DC_tile_RM_F32[c_lane * BLK_P + r_lane];

          // Use PSS value from this lane's register.
          inner_sum = safe_multiply(dC_cr, scaled_prefix_sum_X_lane);
        }

        inner_sum = safe_multiply(inner_sum, scaled_exp_r);

// Reduce inner_sum within the half-warp (16 threads).
// Use full mask 0xffffffff as all threads must participate.
#pragma unroll
        for (int offset = 8; offset > 0; offset /= 2)
          inner_sum += __shfl_down_sync(0xffffffff, inner_sum, offset);

        // Lane 0 accumulates for r_base, Lane 16 accumulates for r_base+1.
        if ((lane % 16) == 0) {
          if (active) {
            grad_t_scaled += inner_sum;
          }
        }
      }

      // Combine results from both half-warps (Lane 0 and Lane 16).
      // participate.
      float grad_t_scaled_other = __shfl_sync(0xffffffff, grad_t_scaled, 16);

      if (lane == 0) {
        // Lane 0 adds the result from Lane 16.
        grad_t_scaled += grad_t_scaled_other;
      }

      // Step 3.6.3: Apply the final scaling factor: exp(M + SuffixMaxS_t).
      float final_scale;
      float log_scale = M + SuffixMaxS_t;

      // Handle potential NaN (0*Inf=0) and M=-Inf.
      if ((isnan(log_scale) && (isinf(M) || isinf(SuffixMaxS_t))) || (isinf(M) && M < 0)) {
        final_scale = 0.0f;
      } else {
        final_scale = __expf(log_scale);
      }

      if (lane == 0) {
        grad_t_scaled = safe_multiply(grad_t_scaled, final_scale);
      }

      // Broadcast the result from lane 0. (Warp collective)
      float final_dLogA_t = __shfl_sync(0xffffffff, grad_t_scaled, 0);

      // Step 3.6.4: Convert to dA[t] and accumulate.
      if (lane == t) {
        float contribution;
        if (final_dLogA_t == 0.0f) {
          contribution = 0.0f;
        } else {
          contribution = safe_multiply(final_dLogA_t, inv_a_val_lane);
        }

        dA_tile_reg += contribution;
      }
    }
  }  // End if (tile_valid)

  // 3.7 Final store of dCoeff (dA = dA_A + dA_V)
  if (tile_valid && lane < BLK) {
    if (l0 + lane < L) {
      dA_tile_ptr[lane] = dA_tile_reg;
    }
  }

  // [DSMEM] All CTAs must converge here.
  cute::cluster_sync();
}

//  Host-side Launcher (DSMEM Cluster Launch)
template <typename ElementT, typename ElementO, int DH_VAL = 16, int WARPS_PER_BLOCK>
void launch_btp_backward(const ElementT* coeff, const ElementT* x, const ElementO* dy,
                         const float* checkpoint_carry, ElementT* dx, float* dA, int B, int H,
                         int L, int k, cudaStream_t stream) {
  if (L <= 0 || B * H <= 0) return;

  constexpr int BLK = BL;
  constexpr int WPB = WARPS_PER_BLOCK;
  constexpr int threads_per_block = 32 * WPB;

  // Launch Configuration
  const int tiles_per_sequence = (L + BLK - 1) / BLK;
  const int total_blocks_x = (tiles_per_sequence + WPB - 1) / WPB;

  const int MAX_CLUSTER = 16;
  int cluster_x = (k == 2) ? std::min(total_blocks_x, MAX_CLUSTER) : 1;

  if (k == 2) {
    // Prefer portable sizes and ensure alignment for segmentation boundaries.
    const int candidates[] = {16, 8, 4, 2, 1};
    for (int c : candidates) {
      if (c <= cluster_x && (total_blocks_x % c == 0)) {
        cluster_x = c;
        break;
      }
    }
  }

  // Grid/Block/Cluster dimensions
  dim3 grid_dims(total_blocks_x, B * H);
  dim3 cluster_dims(cluster_x, 1, 1);
  dim3 block_dims(threads_per_block);

  // Calculate SMEM size
  using SmemLayout = BackwardSmemLayout<ElementT, DH_VAL, BLK, WPB>;
  const size_t smem_size = sizeof(SmemLayout);

  auto kernel_ptr = btp_backward_kernel<ElementT, ElementO, DH_VAL, WPB>;


  // Allow non-portable cluster sizes (if k=2)
  if (k == 2) {
    cudaError_t err1 =
        cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (err1 != cudaSuccess) {
      printf(
          "[BTP_BACKWARD LAUNCH ERROR] Failed to set non-portable cluster "
          "size: %s\n",
          cudaGetErrorString(err1));
      return;
    }
  }

  // Request dynamic shared memory
  cudaError_t err2 =
      cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  if (err2 != cudaSuccess) {
    printf(
        "[BTP_BACKWARD LAUNCH ERROR] Failed to set dynamic shared memory "
        "size (%zu bytes): %s\n",
        smem_size, cudaGetErrorString(err2));
    // Recommended Action: If SMEM is too large (e.g., WPB=32), try reducing WPB
    // (e.g., to 8) which significantly reduces SMEM usage and often improves
    // performance due to higher occupancy (Optimization 1).
    return;
  }

  // Launch Kernel using CUTLASS helper
  cutlass::launch_kernel_on_cluster({grid_dims, block_dims, cluster_dims, smem_size, stream},
                                    (const void*)kernel_ptr,
                                    // Kernel arguments:
                                    coeff, x, dy, checkpoint_carry, dx, dA, L, H, k,
                                    int(cluster_dims.x));  // Pass cluster size
}

//  Explicit Template Instantiations
using half = cutlass::half_t;
using bf16 = cutlass::bfloat16_t;

#define INSTANTIATE_BTP_BWD(T, O, DH_VAL, WPB)                                                  \
  template void launch_btp_backward<T, O, DH_VAL, WPB>(                                         \
      const T* coeff, const T* x, const O* dy, const float* checkpoint_carry, T* dx, float* dA, \
      int B, int H, int L, int k, cudaStream_t stream);

// Instantiating various WPB configurations. WPB=8 or 16 is often optimal for
// this kernel.
#define INSTANTIATE_ALL_WPB_BWD(T, O, DH_VAL) \
  INSTANTIATE_BTP_BWD(T, O, DH_VAL, 4)        \
  INSTANTIATE_BTP_BWD(T, O, DH_VAL, 8)        \
  INSTANTIATE_BTP_BWD(T, O, DH_VAL, 16)       \
  INSTANTIATE_BTP_BWD(T, O, DH_VAL, 32)

// Instantiations for DH=16
INSTANTIATE_ALL_WPB_BWD(half, float, 16)
INSTANTIATE_ALL_WPB_BWD(half, half, 16)
INSTANTIATE_ALL_WPB_BWD(bf16, float, 16)
INSTANTIATE_ALL_WPB_BWD(bf16, bf16, 16)

#undef INSTANTIATE_BTP_BWD
#undef INSTANTIATE_ALL_WPB_BWD

}  // namespace sss
