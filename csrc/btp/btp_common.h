// Copyright 2025 Radical Numerics Inc.
//
// This source code is licensed under the Apache License, Version 2.0, found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cutlass/arch/mma.h>
#include <cutlass/gemm/warp/mma_tensor_op.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <cmath>  // For isnan, isinf, copysignf
#include <cute/tensor.hpp>

namespace sss {

using namespace cute;
constexpr int BL = 16;  // Tile size in timesteps
constexpr int DH = 16;  // Channel dimension

// Prevent 0*Inf=NaN in FP32 intermediates.
__device__ __forceinline__ float safe_multiply(float a, float b) {
  float result = a * b;
  // Check for NaN result, specifically from 0*Inf or Inf*0.
  if (isnan(result)) {
    if ((a == 0.0f && isinf(b)) || (isinf(a) && b == 0.0f)) {
      // Treat 0*Inf as 0. Maintain the sign if possible.
      return copysignf(0.0f, result);
    }
  }
  return result;
}

template <typename ElementT>
using WarpMmaPolicy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<cutlass::gemm::GemmShape<16, 8, 16>, 32, ElementT, cutlass::layout::RowMajor,
                       ElementT, cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor,
                       cutlass::arch::OpMultiplyAdd>,
    cutlass::MatrixShape<1, 1>>;

template <typename ElementT>
using WarpMma = cutlass::gemm::warp::MmaTensorOp<
    cutlass::gemm::GemmShape<16, 16, 16>, ElementT, cutlass::layout::RowMajor, ElementT,
    cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, WarpMmaPolicy<ElementT>>;

template <typename ElementT>
struct SharedMemoryLayout {
  static constexpr int BYTES_A = sizeof(ElementT) * BL * BL;
  static constexpr int BYTES_B = sizeof(ElementT) * BL * DH;
  static constexpr int BYTES_C = sizeof(float) * BL * DH;
  static constexpr int BYTES_CARRY = sizeof(float) * DH;

  static constexpr int BYTES_PER_WARP = BYTES_A + BYTES_B + BYTES_C + BYTES_CARRY;
};
__device__ __forceinline__ void cp_async_bulk_dsmem(uint32_t dst_dsmem_addr, uint32_t src_smem_addr,
                                                    uint32_t size, uint32_t remote_mbarrier_addr) {
  asm volatile(
      "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_"
      "tx::bytes [%0], [%1], %2, [%3];"
      :
      : "r"(dst_dsmem_addr), "r"(src_smem_addr), "r"(size), "r"(remote_mbarrier_addr)
      : "memory");
}

}  
