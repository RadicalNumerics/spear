// Copyright 2025 Radical Numerics Inc.
//
// This source code is licensed under the Apache License, Version 2.0, found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include <torch/extension.h>

#include "btp-backwards.cu"
#include "btp-forward.cu"

using cutlass::bfloat16_t;
using cutlass::half_t;

namespace {

static inline void check_common_forward(const torch::Tensor& coeff, const torch::Tensor& x,
                                        const torch::Tensor& y, const torch::Tensor& checkpoint,
                                        int64_t k, int64_t wpb) {
  TORCH_CHECK(coeff.is_cuda() && x.is_cuda() && y.is_cuda() && checkpoint.is_cuda(),
              "All tensors must be CUDA.");
  TORCH_CHECK(x.dim() == 4, "x must be [B, H, 16, L].");
  TORCH_CHECK(x.size(2) == 16, "Channel dim (DH) must be 16.");
  TORCH_CHECK(k == 1 || k == 2, "k must be 1 or 2.");
  TORCH_CHECK(wpb == 4 || wpb == 8 || wpb == 16 || wpb == 32, "Unsupported WPB value: ", wpb);

  TORCH_CHECK(coeff.scalar_type() == at::kHalf || coeff.scalar_type() == at::kBFloat16,
              "coeff must be float16 or bfloat16.");
  TORCH_CHECK(x.scalar_type() == coeff.scalar_type(), "x dtype must match coeff.");

  TORCH_CHECK(y.scalar_type() == at::kFloat || y.scalar_type() == x.scalar_type(),
              "y must be float32 or match x dtype.");

  TORCH_CHECK(checkpoint.scalar_type() == at::kFloat, "checkpoint must be float32.");
}

static inline void check_common_backward(const torch::Tensor& coeff, const torch::Tensor& x,
                                         const torch::Tensor& dy, const torch::Tensor& checkpoint,
                                         const torch::Tensor& dx, const torch::Tensor& dA,
                                         int64_t k, int64_t wpb) {
  TORCH_CHECK(coeff.is_cuda() && x.is_cuda() && dy.is_cuda() && checkpoint.is_cuda(),
              "coeff, x, dy, checkpoint must be CUDA.");
  TORCH_CHECK(dx.is_cuda() && dA.is_cuda(), "dx and dA must be CUDA.");
  TORCH_CHECK(x.dim() == 4, "x must be [B, H, 16, L].");
  TORCH_CHECK(x.size(2) == 16, "Channel dim (DH) must be 16.");
  TORCH_CHECK(k == 1 || k == 2, "k must be 1 or 2.");
  TORCH_CHECK(wpb == 4 || wpb == 8 || wpb == 16 || wpb == 32,
              "Unsupported WPB value for backward: ", wpb);

  TORCH_CHECK(coeff.scalar_type() == at::kHalf || coeff.scalar_type() == at::kBFloat16,
              "coeff must be float16 or bfloat16.");
  TORCH_CHECK(x.scalar_type() == coeff.scalar_type(), "x dtype must match coeff.");
  TORCH_CHECK(dx.scalar_type() == x.scalar_type(), "dx dtype must match x.");
  TORCH_CHECK(dA.scalar_type() == at::kFloat, "dA must be float32.");

  TORCH_CHECK(dy.scalar_type() == at::kFloat || dy.scalar_type() == x.scalar_type(),
              "dy must be float32 or match x dtype.");
}

#define DISPATCH_WPB_FWD(WPB_VAL, ElementT, ElementO)                                              \
  case WPB_VAL: {                                                                                  \
    sss::launch_block_two_pass<ElementT, ElementO, WPB_VAL>(coeff_ptr, x_ptr, y_ptr,               \
                                                            checkpoint_ptr, B, H, L, k32, stream); \
    break;                                                                                         \
  }

#define DISPATCH_WPB_BWD(WPB_VAL, ElementT, ElementO)                                    \
  case WPB_VAL: {                                                                        \
    sss::launch_btp_backward<ElementT, ElementO, 16, WPB_VAL>(                           \
        coeff_ptr, x_ptr, dy_ptr, checkpoint_ptr, dx_ptr, dA_ptr, B, H, L, k32, stream); \
    break;                                                                               \
  }

void btp_forward(torch::Tensor coeff, torch::Tensor x, torch::Tensor y, torch::Tensor checkpoint,
                 int64_t k, int64_t wpb) {
  check_common_forward(coeff, x, y, checkpoint, k, wpb);

  const int B = static_cast<int>(x.size(0));
  const int H = static_cast<int>(x.size(1));
  const int L = static_cast<int>(x.size(3));
  const int k32 = static_cast<int>(k);
  const int wpb32 = static_cast<int>(wpb);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  float* checkpoint_ptr = checkpoint.data_ptr<float>();

  if (coeff.scalar_type() == at::kHalf) {
    using ElementT = half_t;
    const ElementT* coeff_ptr = reinterpret_cast<const ElementT*>(coeff.data_ptr<at::Half>());
    const ElementT* x_ptr = reinterpret_cast<const ElementT*>(x.data_ptr<at::Half>());

    if (y.scalar_type() == at::kFloat) {
      using ElementO = float;
      ElementO* y_ptr = y.data_ptr<float>();

      switch (wpb32) {
        DISPATCH_WPB_FWD(4, ElementT, ElementO)
        DISPATCH_WPB_FWD(8, ElementT, ElementO)
        DISPATCH_WPB_FWD(16, ElementT, ElementO)
        DISPATCH_WPB_FWD(32, ElementT, ElementO)
        default:
          TORCH_CHECK(false, "Unsupported WPB value: ", wpb32);
      }
    } else {
      using ElementO = ElementT;
      ElementO* y_ptr = reinterpret_cast<ElementO*>(y.data_ptr<at::Half>());

      switch (wpb32) {
        DISPATCH_WPB_FWD(4, ElementT, ElementO)
        DISPATCH_WPB_FWD(8, ElementT, ElementO)
        DISPATCH_WPB_FWD(16, ElementT, ElementO)
        DISPATCH_WPB_FWD(32, ElementT, ElementO)
        default:
          TORCH_CHECK(false, "Unsupported WPB value: ", wpb32);
      }
    }
  } else {
    using ElementT = bfloat16_t;
    const ElementT* coeff_ptr = reinterpret_cast<const ElementT*>(coeff.data_ptr<at::BFloat16>());
    const ElementT* x_ptr = reinterpret_cast<const ElementT*>(x.data_ptr<at::BFloat16>());

    if (y.scalar_type() == at::kFloat) {
      using ElementO = float;
      ElementO* y_ptr = y.data_ptr<float>();

      switch (wpb32) {
        DISPATCH_WPB_FWD(4, ElementT, ElementO)
        DISPATCH_WPB_FWD(8, ElementT, ElementO)
        DISPATCH_WPB_FWD(16, ElementT, ElementO)
        DISPATCH_WPB_FWD(32, ElementT, ElementO)
        default:
          TORCH_CHECK(false, "Unsupported WPB value: ", wpb32);
      }
    } else {
      using ElementO = ElementT;
      ElementO* y_ptr = reinterpret_cast<ElementO*>(y.data_ptr<at::BFloat16>());

      switch (wpb32) {
        DISPATCH_WPB_FWD(4, ElementT, ElementO)
        DISPATCH_WPB_FWD(8, ElementT, ElementO)
        DISPATCH_WPB_FWD(16, ElementT, ElementO)
        DISPATCH_WPB_FWD(32, ElementT, ElementO)
        default:
          TORCH_CHECK(false, "Unsupported WPB value: ", wpb32);
      }
    }
  }
}

void btp_backward(torch::Tensor coeff, torch::Tensor x, torch::Tensor dy, torch::Tensor checkpoint,
                  torch::Tensor dx, torch::Tensor dA, int64_t k, int64_t wpb) {
  check_common_backward(coeff, x, dy, checkpoint, dx, dA, k, wpb);

  const int B = static_cast<int>(x.size(0));
  const int H = static_cast<int>(x.size(1));
  const int L = static_cast<int>(x.size(3));
  const int k32 = static_cast<int>(k);
  const int wpb32 = static_cast<int>(wpb);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  float* checkpoint_ptr = checkpoint.data_ptr<float>();
  float* dA_ptr = dA.data_ptr<float>();

  if (coeff.scalar_type() == at::kHalf) {
    using ElementT = half_t;
    const ElementT* coeff_ptr = reinterpret_cast<const ElementT*>(coeff.data_ptr<at::Half>());
    const ElementT* x_ptr = reinterpret_cast<const ElementT*>(x.data_ptr<at::Half>());
    ElementT* dx_ptr = reinterpret_cast<ElementT*>(dx.data_ptr<at::Half>());

    if (dy.scalar_type() == at::kFloat) {
      using ElementO = float;
      const ElementO* dy_ptr = dy.data_ptr<float>();

      switch (wpb32) {
        DISPATCH_WPB_BWD(4, ElementT, ElementO)
        DISPATCH_WPB_BWD(8, ElementT, ElementO)
        DISPATCH_WPB_BWD(16, ElementT, ElementO)
        DISPATCH_WPB_BWD(32, ElementT, ElementO)
        default:
          TORCH_CHECK(false, "Unsupported WPB value for backward: ", wpb32);
      }
    } else {
      using ElementO = ElementT;
      const ElementO* dy_ptr = reinterpret_cast<const ElementO*>(dy.data_ptr<at::Half>());

      switch (wpb32) {
        DISPATCH_WPB_BWD(4, ElementT, ElementO)
        DISPATCH_WPB_BWD(8, ElementT, ElementO)
        DISPATCH_WPB_BWD(16, ElementT, ElementO)
        DISPATCH_WPB_BWD(32, ElementT, ElementO)
        default:
          TORCH_CHECK(false, "Unsupported WPB value for backward: ", wpb32);
      }
    }
  } else {
    using ElementT = bfloat16_t;
    const ElementT* coeff_ptr = reinterpret_cast<const ElementT*>(coeff.data_ptr<at::BFloat16>());
    const ElementT* x_ptr = reinterpret_cast<const ElementT*>(x.data_ptr<at::BFloat16>());
    ElementT* dx_ptr = reinterpret_cast<ElementT*>(dx.data_ptr<at::BFloat16>());

    if (dy.scalar_type() == at::kFloat) {
      using ElementO = float;
      const ElementO* dy_ptr = dy.data_ptr<float>();

      switch (wpb32) {
        DISPATCH_WPB_BWD(4, ElementT, ElementO)
        DISPATCH_WPB_BWD(8, ElementT, ElementO)
        DISPATCH_WPB_BWD(16, ElementT, ElementO)
        DISPATCH_WPB_BWD(32, ElementT, ElementO)
        default:
          TORCH_CHECK(false, "Unsupported WPB value for backward: ", wpb32);
      }
    } else {
      using ElementO = ElementT;
      const ElementO* dy_ptr = reinterpret_cast<const ElementO*>(dy.data_ptr<at::BFloat16>());

      switch (wpb32) {
        DISPATCH_WPB_BWD(4, ElementT, ElementO)
        DISPATCH_WPB_BWD(8, ElementT, ElementO)
        DISPATCH_WPB_BWD(16, ElementT, ElementO)
        DISPATCH_WPB_BWD(32, ElementT, ElementO)
        default:
          TORCH_CHECK(false, "Unsupported WPB value for backward: ", wpb32);
      }
    }
  }
}

#undef DISPATCH_WPB_FWD
#undef DISPATCH_WPB_BWD

}  // anonymous namespace

// Optional torch library registration (kept empty)
// TORCH_LIBRARY(btp, m) {}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &btp_forward, "BTP forward (tiled, DSMEM/cluster)");
  m.def("backward", &btp_backward, "BTP backward (tiled, DSMEM/cluster)");
}
