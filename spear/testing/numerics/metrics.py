# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""Error metric helpers shared across numerics tests."""

from __future__ import annotations

import torch

__all__ = [
    "compute_error_metrics",
    "compute_backward_metrics",
    "maybe_synchronize",
]


def _to_float(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype.is_floating_point:
        return tensor.to(torch.float32)
    return tensor.float()


def compute_error_metrics(
    kernel_out: torch.Tensor,
    reference_out: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> dict[str, float]:
    """Return absolute and relative error statistics between two tensors."""

    ker = _to_float(kernel_out)
    ref = _to_float(reference_out)
    abs_diff = (ker - ref).abs()
    rel_diff = abs_diff / (ref.abs() + eps)
    sq_diff = abs_diff.square()
    return {
        "max_abs": abs_diff.max().item(),
        "mean_abs": abs_diff.mean().item(),
        "max_rel": rel_diff.max().item(),
        "mean_rel": rel_diff.mean().item(),
        "rmse": torch.sqrt(sq_diff.mean()).item(),
    }


def compute_backward_metrics(
    grad_kernel: torch.Tensor,
    grad_reference: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> dict[str, float]:
    """Convenience wrapper mirroring ``compute_error_metrics`` for gradients."""

    return compute_error_metrics(grad_kernel, grad_reference, eps=eps)


def maybe_synchronize(device: torch.device) -> None:
    """Synchronise CUDA work if needed to stabilise timing-dependent tests."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
