# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""Numerics runners for Block Two-Pass (BTP) kernels."""

from __future__ import annotations

from collections.abc import Callable

import torch

from spear.testing.numerics import (
    DEFAULT_INPUT_DISTRIBUTION,
    InputDistributionConfig,
    compute_backward_metrics,
    make_generator,
    manual_seed,
    maybe_synchronize,
    sample_tensor_from_spec,
)

from .interface import DH, btp
from .reference import block_two_pass_log, block_two_pass_ref

__all__ = [
    "sample_inputs",
    "run_kernel_forward",
    "run_reference",
    "run_reference_logbtp",
    "compare_backward_gradients",
    "RUNNERS",
    "get_runner",
]


def sample_inputs(
    B: int,
    H: int,
    L: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    seed: int | None = None,
    stress_scale: float = 1.0,
    distribution: InputDistributionConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate coefficients and activations suitable for the BTP kernel."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("BTP CUDA kernels require a CUDA device; ensure tests skip earlier.")

    if seed is not None:
        manual_seed(seed, device)

    generator = make_generator(device, seed)
    dist_cfg = distribution or DEFAULT_INPUT_DISTRIBUTION

    x = sample_tensor_from_spec(
        dist_cfg.activation,
        (B, H, DH, L),
        dtype=dtype,
        generator=generator,
        device=device,
        scale_multiplier=1.0,
    )
    coeff = sample_tensor_from_spec(
        dist_cfg.coeff,
        (H, L),
        dtype=dtype,
        generator=generator,
        device=device,
        scale_multiplier=stress_scale,
    )
    return coeff, x


def run_kernel_forward(
    coeff: torch.Tensor,
    x: torch.Tensor,
    *,
    k: int = 2,
    output_dtype: torch.dtype = torch.float32,
    wpb: int | None = None,
    compile: bool = False,
    compile_opts: dict = None,
) -> torch.Tensor:
    """Execute the compiled CUDA kernel via the functional ``btp`` API."""

    B = x.shape[0]
    coeff_batched = _broadcast_coeff(coeff, B).contiguous()

    btp_fn = btp
    if compile:
        compile_opts = compile_opts or {}
        btp_fn = torch.compile(btp, **compile_opts)

    return btp_fn(coeff_batched, x, k=k, wpb=wpb, output_dtype=output_dtype)


def _broadcast_coeff(coeff: torch.Tensor, batch: int) -> torch.Tensor:
    if coeff.dim() == 2:
        return coeff.unsqueeze(0).expand(batch, -1, -1)
    if coeff.shape[0] == batch:
        return coeff
    raise ValueError("coeff must be [H, L] or [B, H, L]")


def run_reference(coeff: torch.Tensor, x: torch.Tensor, BL: int = 16) -> torch.Tensor:
    """Gold-standard reference using double precision."""
    B = x.shape[0]
    a_batched = _broadcast_coeff(coeff, B)
    y = block_two_pass_ref(a_batched.double(), x.double(), BL)
    return y.float()


def run_reference_logbtp(coeff: torch.Tensor, x: torch.Tensor, BL: int = 16) -> torch.Tensor:
    """Log-space reference (training-style)."""
    B = x.shape[0]
    a_batched = _broadcast_coeff(coeff, B)
    y = block_two_pass_log(a_batched, x, BL)
    return y.float()


def compare_backward_gradients(
    coeff: torch.Tensor,
    x: torch.Tensor,
    *,
    grad_out: torch.Tensor | None = None,
    k: int = 2,
    output_dtype: torch.dtype = torch.float32,
    compile: bool = False,
    compile_opts: dict = None,
) -> dict[str, dict[str, float]]:
    """Compare backward gradients between the CUDA kernel and exact reference."""

    if grad_out is None:
        grad_out = torch.randn_like(x, dtype=output_dtype)
    else:
        grad_out = grad_out.to(output_dtype)

    device = x.device
    B = x.shape[0]

    btp_fn = btp
    if compile:
        compile_opts = compile_opts or {}
        btp_fn = torch.compile(btp, **compile_opts)

    coeff_kernel = _broadcast_coeff(coeff, B).contiguous().clone().detach().requires_grad_(True)
    x_kernel = x.clone().detach().requires_grad_(True)
    y_kernel = btp_fn(coeff_kernel, x_kernel, k=k, output_dtype=output_dtype)
    torch.autograd.backward(y_kernel, grad_out)
    grad_coeff_kernel = coeff_kernel.grad.detach().sum(dim=0)  # Sum over batch to match reference shape
    grad_x_kernel = x_kernel.grad.detach()

    coeff_ref = coeff.clone().detach().requires_grad_(True)
    x_ref = x.clone().detach().requires_grad_(True)
    y_ref = run_reference(coeff_ref, x_ref).to(output_dtype)
    (y_ref * grad_out).sum().backward()
    grad_coeff_ref = coeff_ref.grad.detach()
    grad_x_ref = x_ref.grad.detach()

    maybe_synchronize(device)

    return {
        "coeff": compute_backward_metrics(grad_coeff_kernel, grad_coeff_ref),
        "x": compute_backward_metrics(grad_x_kernel, grad_x_ref),
    }


RUNNERS: dict[str, Callable[..., torch.Tensor]] = {
    "kernel": run_kernel_forward,
    "reference_logbtp": run_reference_logbtp,
    "reference": run_reference,
}


def get_runner(name: str) -> Callable[..., torch.Tensor]:
    """Fetch a registered runner by name."""

    try:
        return RUNNERS[name]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unknown BTP runner '{name}'") from exc
