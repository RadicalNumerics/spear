# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""
Single-entry BTP (Block Two-Pass) op that directly calls compiled extensions.
"""

from __future__ import annotations

import torch
import torch.library

# Import the compiled extension directly
import spear._btp

__all__ = ["btp", "btp_bidir"]

# -----------------------------------------------------------------------------
# Constants (must match kernels)
# -----------------------------------------------------------------------------
DH = 16
BLK = 16
SUPPORTED_WPB = [4, 8, 16, 32]
WPB_K1_DEFAULT = 8
WPB_K2_DEFAULT = 32

# -----------------------------------------------------------------------------
# Register C extensions as custom operators for torch.compile compatibility
# -----------------------------------------------------------------------------
# Define the custom operator library
lib = torch.library.Library("spear_btp", "DEF")


def register_traceable_op(
    lib: torch.library.Library,
    op_name: str,
    op_impl: callable,
    op_fake_impl: callable,
    *,
    mutates_args: str | None = None,
    tags: str | None = (),
    dispatch_key: str | None = "CUDA",
):
    """
    Register a traceable (`torch.compile` compatible) custom op
    """
    schema_str = torch.library.infer_schema(op_impl, mutates_args=mutates_args)
    lib.define(op_name + schema_str, tags=tags)
    lib.impl(op_name, op_impl, dispatch_key=dispatch_key)
    lib._register_fake(op_name, op_fake_impl)


def _forward_impl(
    coeff: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    checkpoint: torch.Tensor,
    k: int,
    wpb: int,
    reverse: bool = False,
) -> None:
    spear._btp.forward(coeff, x, y, checkpoint, k, wpb, reverse)


def _forward_impl_fake(
    coeff: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    checkpoint: torch.Tensor,
    k: int,
    wpb: int,
    reverse: bool = False,
) -> None:
    pass


def _backward_impl(
    coeff: torch.Tensor,
    x: torch.Tensor,
    dy: torch.Tensor,
    checkpoint: torch.Tensor,
    dx: torch.Tensor,
    dA: torch.Tensor,
    k: int,
    wpb: int,
    reverse: bool = False,
) -> None:
    spear._btp.backward(coeff, x, dy, checkpoint, dx, dA, k, wpb, reverse)


def _backward_impl_fake(
    coeff: torch.Tensor,
    x: torch.Tensor,
    dy: torch.Tensor,
    checkpoint: torch.Tensor,
    dx: torch.Tensor,
    dA: torch.Tensor,
    k: int,
    wpb: int,
    reverse: bool = False,
) -> None:
    pass


# Register forward and backward custom ops
register_traceable_op(
    lib,
    "forward",
    _forward_impl,
    _forward_impl_fake,
    mutates_args=["y", "checkpoint"],
    dispatch_key="CUDA",
)

register_traceable_op(
    lib,
    "backward",
    _backward_impl,
    _backward_impl_fake,
    mutates_args=["dx", "dA"],
    dispatch_key="CUDA",
)


# -----------------------------------------------------------------------------
# Autograd Function
# -----------------------------------------------------------------------------
class _BTPFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        coeff: torch.Tensor,
        x: torch.Tensor,
        k: int,
        output_dtype: torch.dtype,
        wpb: int,
        reverse: bool,
    ):
        B, H, dh_in, L = x.shape
        assert dh_in == DH, f"Channel dimension must be {DH}"
        assert L % BLK == 0, f"Length L must be multiple of {BLK}"
        assert wpb in SUPPORTED_WPB, f"WPB must be one of {SUPPORTED_WPB}"

        y = torch.empty_like(x, dtype=output_dtype)
        checkpoint = torch.empty((B, H, L // BLK, DH), device=x.device, dtype=torch.float32)

        torch.ops.spear_btp.forward(coeff, x, y, checkpoint, int(k), int(wpb), reverse)

        ctx.save_for_backward(coeff, x, checkpoint)
        ctx.k = int(k)
        ctx.wpb = int(wpb)
        ctx.reverse = reverse
        ctx.input_dtype = x.dtype
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        coeff, x, checkpoint = ctx.saved_tensors
        dy = dy.contiguous()

        dx = torch.empty_like(x, dtype=ctx.input_dtype)
        dA = torch.zeros_like(coeff, dtype=torch.float32)

        torch.ops.spear_btp.backward(coeff, x, dy, checkpoint, dx, dA, ctx.k, ctx.wpb, ctx.reverse)
        return dA, dx, None, None, None, None


def btp(
    coeff: torch.Tensor,
    x: torch.Tensor,
    k: int = 2,
    wpb: int | None = None,
    output_dtype: torch.dtype | None = None,
    reverse: bool = False,
) -> torch.Tensor:
    """
    Apply BTP (Block Two-Pass) operation.

    Args:
        coeff: Coefficient tensor of shape [B, H, L]
        x: Input tensor of shape [B, H, DH, L]
        k: Order parameter (1 or 2)
        wpb: Warps per block (default: 32 for k=2, 8 for k=1)
        output_dtype: Output dtype (default: float32)
        reverse: If True, compute anti-causal pass (read/write in reverse order)

    Returns:
        Output tensor of shape [B, H, DH, L]
    """
    out_dtype = output_dtype or torch.float32
    wpb_val = wpb if wpb is not None else (WPB_K2_DEFAULT if k == 2 else WPB_K1_DEFAULT)
    coeff = coeff.contiguous()
    x = x.contiguous()
    return _BTPFunction.apply(coeff, x, k, out_dtype, wpb_val, reverse)


# -----------------------------------------------------------------------------
# Bidirectional BTP
# -----------------------------------------------------------------------------
class _BTPBidirFunction(torch.autograd.Function):
    """Bidirectional BTP via two causal passes (forward + reversed)."""

    @staticmethod
    def forward(
        ctx,
        coeff_f: torch.Tensor,
        coeff_b: torch.Tensor,
        x: torch.Tensor,
        alpha: float,
        beta: float,
        k: int,
        output_dtype: torch.dtype,
        wpb: int,
    ):
        B, H, dh_in, L = x.shape
        assert dh_in == DH, f"Channel dimension must be {DH}"
        assert L % BLK == 0, f"Length L must be multiple of {BLK}"
        assert wpb in SUPPORTED_WPB, f"WPB must be one of {SUPPORTED_WPB}"

        # Forward causal pass: y = L_f @ x
        y = torch.empty_like(x, dtype=output_dtype)
        checkpoint_f = torch.empty((B, H, L // BLK, DH), device=x.device, dtype=torch.float32)
        torch.ops.spear_btp.forward(coeff_f, x, y, checkpoint_f, int(k), int(wpb), False)

        # Anti-causal pass: y_b = R @ L_b @ R @ x
        y_b = torch.empty_like(x, dtype=output_dtype)
        checkpoint_b = torch.empty((B, H, L // BLK, DH), device=x.device, dtype=torch.float32)
        torch.ops.spear_btp.forward(coeff_b, x, y_b, checkpoint_b, int(k), int(wpb), True)

        # Combine in-place: y = α·y_f + β·y_b - (α+β-1)·x
        if alpha == 1.0 and beta == 1.0:
            # Optimized path: y = y_f + y_b - x using in-place ops
            y += y_b
            y -= x.to(output_dtype)
        else:
            # General case with scalar multiplications
            y *= alpha
            y += beta * y_b
            y -= (alpha + beta - 1.0) * x.to(output_dtype)

        ctx.save_for_backward(coeff_f, coeff_b, x, checkpoint_f, checkpoint_b)
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.k = int(k)
        ctx.wpb = int(wpb)
        ctx.input_dtype = x.dtype
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        coeff_f, coeff_b, x, checkpoint_f, checkpoint_b = ctx.saved_tensors
        alpha, beta = ctx.alpha, ctx.beta
        k, wpb = ctx.k, ctx.wpb
        input_dtype = ctx.input_dtype

        dy = dy.contiguous()

        # Special-case α=β=1 to avoid scalar multiplications
        if alpha == 1.0 and beta == 1.0:
            # dy_f = dy, dy_b = dy (no scaling needed)
            # dx_identity = -dy (since -(1+1-1) = -1)

            # Forward causal branch gradients
            dx_f = torch.empty_like(x, dtype=input_dtype)
            dcoeff_f = torch.zeros_like(coeff_f, dtype=torch.float32)
            torch.ops.spear_btp.backward(coeff_f, x, dy, checkpoint_f, dx_f, dcoeff_f, k, wpb, False)

            # Anti-causal branch gradients
            dx_b = torch.empty_like(x, dtype=input_dtype)
            dcoeff_b = torch.zeros_like(coeff_b, dtype=torch.float32)
            torch.ops.spear_btp.backward(coeff_b, x, dy, checkpoint_b, dx_b, dcoeff_b, k, wpb, True)

            # Combine gradients: dx = -dy + dx_f + dx_b
            dx = dx_f + dx_b - dy.to(input_dtype)
        else:
            # General case with scalar multiplications
            # Gradient from identity correction term: dx_id = -(α+β-1)·dy
            dx_identity = -(alpha + beta - 1.0) * dy

            # Forward causal branch gradients: dy_f = α·dy
            dy_f = (alpha * dy).contiguous()
            dx_f = torch.empty_like(x, dtype=input_dtype)
            dcoeff_f = torch.zeros_like(coeff_f, dtype=torch.float32)
            torch.ops.spear_btp.backward(coeff_f, x, dy_f, checkpoint_f, dx_f, dcoeff_f, k, wpb, False)

            # Anti-causal branch gradients: dy_b = β·dy
            dy_b = (beta * dy).contiguous()
            dx_b = torch.empty_like(x, dtype=input_dtype)
            dcoeff_b = torch.zeros_like(coeff_b, dtype=torch.float32)
            torch.ops.spear_btp.backward(coeff_b, x, dy_b, checkpoint_b, dx_b, dcoeff_b, k, wpb, True)

            # Combine gradients
            dx = dx_identity.to(input_dtype) + dx_f + dx_b

        # Return gradients for: coeff_f, coeff_b, x, alpha, beta, k, output_dtype, wpb
        return dcoeff_f, dcoeff_b, dx, None, None, None, None, None


def btp_bidir(
    coeff_f: torch.Tensor,
    coeff_b: torch.Tensor | None,
    x: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    k: int = 2,
    wpb: int | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Apply Bidirectional BTP (Block Two-Pass) operation.

    Computes: y = α·L_f·x + β·R·L_b·R·x - (α+β-1)·x

    Where L_f is the causal operator, R is time-reversal, and L_b is the
    anti-causal operator applied via R·L_b·R.

    Args:
        coeff_f: Forward coefficient tensor of shape [B, H, L]
        coeff_b: Backward coefficient tensor of shape [B, H, L], or None to use coeff_f
        x: Input tensor of shape [B, H, DH, L]
        alpha: Forward direction weight (default: 1.0)
        beta: Backward direction weight (default: 1.0)
        k: Order parameter (1 or 2)
        wpb: Warps per block (default: 32 for k=2, 8 for k=1)
        output_dtype: Output dtype (default: float32)

    Returns:
        Output tensor of shape [B, H, DH, L]

    Special cases:
        - alpha=1, beta=1: Standard BD = C + A - I (identity-corrected sum)
        - alpha=0.5, beta=0.5: Symmetric average
        - alpha=1, beta=0: Pure causal (equivalent to btp())
        - alpha=0, beta=1: Pure anti-causal
    """
    out_dtype = output_dtype or torch.float32
    wpb_val = wpb if wpb is not None else (WPB_K2_DEFAULT if k == 2 else WPB_K1_DEFAULT)

    coeff_f = coeff_f.contiguous()
    coeff_b_eff = coeff_b if coeff_b is not None else coeff_f
    coeff_b_eff = coeff_b_eff.contiguous()
    x = x.contiguous()

    return _BTPBidirFunction.apply(coeff_f, coeff_b_eff, x, alpha, beta, k, out_dtype, wpb_val)
