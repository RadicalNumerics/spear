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

__all__ = ["btp"]

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
) -> None:
    spear._btp.forward(coeff, x, y, checkpoint, k, wpb)


def _forward_impl_fake(
    coeff: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    checkpoint: torch.Tensor,
    k: int,
    wpb: int,
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
) -> None:
    spear._btp.backward(coeff, x, dy, checkpoint, dx, dA, k, wpb)


def _backward_impl_fake(
    coeff: torch.Tensor,
    x: torch.Tensor,
    dy: torch.Tensor,
    checkpoint: torch.Tensor,
    dx: torch.Tensor,
    dA: torch.Tensor,
    k: int,
    wpb: int,
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
    ):
        B, H, dh_in, L = x.shape
        assert dh_in == DH, f"Channel dimension must be {DH}"
        assert L % BLK == 0, f"Length L must be multiple of {BLK}"
        assert wpb in SUPPORTED_WPB, f"WPB must be one of {SUPPORTED_WPB}"

        y = torch.empty_like(x, dtype=output_dtype)
        checkpoint = torch.empty((B, H, L // BLK, DH), device=x.device, dtype=torch.float32)

        torch.ops.spear_btp.forward(coeff, x, y, checkpoint, int(k), int(wpb))

        ctx.save_for_backward(coeff, x, checkpoint)
        ctx.k = int(k)
        ctx.wpb = int(wpb)
        ctx.input_dtype = x.dtype
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        coeff, x, checkpoint = ctx.saved_tensors
        dy = dy.contiguous()

        dx = torch.empty_like(x, dtype=ctx.input_dtype)
        dA = torch.zeros_like(coeff, dtype=torch.float32)

        torch.ops.spear_btp.backward(coeff, x, dy, checkpoint, dx, dA, ctx.k, ctx.wpb)
        return dA, dx, None, None, None


def btp(
    coeff: torch.Tensor,
    x: torch.Tensor,
    k: int = 2,
    wpb: int | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Apply BTP (Block Two-Pass) operation.

    Args:
        coeff: Coefficient tensor of shape [B, H, L]
        x: Input tensor of shape [B, H, DH, L]
        k: Order parameter (1 or 2)
        wpb: Warps per block (default: 32 for k=2, 8 for k=1)
        output_dtype: Output dtype (default: float32)

    Returns:
        Output tensor of shape [B, H, DH, L]
    """
    out_dtype = output_dtype or torch.float32
    wpb_val = wpb if wpb is not None else (WPB_K2_DEFAULT if k == 2 else WPB_K1_DEFAULT)
    coeff = coeff.contiguous()
    x = x.contiguous()
    return _BTPFunction.apply(coeff, x, k, out_dtype, wpb_val)
