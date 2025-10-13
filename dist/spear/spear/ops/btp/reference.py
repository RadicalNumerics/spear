# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""Reference implementations for Block Two-Pass (BTP) kernels.

These PyTorch-only routines mirror the semantics of the CUDA kernels and are
used for numerics validation as well as gradient checks.
"""

from __future__ import annotations

import torch

from einops import rearrange, repeat

__all__ = ["block_two_pass_ref", "block_two_pass_log"]


def segsum(x):
    """Naive segment sum calculation. exp(segsum(A)) produces a 1-SS matrix."""
    L = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(L, L, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def construct_L_exp_segsum_log(a):
    """Construct L matrices for exp(segsum(log(a))) computation."""
    return torch.exp(segsum(torch.log(a)))


# More stable version in log space that computes partial sums from Tri Dao's code
# (https://github.com/state-spaces/mamba/blob/3d3f2d546824bb132193af316444504e075450c4/mamba_ssm/modules/ssd_minimal.py#L23)
def segsum_stable(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def construct_L_exp_segsum_log_stable(a):
    """Construct L matrices for exp(segsum(log(a))) computation."""
    return torch.exp(segsum_stable(torch.log(a)))


def construct_L_ref(a):
    """
    Construct L matrices using explicit product in double precision.
    L_ij = a_i * a_(i-1) * ... * a_(j+1) if i > j,
    L_ij = 1                             if i = j,
    L_ij = 0                             if i < j.

    Input:
    - a: (..., N) - tensor of log coefficients

    Output:
    - L: (..., N, N) - L matrices
    """
    a_type = a.dtype
    a = a.double()
    L = torch.zeros(a.shape[:-1] + (a.shape[-1], a.shape[-1]), device=a.device, dtype=a.dtype)
    for i in range(a.shape[-1]):
        for j in range(i):
            if i > j:
                L[..., i, j] = torch.prod(a[..., j + 1 : i + 1], dim=-1)
        L[..., i, i] = 1
    L = L.to(a_type)
    return L

def construct_L_masked_cumprod(
    x: torch.Tensor,
) -> torch.Tensor:
    """Build L via masked column-wise cumprod in linear space."""
    *batch, L = x.shape
    X = x.unsqueeze(-1).expand(*batch, L, L)
    strict_lower = torch.tril(torch.ones_like(X, dtype=torch.bool), diagonal=-1)
    X = X.masked_fill(~strict_lower, 1.0)
    P = torch.cumprod(X, dim=-2)
    upper = torch.triu(torch.ones_like(P, dtype=torch.bool), diagonal=1)
    L = P.masked_fill(upper, 0.0)
    return L

def construct_L_cumprod_ratio(x):
    """Build L via ratio of cumprod in linear space."""
    p = torch.cumprod(x, dim=-1)
    L = torch.where(
        p[..., None, :] != 0,
        p[..., None] / p[..., None, :],
        torch.zeros_like(p[..., None]),
    )
    mask = torch.tril(torch.ones_like(L, dtype=torch.bool))
    L = L.masked_fill(~mask, 0.0)
    return L


def block_two_pass(a, u, L_constructor, BL=16):
    """
    Two-pass convolution algorithm optimized for torch.compile and Triton.

    Input:
    - a: (B, H, N) - coefficients
    - u: (B, H, DH, N) - input tensor
    - L_constructor: function that constructs the L matrices
    - BL: block size

    Output:
    - x: (B, H, DH, N) - output tensor
    """

    u, a = [rearrange(x, "... (c l) -> ... c l", l=BL) for x in (u, a)]

    x = torch.zeros_like(u)

    # First pass: compute local block contractions
    L = L_constructor(a)
    v = torch.einsum("shbij,shdbj->shdbi", L, u)

    # Second pass: propagate rank-1 cross-chunk updates using previous carry
    x[:, :, :, 0, :] = v[:, :, :, 0, :]
    scale_factors = a[:, :, 1:, 0]  # (B, H, NBL-1)
    first_cols = L[:, :, 1:, :, 0]  # (B, H, NBL-1, BL)

    g = scale_factors[..., None] * first_cols  # (B, H, NBL-1, BL)
    prev_carry = v[:, :, :, :-1, -1][..., None]  # (B, H, DH, NBL-1, 1)
    carry_over = g[:, :, None] * prev_carry  # (B, H, DH, NBL-1, BL)

    x[:, :, :, 1:, :] = carry_over + v[:, :, :, 1:, :]

    x = rearrange(x, "... c l -> ... (c l)")
    return x


def block_two_pass_ref(a, u, BL=16):
    assert u.dtype == a.dtype == torch.float64, "u and a must be float64"
    return block_two_pass(a, u, construct_L_ref, BL)


def block_two_pass_log(a, u, BL=16):
    return block_two_pass(a, u, construct_L_exp_segsum_log_stable, BL)


def block_two_pass_linspace(a, u, BL=16):
    return block_two_pass(a, u, construct_L_masked_cumprod, BL)


__all__ = ["block_two_pass_ref", "block_two_pass_log", "block_two_pass_linspace"]
