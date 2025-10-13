# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""Jagged-band (width-1) sequential inference for y_t = a_t y_{t-1} + u_t.

Implements block-local (zero-init) scan with length `block_len` and one-hop neighbor
injection between blocks by snapshotting the prior block's terminal local state.

Inputs:
- u: torch.Tensor (B, H, D, N)
- a: torch.Tensor (B, H, N) — raw gates unless use_log_a=True
- block_len: int
- use_log_a: bool (keyword-only)
- carry_init: Optional[torch.Tensor] (B, H, D)

Output:
- y_out: torch.Tensor (B, H, D, N)
"""

from dataclasses import dataclass

import torch

__all__ = [
    "jagged_band_infer_sequential",
    "JaggedBandState",
    "JagState",
    "step_jagged",
    "parallel_jagged",
]


@torch.no_grad()
def jagged_band_infer_sequential(
    u: torch.Tensor,  # (B, H, D, N)
    a: torch.Tensor,  # (B, H, N)  (raw gates unless use_log_a=True)
    block_len: int,
    *,
    use_log_a: bool = False,
    carry_init: (torch.Tensor | None) = None,  # (B, H, D), optional prior local carry for block -1
) -> torch.Tensor:
    """
    Jagged-band (width-1) sequential inference for y_t = a_t y_{t-1} + u_t,
    with block-local (zero-init) scan + one-hop neighbor injection.

    y_t = y_loc_t + r_pref_t * c_prev_loc
      where y_loc_t runs inside the block with zero init,
            r_pref_t = ∏_{k=block_start..t} a_k  (inclusive),
            c_prev_loc is the *frozen snapshot* of the previous block's y_loc at its last index.

    Returns:
      y_out: (B, H, D, N)
    """
    assert u.ndim == 4 and a.ndim == 3, "u: (B,H,D,N), a: (B,H,N)"
    B, H, D, N = u.shape
    assert a.shape == (B, H, N)

    gates = a.exp() if use_log_a else a  # (B,H,N)
    device, dtype = u.device, u.dtype

    y_out = torch.empty_like(u)

    c_prev_loc = (
        torch.zeros(B, H, D, device=device, dtype=dtype)
        if carry_init is None
        else carry_init.to(device=device, dtype=dtype).clone()
    )
    y_loc = torch.zeros(B, H, D, device=device, dtype=dtype)
    r_pref = torch.ones(B, H, 1, device=device, dtype=dtype)

    i_in_blk = 0

    for t in range(N):
        a_t = gates[:, :, t].unsqueeze(-1)  # (B,H,1)
        u_t = u[:, :, :, t]  # (B,H,D)

        y_loc = a_t * y_loc + u_t
        r_pref = a_t * r_pref
        y_out[:, :, :, t] = y_loc + r_pref * c_prev_loc

        i_in_blk += 1
        if i_in_blk == block_len:
            # Snapshot the carry BEFORE zeroing; do not alias.
            c_prev_loc = y_loc.clone()  # <-- FIX: clone to freeze the carry

            y_loc.zero_()
            r_pref.fill_(1.0)
            i_in_blk = 0

    return y_out


class JaggedBandState:
    """
    Streaming state for token-by-token inference.

    step() implements:
       y_loc  <- a_t * y_loc + u_t
       r_pref <- a_t * r_pref          (inclusive)
       y_t    <- y_loc + r_pref * c_prev_loc
    Snapshot c_prev_loc at block boundaries.

    If you want exclusive r_pref (product up to t-1), move the r_pref update after y_t is computed.
    """

    def __init__(self, B: int, H: int, D: int, block_len: int, device=None, dtype=None):
        self.block_len = block_len
        self.y_loc = torch.zeros(B, H, D, device=device, dtype=dtype)
        self.r_pref = torch.ones(B, H, 1, device=device, dtype=dtype)
        self.c_prev_loc = torch.zeros(B, H, D, device=device, dtype=dtype)
        self.i = 0

    @torch.no_grad()
    def reset_block(self):
        self.y_loc.zero_()
        self.r_pref.fill_(1.0)
        self.i = 0

    @torch.no_grad()
    def step(self, a_t: torch.Tensor, u_t: torch.Tensor, *, use_log_a: bool = False) -> torch.Tensor:
        """
        a_t : (B,H) or (B,H,1)   (raw or log per use_log_a)
        u_t : (B,H,D)
        """
        if a_t.ndim == 2:
            a_t = a_t.unsqueeze(-1)  # (B,H,1)
        a_t = a_t.exp() if use_log_a else a_t

        self.y_loc = a_t * self.y_loc + u_t
        self.r_pref = a_t * self.r_pref  # inclusive
        y_t = self.y_loc + self.r_pref * self.c_prev_loc

        self.i += 1
        if self.i == self.block_len:
            # Snapshot carry; do not alias before zeroing y_loc.
            self.c_prev_loc = self.y_loc.clone()  # <-- FIX
            self.reset_block()
        return y_t

@dataclass
class JagState:
    """
    Minimal streaming state for jagged-band (width-1) SSM.

    Fields are explicitly sized for per-token stepping:
      - h_loc: (B, H, D)  block-local accumulator
      - r_pref: (B, H)    inclusive product within current block (None if using log space)
      - c_prev_loc: (B, H, D) snapshot of previous block's terminal local state
      - i: int            in-block index for the next token (0..block_len-1)
      - block_len: int    fixed block length used during prefill
      - use_log_r: bool   if True, use log-space accumulation for r_pref
      - log_r_pref: (B, H) log-prefix (None if not using log-space)
    """

    h_loc: torch.Tensor
    r_pref: torch.Tensor | None
    c_prev_loc: torch.Tensor
    i: int
    block_len: int
    use_log_r: bool = False
    log_r_pref: torch.Tensor | None = None

    def clone_detached(self) -> "JagState":
        return JagState(
            h_loc=self.h_loc.clone(),
            r_pref=None if self.r_pref is None else self.r_pref.clone(),
            c_prev_loc=self.c_prev_loc.clone(),
            i=int(self.i),
            block_len=int(self.block_len),
            use_log_r=bool(self.use_log_r),
            log_r_pref=None if self.log_r_pref is None else self.log_r_pref.clone(),
        )


@torch.no_grad()
def step_jagged(
    A_t: torch.Tensor,  # (B, H)      a_t if use_log_a=False, log(a_t) if True
    X_t: torch.Tensor,  # (B, H, D)   driving term (e.g., B_t * V_t)
    C_t: torch.Tensor,  # (B, H, D)
    V_t: torch.Tensor,  # (B, H, D)
    state: JagState,
    *,
    use_log_a: bool = False,
) -> tuple[torch.Tensor, JagState]:
    """
    One-token decode step for jagged-band SSM using block-local scan semantics.

    Inputs:
      - A_t: (B, H)  coefficients per head; interpret as a_t if use_log_a=False, else log(a_t)
      - X_t: (B, H, D) fused driving term (e.g., B_t * V_t)
      - C_t: (B, H, D)
      - V_t: (B, H, D)

    Returns:
      - y_t: (B, H, D)
      - updated JagState
    """
    a_t = A_t.exp() if use_log_a else A_t  # (B,H)
    a_b = a_t.unsqueeze(-1)  # (B,H,1)

    h_loc = a_b * state.h_loc + X_t  # (B,H,D)

    if state.use_log_r or use_log_a:
        tiny = torch.finfo(a_t.dtype).tiny
        log_a = torch.log(a_t.clamp_min(tiny))
        prev_log_r = state.log_r_pref if state.log_r_pref is not None else torch.zeros_like(a_t)
        log_r = prev_log_r + log_a
        # Optionally cap exp for stability on very long runs
        r_pref = torch.exp(log_r)
    else:
        prev_r = state.r_pref if state.r_pref is not None else torch.ones_like(a_t)
        r_pref = a_t * prev_r

    y_core = h_loc + r_pref.unsqueeze(-1) * state.c_prev_loc  # (B,H,D)
    y_t = y_core * C_t + V_t

    i_next = state.i + 1
    if i_next == state.block_len:
        c_prev_loc = h_loc.clone()

        h_zero = torch.zeros_like(h_loc)
        if state.use_log_r or use_log_a:
            new_state = JagState(
                h_loc=h_zero,
                r_pref=None,
                c_prev_loc=c_prev_loc,
                i=0,
                block_len=state.block_len,
                use_log_r=True,
                log_r_pref=torch.zeros_like(A_t),  # log(1)=0
            )
        else:
            new_state = JagState(
                h_loc=h_zero,
                r_pref=torch.ones_like(A_t),
                c_prev_loc=c_prev_loc,
                i=0,
                block_len=state.block_len,
                use_log_r=False,
                log_r_pref=None,
            )
    else:
        if state.use_log_r or use_log_a:
            new_state = JagState(
                h_loc=h_loc,
                r_pref=None,
                c_prev_loc=state.c_prev_loc,
                i=i_next,
                block_len=state.block_len,
                use_log_r=True,
                log_r_pref=log_r,
            )
        else:
            new_state = JagState(
                h_loc=h_loc,
                r_pref=r_pref,
                c_prev_loc=state.c_prev_loc,
                i=i_next,
                block_len=state.block_len,
                use_log_r=False,
                log_r_pref=None,
            )

    return y_t, new_state


@torch.no_grad()
def parallel_jagged(
    A: torch.Tensor,  # (B, H, L)      a if use_log_a=False, log(a) if True
    X: torch.Tensor,  # (B, H, D, L)   driving term per token (e.g., B*V)
    C: torch.Tensor,  # (B, H, D, L)
    V: torch.Tensor,  # (B, H, D, L)
    *,
    block_len: int,
    use_log_a: bool = False,
) -> tuple[torch.Tensor, JagState]:
    """
    Parallel prompt (prefill) via block-local scans + one-hop neighbor injection.

    Returns:
      - Y: (B, H, D, L)
      - JagState representing the tail-of-prompt state for decode handoff
    """
    Bsz, H, D, L = V.shape

    a = A.exp() if use_log_a else A  # (B,H,L)

    pad = (-L) % block_len
    if pad:
        a = torch.nn.functional.pad(a, (0, pad), value=1.0)
        X = torch.nn.functional.pad(X, (0, pad), value=0.0)
        C = torch.nn.functional.pad(C, (0, pad), value=1.0)
        V = torch.nn.functional.pad(V, (0, pad), value=0.0)
    Lp = a.shape[-1]
    nblk = Lp // block_len

    h_loc = torch.zeros((Bsz, H, D, Lp), dtype=V.dtype, device=V.device)
    r_pref = torch.ones((Bsz, H, Lp), dtype=a.dtype, device=a.device)

    for bidx in range(nblk):
        s = bidx * block_len
        e = s + block_len
        a_blk = a[..., s:e]  # (B,H,BL)
        x_blk = X[..., s:e]  # (B,H,D,BL)

        h0 = torch.zeros((Bsz, H, D), dtype=V.dtype, device=V.device)
        rp = torch.ones((Bsz, H), dtype=a.dtype, device=a.device)
        for k in range(block_len):
            ak = a_blk[..., k]  # (B,H)
            xk = x_blk[..., k]  # (B,H,D)
            h0 = ak.unsqueeze(-1) * h0 + xk
            rp = rp * ak
            h_loc[..., s + k] = h0
            r_pref[..., s + k] = rp

    Y = torch.empty_like(V)
    c_prev = torch.zeros((Bsz, H, D), dtype=V.dtype, device=V.device)
    for bidx in range(nblk):
        s = bidx * block_len
        e = s + block_len
        Y[..., s:e] = h_loc[..., s:e] + r_pref[..., s:e].unsqueeze(-2) * c_prev.unsqueeze(-1)
        c_prev = h_loc[..., e - 1].clone()  # snapshot, no aliasing

    Y = Y * C + V

    if pad:
        Y = Y[..., :L]
        r_pref = r_pref[..., :L]
        h_loc = h_loc[..., :L]

    tail_i = (L - 1) % block_len

    h_tail = h_loc[..., L - 1].clone()  # (B,H,D)
    r_tail = r_pref[..., L - 1].clone()  # (B,H)
    if tail_i == block_len - 1:
        c_prev_tail = h_loc[..., L - 1].clone()
        r_tail = torch.ones_like(r_tail)
        i_next = 0
    else:
        n_completed = L // block_len
        if n_completed > 0:
            prev_end = n_completed * block_len - 1
            c_prev_tail = h_loc[..., prev_end].clone()
        else:
            c_prev_tail = torch.zeros_like(h_tail)
        i_next = tail_i + 1

    state = JagState(
        h_loc=h_tail,
        r_pref=r_tail,
        c_prev_loc=c_prev_tail,
        i=i_next,
        block_len=block_len,
        use_log_r=False,
        log_r_pref=None,
    )

    return Y, state
