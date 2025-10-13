# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""
Phalanx layer supporting multiple computation methods.

Methods:
- default: BTP kernel with stable segsum computation
- pytorch: PyTorch implementation of block two-pass algorithm with stable segsum computation
- pytorch_linspace: pytorch implementation of block two-pass algorithm with linspace tiles

All methods use SigmoidA parametrization - sigmoid of W,Q for Phalanx-SWR with optional KV groups.
"""

import torch
import torch.nn as nn


class KVRepeat(nn.Module):
    """Modular KV group repeating module for efficient parameter sharing.

    This module handles the repeating of KV (B gate and C) tensors across groups,
    allowing for parameter-efficient attention mechanisms where multiple query heads
    share the same key-value parameters.
    """

    def __init__(self, kv_heads: int, total_heads: int):
        """
        Args:
            kv_heads: Number of key-value heads
            total_heads: Total number of heads (must be divisible by kv_heads)
        """
        super().__init__()

        if total_heads % kv_heads != 0:
            raise ValueError(f"total_heads ({total_heads}) must be divisible by kv_heads ({kv_heads})")

        self.kv_heads = kv_heads
        self.total_heads = total_heads
        self.kv_groups = total_heads // kv_heads
        self.enabled = kv_heads < total_heads
        
        # Pre-compute indices for KV expansion: [0,0,0,1,1,1,...]
        # Register as buffer so it moves with model to correct device
        if self.enabled:
            indices = torch.arange(total_heads) // self.kv_groups
            self.register_buffer('kv_indices', indices, persistent=False)

    def forward(self, tensor: torch.Tensor, pattern: str = "b h d l") -> torch.Tensor:
        """
        Repeat tensor across KV groups if needed.

        Args:
            tensor: Input tensor with shape matching pattern
            pattern: Einops pattern for input tensor (default: 'b h d l')

        Returns:
            Tensor repeated across groups if kv_groups > 1, otherwise unchanged
        """
        if not self.enabled:
            return tensor

        # Use cached indices for efficient expansion
        return tensor.index_select(1, self.kv_indices)

    def reshape_for_kv(self, tensor: torch.Tensor, batch: int, head_dim: int, length: int) -> torch.Tensor:
        """
        Reshape flat tensor to KV head dimensions.

        Args:
            tensor: Flat tensor of shape [B, kv_heads * head_dim, L]
            batch: Batch size
            head_dim: Dimension per head
            length: Sequence length

        Returns:
            Reshaped tensor of shape [B, kv_heads, head_dim, L]
        """
        return tensor.view(batch, self.kv_heads, head_dim, length)

    def expand_and_reshape(self, tensor: torch.Tensor, batch: int, head_dim: int, length: int) -> torch.Tensor:
        """
        Reshape and expand tensor from KV heads to all heads.

        Args:
            tensor: Flat tensor of shape [B, kv_heads * head_dim, L]
            batch: Batch size
            head_dim: Dimension per head
            length: Sequence length

        Returns:
            Expanded tensor of shape [B, total_heads, head_dim, L]
        """
        if not self.enabled:
            return tensor.view(batch, self.kv_heads, head_dim, length)
        
        # Optimized expansion using pre-computed indices
        # Reshape flat to [B, kv_heads, head_dim, L]
        tensor = tensor.view(batch, self.kv_heads, head_dim, length)
        # Use cached indices: [0,0,0,1,1,1,...] for kv_groups repeats
        return tensor.index_select(1, self.kv_indices)
    
    def expand_flat(self, tensor: torch.Tensor, head_dim: int) -> torch.Tensor:
        """
        Expand flat KV tensor directly without intermediate 4D reshape.
        More efficient for cases where we immediately flatten back.

        Args:
            tensor: Flat tensor of shape [B, kv_heads * head_dim, L]
            head_dim: Dimension per head

        Returns:
            Expanded flat tensor of shape [B, total_heads * head_dim, L]
        """
        if not self.enabled:
            return tensor
        
        B, _, L = tensor.shape
        # Reshape to separate heads: [B, kv_heads, head_dim, L]
        tensor = tensor.view(B, self.kv_heads, head_dim, L)
        # Expand using cached indices: [B, total_heads, head_dim, L]
        tensor = tensor.index_select(1, self.kv_indices)
        # Flatten back: [B, total_heads * head_dim, L]
        return tensor.view(B, self.total_heads * head_dim, L)


class SigmoidA(nn.Module):
    """Sigmoid-gated attention parametrization for SSM with optional KV groups."""

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        mixed_precision: bool = True,
        kv_heads: int | None = None,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.dim = dim
        self.dtype = dtype
        self.compute_dtype = torch.bfloat16

        self.kv_heads = kv_heads if kv_heads is not None else heads
        if heads % self.kv_heads != 0:
            raise ValueError(f"heads ({heads}) must be divisible by kv_heads ({self.kv_heads})")

        self.kv_repeat = KVRepeat(self.kv_heads, heads)

        self.dim_bvc = 3 * heads * head_dim
        self.dim_a = heads

        self.proj_a = nn.Linear(dim, heads, bias=True, dtype=dtype)

        kv_gate_dim = self.kv_heads * head_dim
        self.proj_b = nn.Linear(dim, kv_gate_dim, bias=True, dtype=dtype)
        self.proj_c = nn.Linear(dim, kv_gate_dim, bias=True, dtype=dtype)

        gate_dim = heads * head_dim
        self.proj_v = nn.Linear(dim, gate_dim, bias=False, dtype=dtype)

    def _compute_base_projections(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute base projections A, B_gate_kv, C_kv, V - shared by all forward methods."""
        x_compute = x.to(self.compute_dtype)
        A = self.proj_a(x_compute).transpose(1, 2)  # [B, heads, L]
        B_gate_kv = self.proj_b(x_compute).transpose(1, 2)  # [B, kv_heads*head_dim, L]
        C_kv = self.proj_c(x_compute).transpose(1, 2)  # [B, kv_heads*head_dim, L]
        V = self.proj_v(x_compute).transpose(1, 2)  # [B, heads*head_dim, L]
        return A, B_gate_kv, C_kv, V

    def forward_fused_gates(self, x: torch.Tensor) -> torch.Tensor:
        """Forward for fused gates kernel - returns packed tensor [B, 3*H*D + H, L]."""
        B, L, D = x.shape

        A, B_gate_kv, C_kv, V = self._compute_base_projections(x)

        if self.kv_repeat.enabled:
            B_gate = self.kv_repeat.expand_and_reshape(B_gate_kv, B, self.head_dim, L)
            B_gate = B_gate.reshape(B, self.heads * self.head_dim, L)
            C = self.kv_repeat.expand_and_reshape(C_kv, B, self.head_dim, L)
            C = C.reshape(B, self.heads * self.head_dim, L)
        else:
            B_gate = B_gate_kv
            C = C_kv

        out = torch.cat([B_gate, V, C, A], dim=1)  # [B, 3*H*D + H, L]

        return out

    def forward_axcv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward for default/pytorch kernels - returns (A, X, C, V) tensors.

        This method is used by both the default BTP kernel and pure PyTorch implementation
        since they both need the same tensor format: separate A, X, C, V tensors.
        """
        B, L, D = x.shape
        H, DH = self.heads, self.head_dim

        A_logits, B_gate_kv_logits, C_kv, V_flat = self._compute_base_projections(x)

        V = V_flat.view(B, H, DH, L)  # [B, H, DH, L]

        if self.kv_repeat.enabled:
            B_gate_logits = self.kv_repeat.expand_and_reshape(B_gate_kv_logits, B, DH, L)
            C = self.kv_repeat.expand_and_reshape(C_kv, B, DH, L)
        else:
            B_gate_logits = B_gate_kv_logits.view(B, H, DH, L)
            C = C_kv.view(B, H, DH, L)

        A = torch.sigmoid(A_logits)  # [B, H, L]
        B_gate = torch.sigmoid(B_gate_logits)  # [B, H, DH, L]

        X = B_gate * V  # [B, H, DH, L]

        return A, X, C, V


class Phalanx(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int | None = None,
        length: int = 2048,
        method: str = "default",
        dtype: torch.dtype = torch.float32,
        k: int = 2,
        wpb: int = 32,
        output_dtype: torch.dtype | None = None,
        block_size: int = 16,
        kv_heads: int | None = None,
    ):
        super().__init__()

        # Phalanx requires head_dim = 16, so heads = dim // 16
        if heads is None:
            if dim % 16 != 0:
                raise ValueError(f"dim ({dim}) must be divisible by 16")
            heads = dim // 16
        else:
            if dim % heads != 0:
                raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
            head_dim_check = dim // heads
            if head_dim_check != 16:
                raise ValueError(f"head_dim must be 16, got {head_dim_check}. Use heads=dim//16 or leave heads=None")

        if method not in ("default", "pytorch", "pytorch_linspace"):
            raise ValueError(f"method must be one of 'default', 'pytorch', 'pytorch_linspace', got {method}")

        self.dim = dim
        self.heads = heads
        self.head_dim = 16  # Always 16 for Phalanx
        self.length = (length + 15) // 16 * 16
        self.method = method
        self.dtype = dtype
        self.compute_dtype = torch.bfloat16
        self.k = k
        self.wpb = wpb
        self.output_dtype = output_dtype if output_dtype is not None else dtype
        self.block_size = block_size
        self.kv_heads = kv_heads if kv_heads is not None else heads
        mixed_precision = True

        self.param = SigmoidA(dim, heads, self.head_dim, dtype=dtype, kv_heads=kv_heads)

        self.proj_out = nn.Linear(dim, dim, bias=False, dtype=dtype)

        if method == "default":
            from spear.ops.btp import btp

            self.btp_module = btp
            self._forward_fn = self._forward_default

        elif method == "pytorch":
            from spear.ops.btp.reference import block_two_pass_log

            self.pytorch_block_two_pass = block_two_pass_log
            self._forward_fn = self._forward_pytorch

        elif method == "pytorch_linspace":
            from spear.ops.btp.reference import block_two_pass_linspace

            self.pytorch_block_two_pass = block_two_pass_linspace
            self._forward_fn = self._forward_pytorch

        else:
            raise ValueError(f"Invalid method: {method}. Supported methods: 'default', 'pytorch', 'pytorch_linspace'. ")

    def _forward_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using PyTorch implementation."""
        B, L, D = x.shape

        A, X, C, V = self.param.forward_axcv(x)  # A: [B, H, L], X,C,V: [B, H, D, L]

        y = self.pytorch_block_two_pass(A, X, self.block_size)  # [B, H, D, L]

        y = y * C + V  # [B, H, D, L]

        B_out, H, DH, L_out = y.shape
        y = y.permute(0, 3, 1, 2).reshape(B_out, L_out, H * DH)

        y = self.proj_out(y)

        return y

    def _forward_default(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using BTP kernel."""
        B, L, D = x.shape

        A, X, C, V = self.param.forward_axcv(x)  # A: [B, H, L], X,C,V: [B, H, D, L]

        y = self.btp_module(A, X, self.k, self.wpb, self.output_dtype)
        # if torch.distributed.is_initialized() and torch.cuda.is_available():
        #     torch.cuda.current_stream().synchronize()
        y = y * C + V  # [B, H, D, L]

        B_out, H, DH, L_out = y.shape
        y = y.permute(0, 3, 1, 2).reshape(B_out, L_out, H * DH)

        y = self.proj_out(y)

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Main forward pass - delegates to method-specific implementation."""
        return self._forward_fn(x)
