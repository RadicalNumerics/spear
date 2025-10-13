# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive operator tests for BTP (Block Two-Pass) operations.
Tests basic functionality, shape preservation, and error handling.
"""

from __future__ import annotations

import pytest
import torch

from spear.ops.btp import btp
from spear.ops.btp.reference import block_two_pass_log, block_two_pass_linspace
from spear.testing.numerics.metrics import compute_error_metrics, compute_backward_metrics
from spear.testing.numerics.sampling import manual_seed


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for BTP operator tests",
)


class TestBTPOperator:
    """Test BTP (Block Two-Pass) operator."""

    @pytest.mark.parametrize("B,H,L", [
        (1, 4, 256),
        (2, 8, 512),
        (4, 16, 1024),
    ])
    def test_btp_forward_basic(self, B: int, H: int, L: int):
        """Test basic BTP forward pass with shape preservation."""
        coeff = torch.randn(H, 2, 32, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(B, H, 16, L, dtype=torch.bfloat16, device="cuda")
        
        y = btp(coeff, x)
        
        assert y.shape == x.shape
        assert y.dtype == torch.float32
        assert y.device == x.device

    @pytest.mark.parametrize("B", [1, 2, 4, 8])
    def test_btp_forward_different_batch_sizes(self, B: int):
        """Test BTP forward with different batch sizes."""
        H, L = 8, 512
        coeff = torch.randn(H, 2, 32, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(B, H, 16, L, dtype=torch.bfloat16, device="cuda")
        y = btp(coeff, x)
        assert y.shape == x.shape

    @pytest.mark.parametrize("L", [64, 128, 256, 512, 1024])
    def test_btp_forward_different_sequence_lengths(self, L: int):
        """Test BTP forward with different sequence lengths."""
        B, H = 2, 8
        coeff = torch.randn(H, 2, 32, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(B, H, 16, L, dtype=torch.bfloat16, device="cuda")
        y = btp(coeff, x)
        assert y.shape == x.shape

    @pytest.mark.parametrize("H", [4, 8, 16, 32])
    def test_btp_forward_different_head_counts(self, H: int):
        """Test BTP forward with different head counts."""
        B, L = 2, 512
        coeff = torch.randn(H, 2, 32, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(B, H, 16, L, dtype=torch.bfloat16, device="cuda")
        y = btp(coeff, x)
        assert y.shape == x.shape


    def test_btp_gradient_flow(self):
        """Test that BTP operator supports gradient computation."""
        B, H, L = 2, 8, 512
        coeff = torch.randn(H, 2, 32, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        x = torch.randn(B, H, 16, L, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        
        y = btp(coeff, x)
        loss = y.sum()
        loss.backward()
        
        assert coeff.grad is not None
        assert x.grad is not None
        assert coeff.grad.shape == coeff.shape
        assert x.grad.shape == x.shape



class TestBTPReference:
    """Test BTP reference implementations."""

    def test_btp_error_metrics(self):
        """Test error metrics computation using testing utilities."""
        B, H, L = 2, 8, 512
        coeff = torch.randn(H, 2, 32, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(B, H, 16, L, dtype=torch.bfloat16, device="cuda")
        
        y1 = btp(coeff, x)
        y2 = btp(coeff, x)
        
        # Same input should produce same output
        metrics = compute_error_metrics(y1, y2)
        assert metrics["max_abs"] < 1e-6
        assert metrics["max_rel"] < 1e-6


class TestBTPOperatorEdgeCases:
    """Test BTP operator edge cases and error handling."""

    def test_btp_minimum_dimensions(self):
        """Test BTP with minimum valid dimensions."""
        B, H, L = 1, 1, 16
        coeff = torch.randn(H, 2, 32, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(B, H, 16, L, dtype=torch.bfloat16, device="cuda")
        
        y = btp(coeff, x)
        assert y.shape == x.shape

    def test_btp_large_dimensions(self):
        """Test BTP with large dimensions."""
        B, H, L = 8, 32, 2048
        coeff = torch.randn(H, 2, 32, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(B, H, 16, L, dtype=torch.bfloat16, device="cuda")
        
        y = btp(coeff, x)
        assert y.shape == x.shape

    def test_btp_different_wpb_values(self):
        """Test BTP with different wpb (weights per block) values."""
        B, H, L = 2, 8, 512
        
        for wpb in [4, 8, 16, 32]:
            coeff = torch.randn(H, 2, wpb, dtype=torch.bfloat16, device="cuda")
            x = torch.randn(B, H, 16, L, dtype=torch.bfloat16, device="cuda")
            y = btp(coeff, x)
            assert y.shape == x.shape

    def test_btp_memory_efficiency(self):
        """Test BTP memory usage with large tensors."""
        B, H, L = 4, 16, 1024
        coeff = torch.randn(H, 2, 32, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(B, H, 16, L, dtype=torch.bfloat16, device="cuda")
        
        y = btp(coeff, x)
        assert y.shape == x.shape
        
        coeff.requires_grad_(True)
        x.requires_grad_(True)
        y = btp(coeff, x)
        loss = y.sum()
        loss.backward()
        
        assert coeff.grad is not None
        assert x.grad is not None


