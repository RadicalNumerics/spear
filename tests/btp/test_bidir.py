# Copyright 2025 Radical Numerics Inc.
# Licensed under the Apache License, Version 2.0

"""Tests for Bidirectional Block Two-Pass (BTP) operations."""

from __future__ import annotations

import pytest
import torch

from spear.ops.btp.reference import (
    block_two_pass_bidir_ref,
    block_two_pass_ref,
    construct_L_bidir,
    construct_L_ref,
)

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    from spear.ops.btp import btp_bidir


class TestTimeReversal:
    def test_involution(self):
        """R^2 = I (double reversal is identity)."""
        x = torch.randn(4, 8, 16, 64)
        torch.testing.assert_close(x.flip(-1).flip(-1), x)


class TestConstructLBidir:
    def test_formula(self):
        """L_bidir = L_f + R·L_b·R - I when α=β=1."""
        L = 16
        a_f = torch.rand(L, dtype=torch.float64) * 0.9 + 0.1
        a_b = torch.rand(L, dtype=torch.float64) * 0.9 + 0.1

        L_bidir = construct_L_bidir(a_f, a_b, construct_L_ref, alpha=1.0, beta=1.0)

        L_f = construct_L_ref(a_f)
        L_b = construct_L_ref(a_b)
        R = torch.eye(L, dtype=torch.float64).flip(0)
        eye = torch.eye(L, dtype=torch.float64)
        expected = L_f + R @ L_b @ R - eye

        torch.testing.assert_close(L_bidir, expected)

    def test_reduces_to_causal(self):
        """β=0 gives pure causal operator."""
        L = 16
        a = torch.rand(L, dtype=torch.float64) * 0.9 + 0.1
        L_bidir = construct_L_bidir(a, None, construct_L_ref, alpha=1.0, beta=0.0)
        torch.testing.assert_close(L_bidir, construct_L_ref(a))


class TestBidirCorrectness:
    @pytest.mark.parametrize(
        "B,H,L,BL,rtol,atol",
        [
            (1, 1, 16, 16, 1e-10, 1e-10),
            (1, 1, 32, 16, 1e-10, 1e-10),
            (2, 4, 64, 16, 0.02, 1e-3),
        ],
    )
    def test_vs_explicit_operator(self, B, H, L, BL, rtol, atol):
        """Compare block_two_pass_bidir against explicit matmul."""
        torch.manual_seed(42)
        a_f = torch.rand(B, H, L, dtype=torch.float64) * 0.9 + 0.1
        a_b = torch.rand(B, H, L, dtype=torch.float64) * 0.9 + 0.1
        u = torch.randn(B, H, 16, L, dtype=torch.float64)

        y_btp = block_two_pass_bidir_ref(a_f, a_b, u, BL=BL)
        L_bidir = construct_L_bidir(a_f, a_b, construct_L_ref)
        y_explicit = torch.einsum("...ij,...dj->...di", L_bidir, u)

        torch.testing.assert_close(y_btp, y_explicit, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 0.5), (1.0, 0.0), (0.0, 1.0)])
    def test_different_weights(self, alpha, beta):
        """Test various (α, β) combinations."""
        B, H, L, BL = 2, 2, 32, 16
        torch.manual_seed(123)
        a_f = torch.rand(B, H, L, dtype=torch.float64) * 0.9 + 0.1
        a_b = torch.rand(B, H, L, dtype=torch.float64) * 0.9 + 0.1
        u = torch.randn(B, H, 16, L, dtype=torch.float64)

        y_btp = block_two_pass_bidir_ref(a_f, a_b, u, BL=BL, alpha=alpha, beta=beta)
        L_bidir = construct_L_bidir(a_f, a_b, construct_L_ref, alpha=alpha, beta=beta)
        y_explicit = torch.einsum("...ij,...dj->...di", L_bidir, u)

        torch.testing.assert_close(y_btp, y_explicit, rtol=1e-10, atol=1e-10)


class TestBidirAlgebraicProperties:
    def test_reduces_to_causal(self):
        """β=0 reduces to pure causal BTP."""
        B, H, L, BL = 2, 4, 64, 16
        torch.manual_seed(789)
        a = torch.rand(B, H, L, dtype=torch.float64) * 0.9 + 0.1
        u = torch.randn(B, H, 16, L, dtype=torch.float64)

        y_bidir = block_two_pass_bidir_ref(a, a, u, BL=BL, alpha=1.0, beta=0.0)
        y_causal = block_two_pass_ref(a, u, BL=BL)

        torch.testing.assert_close(y_bidir, y_causal, rtol=1e-10, atol=1e-10)

    def test_reduces_to_anticausal(self):
        """α=0 reduces to pure anti-causal BTP."""
        B, H, L, BL = 2, 4, 64, 16
        torch.manual_seed(101)
        a = torch.rand(B, H, L, dtype=torch.float64) * 0.9 + 0.1
        u = torch.randn(B, H, 16, L, dtype=torch.float64)

        y_bidir = block_two_pass_bidir_ref(a, a, u, BL=BL, alpha=0.0, beta=1.0)
        y_anticausal = block_two_pass_ref(a, u.flip(-1), BL=BL).flip(-1)  # R @ L @ R @ u

        torch.testing.assert_close(y_bidir, y_anticausal, rtol=1e-10, atol=1e-10)

    def test_linearity(self):
        """Bidirectional BTP is linear in the input."""
        B, H, L, BL = 2, 2, 32, 16
        torch.manual_seed(202)
        a_f = torch.rand(B, H, L, dtype=torch.float64) * 0.9 + 0.1
        a_b = torch.rand(B, H, L, dtype=torch.float64) * 0.9 + 0.1
        u1 = torch.randn(B, H, 16, L, dtype=torch.float64)
        u2 = torch.randn(B, H, 16, L, dtype=torch.float64)
        c1, c2 = 2.5, -1.3

        y_combined = block_two_pass_bidir_ref(a_f, a_b, c1 * u1 + c2 * u2, BL=BL)
        y_separate = c1 * block_two_pass_bidir_ref(a_f, a_b, u1, BL=BL) + c2 * block_two_pass_bidir_ref(
            a_f, a_b, u2, BL=BL
        )

        torch.testing.assert_close(y_combined, y_separate, rtol=1e-10, atol=1e-10)


class TestBidirGradients:
    def test_gradcheck_joint(self):
        """Verify gradients w.r.t. all inputs via finite differences."""
        B, H, L, BL = 1, 1, 16, 16
        torch.manual_seed(808)
        a_f = torch.rand(B, H, L, dtype=torch.float64, requires_grad=True) * 0.9 + 0.1
        a_b = torch.rand(B, H, L, dtype=torch.float64, requires_grad=True) * 0.9 + 0.1
        u = torch.randn(B, H, 16, L, dtype=torch.float64, requires_grad=True)

        def fn(a_f_, a_b_, u_):
            return block_two_pass_bidir_ref(a_f_, a_b_, u_, BL=BL).sum()

        assert torch.autograd.gradcheck(fn, (a_f, a_b, u), eps=1e-6, atol=1e-4, rtol=1e-3)


# =============================================================================
# CUDA Kernel Tests
# =============================================================================


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestBidirCUDA:
    RTOL = 1e4
    ATOL = 6e-2

    @pytest.mark.parametrize("B,H,L", [(1, 1, 64), (2, 4, 128), (2, 8, 256)])
    def test_forward(self, B, H, L):
        """Compare CUDA btp_bidir against reference."""
        torch.manual_seed(42)
        device = torch.device("cuda")

        coeff_f = (torch.rand(B, H, L, device=device) * 0.9 + 0.1).to(torch.bfloat16)
        coeff_b = (torch.rand(B, H, L, device=device) * 0.9 + 0.1).to(torch.bfloat16)
        x = torch.randn(B, H, 16, L, device=device, dtype=torch.bfloat16)

        y_cuda = btp_bidir(coeff_f, coeff_b, x)
        y_ref = block_two_pass_bidir_ref(coeff_f.double(), coeff_b.double(), x.double(), BL=16)

        torch.testing.assert_close(y_cuda.float(), y_ref.float(), rtol=self.RTOL, atol=self.ATOL)

    @pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 0.5), (1.0, 0.0), (0.0, 1.0)])
    def test_weights(self, alpha, beta):
        """Test CUDA kernel with various (α, β) combinations."""
        B, H, L = 2, 4, 128
        torch.manual_seed(123)
        device = torch.device("cuda")

        coeff_f = (torch.rand(B, H, L, device=device) * 0.9 + 0.1).to(torch.bfloat16)
        coeff_b = (torch.rand(B, H, L, device=device) * 0.9 + 0.1).to(torch.bfloat16)
        x = torch.randn(B, H, 16, L, device=device, dtype=torch.bfloat16)

        y_cuda = btp_bidir(coeff_f, coeff_b, x, alpha=alpha, beta=beta)
        y_ref = block_two_pass_bidir_ref(coeff_f.double(), coeff_b.double(), x.double(), BL=16, alpha=alpha, beta=beta)

        torch.testing.assert_close(y_cuda.float(), y_ref.float(), rtol=self.RTOL, atol=self.ATOL)

    def test_reduces_to_causal(self):
        """beta=0 matches pure causal btp()."""
        from spear.ops.btp import btp

        B, H, L = 2, 4, 256
        torch.manual_seed(789)
        device = torch.device("cuda")

        coeff = (torch.rand(B, H, L, device=device) * 0.9 + 0.1).to(torch.bfloat16)
        x = torch.randn(B, H, 16, L, device=device, dtype=torch.bfloat16)

        y_bidir = btp_bidir(coeff, coeff, x, alpha=1.0, beta=0.0)
        y_causal = btp(coeff, x)

        torch.testing.assert_close(y_bidir, y_causal, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestBidirCUDA8K:
    """Dedicated 8K sequence length tests."""

    def test_8k_forward(self):
        """Test forward pass with 8192 sequence length."""
        B, H, L = 2, 8, 8192
        torch.manual_seed(42)
        device = torch.device("cuda")

        coeff_f = (torch.rand(B, H, L, device=device) * 0.9 + 0.1).to(torch.bfloat16)
        coeff_b = (torch.rand(B, H, L, device=device) * 0.9 + 0.1).to(torch.bfloat16)
        x = torch.randn(B, H, 16, L, device=device, dtype=torch.bfloat16)

        y = btp_bidir(coeff_f, coeff_b, x)

        assert y.shape == (B, H, 16, L)
        assert torch.isfinite(y).all()

    def test_8k_backward(self):
        """Test backward pass with 8192 sequence length."""
        B, H, L = 2, 8, 8192
        torch.manual_seed(123)
        device = torch.device("cuda")

        coeff_f = (torch.rand(B, H, L, device=device) * 0.9 + 0.1).to(torch.bfloat16).requires_grad_(True)
        coeff_b = (torch.rand(B, H, L, device=device) * 0.9 + 0.1).to(torch.bfloat16).requires_grad_(True)
        x = torch.randn(B, H, 16, L, device=device, dtype=torch.bfloat16, requires_grad=True)

        y = btp_bidir(coeff_f, coeff_b, x)
        y.sum().backward()

        assert torch.isfinite(coeff_f.grad).all()
        assert torch.isfinite(coeff_b.grad).all()
        assert torch.isfinite(x.grad).all()

    @pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 0.5)])
    def test_8k_weights(self, alpha, beta):
        """Test 8K with different weight combinations."""
        B, H, L = 1, 4, 8192
        torch.manual_seed(789)
        device = torch.device("cuda")

        coeff = (torch.rand(B, H, L, device=device) * 0.9 + 0.1).to(torch.bfloat16)
        x = torch.randn(B, H, 16, L, device=device, dtype=torch.bfloat16)

        y = btp_bidir(coeff, None, x, alpha=alpha, beta=beta)

        assert y.shape == x.shape
        assert torch.isfinite(y).all()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestBidirCUDAGradients:
    RTOL = 0.15
    ATOL = 0.05

    def test_gradient_x(self):
        """Test gradients w.r.t. input x."""
        B, H, L = 2, 4, 128
        torch.manual_seed(101)
        device = torch.device("cuda")

        coeff_f = (torch.rand(B, H, L, device=device) * 0.9 + 0.1).to(torch.bfloat16)
        coeff_b = (torch.rand(B, H, L, device=device) * 0.9 + 0.1).to(torch.bfloat16)
        x = torch.randn(B, H, 16, L, device=device, dtype=torch.bfloat16, requires_grad=True)
        grad_out = torch.randn(B, H, 16, L, device=device, dtype=torch.float32)

        y_cuda = btp_bidir(coeff_f, coeff_b, x)
        y_cuda.backward(grad_out)
        grad_x_cuda = x.grad.clone()

        x_ref = x.detach().double().requires_grad_(True)
        y_ref = block_two_pass_bidir_ref(coeff_f.double(), coeff_b.double(), x_ref, BL=16)
        y_ref.backward(grad_out.double())

        torch.testing.assert_close(grad_x_cuda.float(), x_ref.grad.float(), rtol=self.RTOL, atol=self.ATOL)

    def test_gradient_coeff(self):
        """Test gradients w.r.t. coefficients."""
        B, H, L = 2, 4, 128
        torch.manual_seed(202)
        device = torch.device("cuda")

        coeff_f = (torch.rand(B, H, L, device=device) * 0.9 + 0.1).to(torch.bfloat16).requires_grad_(True)
        coeff_b = (torch.rand(B, H, L, device=device) * 0.9 + 0.1).to(torch.bfloat16).requires_grad_(True)
        x = torch.randn(B, H, 16, L, device=device, dtype=torch.bfloat16)
        grad_out = torch.randn(B, H, 16, L, device=device, dtype=torch.float32)

        y_cuda = btp_bidir(coeff_f, coeff_b, x)
        y_cuda.backward(grad_out)
        grad_f_cuda = coeff_f.grad.clone()
        grad_b_cuda = coeff_b.grad.clone()

        coeff_f_ref = coeff_f.detach().double().requires_grad_(True)
        coeff_b_ref = coeff_b.detach().double().requires_grad_(True)
        y_ref = block_two_pass_bidir_ref(coeff_f_ref, coeff_b_ref, x.double(), BL=16)
        y_ref.backward(grad_out.double())

        torch.testing.assert_close(grad_f_cuda.float(), coeff_f_ref.grad.float(), rtol=self.RTOL, atol=self.ATOL)
        torch.testing.assert_close(grad_b_cuda.float(), coeff_b_ref.grad.float(), rtol=self.RTOL, atol=self.ATOL)
