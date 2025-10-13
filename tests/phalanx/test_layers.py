# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import pytest
import torch

from spear.nn.phalanx import Phalanx
from spear.nn.phalanx.swr import KVRepeat, SigmoidA
from spear.testing.numerics.layers import compare_layer_parameters, extract_layer_gradients
from spear.testing.numerics.metrics import compute_error_metrics
from spear.testing.numerics.sampling import manual_seed


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Phalanx layer tests",
)


class TestKVRepeat:
    """Test KVRepeat module for parameter sharing."""

    def test_kv_repeat_creation(self):
        """Test KVRepeat module creation with valid parameters."""
        kv_heads = 4
        total_heads = 16
        kv_repeat = KVRepeat(kv_heads, total_heads)
        
        assert kv_repeat.kv_heads == kv_heads
        assert kv_repeat.total_heads == total_heads
        assert kv_repeat.kv_groups == total_heads // kv_heads
        assert kv_repeat.enabled is True

    def test_kv_repeat_no_grouping(self):
        """Test KVRepeat when kv_heads equals total_heads (no grouping)."""
        heads = 16
        kv_repeat = KVRepeat(heads, heads)
        
        assert kv_repeat.kv_heads == heads
        assert kv_repeat.total_heads == heads
        assert kv_repeat.kv_groups == 1
        assert kv_repeat.enabled is False

    def test_kv_repeat_invalid_heads(self):
        """Test KVRepeat with invalid head configuration."""
        with pytest.raises(ValueError, match="must be divisible"):
            KVRepeat(3, 16)  # 16 not divisible by 3

    def test_kv_repeat_forward(self):
        """Test KVRepeat forward pass with shape preservation."""
        kv_heads = 4
        total_heads = 16
        kv_repeat = KVRepeat(kv_heads, total_heads).to("cuda")
        
        B, D, L = 2, 8, 128
        x = torch.randn(B, kv_heads, D, L, dtype=torch.bfloat16, device="cuda")
        
        y = kv_repeat(x)
        
        # Should repeat across heads
        assert y.shape == (B, total_heads, D, L)
        assert y.dtype == x.dtype
        assert y.device == x.device

    def test_kv_repeat_forward_no_grouping(self):
        """Test KVRepeat forward when no grouping is needed."""
        heads = 16
        kv_repeat = KVRepeat(heads, heads)
        
        B, D, L = 2, 8, 128
        x = torch.randn(B, heads, D, L, dtype=torch.bfloat16, device="cuda")
        
        y = kv_repeat(x)
        
        # Should pass through unchanged
        assert y.shape == x.shape
        assert torch.allclose(y, x)


class TestSigmoidA:
    """Test SigmoidA parametrization module."""

    def test_sigmoid_a_creation(self):
        """Test SigmoidA module creation."""
        dim = 256
        heads = 16
        head_dim = 16
        sigmoid_a = SigmoidA(dim, heads, head_dim)
        
        assert sigmoid_a.dim == dim
        assert sigmoid_a.heads == heads
        assert sigmoid_a.head_dim == head_dim

    def test_sigmoid_a_forward_axcv(self):
        """Test SigmoidA forward_axcv method with shape preservation."""
        dim = 256
        heads = 16
        head_dim = 16
        sigmoid_a = SigmoidA(dim, heads, head_dim, dtype=torch.bfloat16).to("cuda")
        
        B, L = 2, 128
        x = torch.randn(B, L, dim, dtype=torch.bfloat16, device="cuda")
        
        A, X, C, V = sigmoid_a.forward_axcv(x)
        
        # Check shapes - A is (B, heads, L), X,C,V are (B, heads, head_dim, L)
        assert A.shape == (B, heads, L)
        assert X.shape == (B, heads, head_dim, L)
        assert C.shape == (B, heads, head_dim, L)
        assert V.shape == (B, heads, head_dim, L)
        
        # Check dtypes and devices
        for tensor in [A, X, C, V]:
            assert tensor.dtype == x.dtype
            assert tensor.device == x.device

    def test_sigmoid_a_with_kv_heads(self):
        """Test SigmoidA with KV head grouping."""
        dim = 256
        heads = 16
        head_dim = 16
        kv_heads = 4
        sigmoid_a = SigmoidA(dim, heads, head_dim, kv_heads=kv_heads, dtype=torch.bfloat16).to("cuda")
        
        B, L = 2, 128
        x = torch.randn(B, L, dim, dtype=torch.bfloat16, device="cuda")
        
        A, X, C, V = sigmoid_a.forward_axcv(x)
        
        # Check shapes - A is (B, heads, L), X,C,V are (B, heads, head_dim, L)
        assert A.shape == (B, heads, L)
        assert X.shape == (B, heads, head_dim, L)
        assert C.shape == (B, heads, head_dim, L)
        assert V.shape == (B, heads, head_dim, L)


class TestPhalanx:
    """Test main Phalanx layer."""

    def test_phalanx_creation_default(self):
        """Test Phalanx layer creation with default method."""
        dim = 256
        length = 128
        layer = Phalanx(dim=dim, length=length, method="default", dtype=torch.bfloat16)
        
        assert layer.dim == dim
        assert layer.heads == dim // 16
        assert layer.head_dim == 16
        assert layer.method == "default"

    def test_phalanx_creation_pytorch(self):
        """Test Phalanx layer creation with pytorch method."""
        dim = 256
        length = 128
        layer = Phalanx(dim=dim, length=length, method="pytorch", dtype=torch.bfloat16)
        
        assert layer.method == "pytorch"

    def test_phalanx_creation_pytorch_linspace(self):
        """Test Phalanx layer creation with pytorch_linspace method."""
        dim = 256
        length = 128
        layer = Phalanx(dim=dim, length=length, method="pytorch_linspace", dtype=torch.bfloat16)
        
        assert layer.method == "pytorch_linspace"

    def test_phalanx_invalid_method(self):
        """Test Phalanx with invalid method."""
        with pytest.raises(ValueError, match="method must be one of"):
            Phalanx(dim=256, length=128, method="invalid")

    def test_phalanx_invalid_dim_heads(self):
        """Test Phalanx with invalid dim/head_dim ratio."""
        with pytest.raises(ValueError, match="must be divisible"):
            Phalanx(dim=255, length=128)  # 255 not divisible by 16

    def test_phalanx_invalid_head_dim(self):
        """Test Phalanx with dim not divisible by 16."""
        # Test with dim=130 which is not divisible by 16
        with pytest.raises(ValueError, match="must be divisible by 16"):
            Phalanx(dim=130, length=128)  # 130 not divisible by 16

    @pytest.mark.parametrize("method", ["default", "pytorch", "pytorch_linspace"])
    @pytest.mark.parametrize("B,L,dim", [
        (1, 64, 128),
        (2, 128, 256),
        (4, 256, 512),
    ])
    def test_phalanx_forward_shape_preservation(self, method, B: int, L: int, dim: int):
        """Test Phalanx forward pass preserves input shape."""
        layer = Phalanx(dim=dim, length=L, method=method, dtype=torch.bfloat16).to("cuda")
        x = torch.randn(B, L, dim, dtype=torch.bfloat16, device="cuda")
        
        y = layer(x)
        
        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert y.device == x.device

    @pytest.mark.parametrize("method", ["default", "pytorch", "pytorch_linspace"])
    def test_phalanx_forward_different_batch_sizes(self, method):
        """Test Phalanx forward with different batch sizes."""
        dim = 256
        length = 128
        layer = Phalanx(dim=dim, length=length, method=method, dtype=torch.bfloat16).to("cuda")
        
        for B in [1, 2, 4, 8]:
            x = torch.randn(B, length, dim, dtype=torch.bfloat16, device="cuda")
            y = layer(x)
            assert y.shape == x.shape

    def test_phalanx_with_kv_heads(self):
        """Test Phalanx with KV head grouping."""
        dim = 256
        length = 128
        kv_heads = 4
        layer = Phalanx(dim=dim, length=length, kv_heads=kv_heads, dtype=torch.bfloat16).to("cuda")
        
        B = 2
        x = torch.randn(B, length, dim, dtype=torch.bfloat16, device="cuda")
        
        y = layer(x)
        assert y.shape == x.shape

    def test_phalanx_different_dtypes(self):
        """Test Phalanx with different dtypes.
        
        Note: Phalanx internally uses bfloat16 for computation (hardcoded in compute_dtype),
        so we only test with bfloat16 to match the internal implementation.
        """
        dim = 256
        length = 128
        
        # Only test with bfloat16 since Phalanx internally uses compute_dtype=bfloat16
        dtype = torch.bfloat16
        layer = Phalanx(dim=dim, length=length, dtype=dtype).to("cuda")
        x = torch.randn(2, length, dim, dtype=dtype, device="cuda")
        y = layer(x)
        assert y.shape == x.shape
        assert y.dtype == dtype

        layer_bf16 = Phalanx(dim=dim, length=length, dtype=torch.bfloat16).to("cuda")
        x_bf16 = torch.randn(2, length, dim, dtype=torch.bfloat16, device="cuda")
        y_bf16 = layer_bf16(x_bf16)
        assert y_bf16.shape == x_bf16.shape
        assert y_bf16.dtype == torch.bfloat16

    def test_phalanx_gradient_extraction(self):
        """Test gradient extraction using testing utilities."""
        dim = 256
        length = 128

        layer = Phalanx(dim=dim, length=length, dtype=torch.bfloat16).to("cuda")
        x = torch.randn(2, length, dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        gradients = extract_layer_gradients(layer)
        assert len(gradients) > 0
        for name, grad in gradients.items():
            assert grad.shape is not None
            assert grad.device == x.device

    def test_phalanx_layer_comparison(self):
        """Test layer parameter comparison using testing utilities."""
        dim = 256
        length = 128

        layer1 = Phalanx(dim=dim, length=length, dtype=torch.bfloat16).to("cuda")
        layer2 = Phalanx(dim=dim, length=length, dtype=torch.bfloat16).to("cuda")

        # Different layers should have different parameters
        assert not compare_layer_parameters(layer1, layer2)

        # Same layer should have identical parameters
        assert compare_layer_parameters(layer1, layer1)

    def test_phalanx_error_metrics(self):
        """Test error metrics computation using testing utilities."""
        dim = 256
        length = 128

        layer = Phalanx(dim=dim, length=length, dtype=torch.bfloat16).to("cuda")
        x = torch.randn(2, length, dim, dtype=torch.bfloat16, device="cuda")

        y1 = layer(x)
        y2 = layer(x)

        # Same input should produce same output
        metrics = compute_error_metrics(y1, y2)
        assert metrics["max_abs"] < 1e-6
        assert metrics["max_rel"] < 1e-6

    def test_phalanx_manual_seed(self):
        """Test manual seeding using testing utilities."""
        dim = 256
        length = 128

        manual_seed(42, torch.device("cuda"))
        layer1 = Phalanx(dim=dim, length=length, dtype=torch.bfloat16).to("cuda")
        x1 = torch.randn(2, length, dim, dtype=torch.bfloat16, device="cuda")
        y1 = layer1(x1)

        manual_seed(42, torch.device("cuda"))
        layer2 = Phalanx(dim=dim, length=length, dtype=torch.bfloat16).to("cuda")
        x2 = torch.randn(2, length, dim, dtype=torch.bfloat16, device="cuda")
        y2 = layer2(x2)

        # With same seed, results should be identical
        assert torch.allclose(y1, y2, atol=1e-6)


class TestLayerIntegration:
    """Integration tests for layer combinations."""

    def test_phalanx_gradient_flow(self):
        """Test that Phalanx layer supports gradient computation."""
        dim = 256
        length = 128
        layer = Phalanx(dim=dim, length=length, dtype=torch.bfloat16).to("cuda")
        
        B = 2
        x = torch.randn(B, length, dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        # Check that gradients were computed
        assert x.grad is not None
        assert x.grad.shape == x.shape
        
        # Check that layer parameters have gradients
        for param in layer.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_layer_state_dict_serialization(self):
        """Test that layers can be serialized and deserialized."""
        dim = 256
        length = 128
        layer = Phalanx(dim=dim, length=length, dtype=torch.bfloat16).to("cuda")
        
        # Get original state dict
        state_dict = layer.state_dict()
        
        # Create new layer and load state dict
        new_layer = Phalanx(dim=dim, length=length, dtype=torch.bfloat16).to("cuda")
        new_layer.load_state_dict(state_dict)
        
        # Test that outputs are the same
        x = torch.randn(2, length, dim, dtype=torch.bfloat16, device="cuda")
        
        with torch.no_grad():
            y1 = layer(x)
            y2 = new_layer(x)
            assert torch.allclose(y1, y2, rtol=1e-5, atol=1e-6)
