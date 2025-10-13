# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""Layer-specific testing utilities."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "copy_layer_weights",
    "compare_layer_parameters",
    "extract_layer_gradients",
]


def copy_layer_weights(source_layer: nn.Module, target_layer: nn.Module) -> None:
    """Copy weights from source layer to target layer."""
    target_layer.load_state_dict(source_layer.state_dict())


def compare_layer_parameters(layer1: nn.Module, layer2: nn.Module, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Verify that two layers have identical parameters."""
    state_dict1 = layer1.state_dict()
    state_dict2 = layer2.state_dict()

    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.allclose(state_dict1[key], state_dict2[key], rtol=rtol, atol=atol):
            return False

    return True


def extract_layer_gradients(layer: nn.Module) -> dict[str, torch.Tensor]:
    """Extract all gradients from a layer's parameters."""
    gradients = {}
    for name, param in layer.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.detach().clone()
        else:
            gradients[name] = torch.zeros_like(param)
    return gradients
