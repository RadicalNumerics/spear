# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""Sampling utilities used across numerics tests."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from .config import DistributionSpec

__all__ = [
    "manual_seed",
    "make_generator",
    "sample_tensor_from_spec",
]


def manual_seed(seed: int, device: torch.device) -> None:
    """Seed CPU and optionally CUDA RNGs to stabilise sampling."""

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def make_generator(device: torch.device, seed: int | None = None) -> torch.Generator:
    """Create a ``torch.Generator`` on ``device`` optionally initialised by ``seed``."""

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def sample_tensor_from_spec(
    spec: DistributionSpec,
    shape: Iterable[int] | tuple[int, ...],
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
    device: torch.device,
    scale_multiplier: float = 1.0,
) -> torch.Tensor:
    """Sample a tensor described by ``spec`` and return it as ``dtype``."""

    name = spec.name.lower()
    if name == "normal":
        std = spec.params.get("std", 1.0) * scale_multiplier
        data = torch.randn(tuple(shape), generator=generator, device=device, dtype=torch.float32) * std
        return data.to(dtype)
    if name == "sigmoid_normal":
        std = spec.params.get("std", 1.0) * scale_multiplier
        logits = torch.randn(tuple(shape), generator=generator, device=device, dtype=torch.float32) * std
        return torch.sigmoid(logits).to(dtype)
    if name == "uniform":
        low = spec.params.get("low", 0.0)
        high = spec.params.get("high", 1.0)
        tensor = torch.empty(tuple(shape), device=device, dtype=torch.float32)
        tensor.uniform_(low, high, generator=generator)
        return tensor.to(dtype)
    raise ValueError(f"Unsupported distribution '{spec.name}'")
