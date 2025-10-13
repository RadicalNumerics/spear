# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""Configuration objects for numerics harnesses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class DistributionSpec:
    """Describes how to sample a tensor by distribution name and parameters."""

    name: str
    params: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class InputDistributionConfig:
    """Couples activation and coefficient sampling distributions."""

    label: str
    activation: DistributionSpec
    coeff: DistributionSpec


@dataclass(frozen=True)
class DimensionSweep:
    """Collection of values explored for a single tensor dimension."""

    name: str
    values: Sequence[int]
    label: str | None = None


@dataclass(frozen=True)
class ReferenceConfig:
    """Reference implementation metadata used for comparisons."""

    label: str
    runner: str
    coeff_dtype: torch.dtype | None = None
    x_dtype: torch.dtype | None = None


@dataclass(frozen=True)
class NumericsConfig:
    """Top-level configuration for numerics sweeps."""

    kernel_dtype: torch.dtype
    output_dtype: torch.dtype = torch.float32
    k: int = 2
    base_seed: int = 1234
    samples_per_point: int = 1
    metric: str = "max_abs"
    dimension_sweeps: Sequence[DimensionSweep] = field(default_factory=tuple)
    distributions: Sequence[InputDistributionConfig] = field(default_factory=tuple)
    references: Sequence[ReferenceConfig] = field(default_factory=tuple)


DEFAULT_INPUT_DISTRIBUTION = InputDistributionConfig(
    label="standard",
    activation=DistributionSpec("normal", {"std": 1.0}),
    coeff=DistributionSpec("sigmoid_normal", {"std": 1.0}),
)

HEAVY_TAIL_INPUT_DISTRIBUTION = InputDistributionConfig(
    label="heavy_tail",
    activation=DistributionSpec("normal", {"std": 1.5}),
    coeff=DistributionSpec("sigmoid_normal", {"std": 3.0}),
)

DEFAULT_DIMENSION_SWEEPS = (
    DimensionSweep("B", (1, 2, 4, 8, 16), label="Batch Size"),
    DimensionSweep("H", (1, 2, 4, 8, 16, 32), label="Heads"),
    DimensionSweep("L", (64, 128, 256, 512, 1024, 2048), label="Sequence Length"),
)

__all__ = [
    "DistributionSpec",
    "InputDistributionConfig",
    "DimensionSweep",
    "ReferenceConfig",
    "NumericsConfig",
    "DEFAULT_INPUT_DISTRIBUTION",
    "HEAVY_TAIL_INPUT_DISTRIBUTION",
    "DEFAULT_DIMENSION_SWEEPS",
]
