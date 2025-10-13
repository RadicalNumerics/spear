# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""Numerics testing utilities facilitating harness-driven comparisons."""

from .config import (
    DEFAULT_DIMENSION_SWEEPS,
    DEFAULT_INPUT_DISTRIBUTION,
    HEAVY_TAIL_INPUT_DISTRIBUTION,
    DimensionSweep,
    DistributionSpec,
    InputDistributionConfig,
    NumericsConfig,
    ReferenceConfig,
)
from .metrics import compute_backward_metrics, compute_error_metrics, maybe_synchronize
from .sampling import make_generator, manual_seed, sample_tensor_from_spec

__all__ = [
    "DistributionSpec",
    "InputDistributionConfig",
    "DimensionSweep",
    "ReferenceConfig",
    "NumericsConfig",
    "DEFAULT_INPUT_DISTRIBUTION",
    "HEAVY_TAIL_INPUT_DISTRIBUTION",
    "DEFAULT_DIMENSION_SWEEPS",
    "manual_seed",
    "make_generator",
    "sample_tensor_from_spec",
    "compute_error_metrics",
    "compute_backward_metrics",
    "maybe_synchronize",
]
