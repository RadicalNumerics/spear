# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from spear.testing.numerics import (
    DEFAULT_INPUT_DISTRIBUTION,
    HEAVY_TAIL_INPUT_DISTRIBUTION,
    DimensionSweep,
    DistributionSpec,
    InputDistributionConfig,
    NumericsConfig,
    ReferenceConfig,
)

LAYER_DIMENSION_SWEEPS = (
    DimensionSweep("B", (1, 2, 4, 8), label="Batch Size"),
    DimensionSweep("dim", (128, 256, 512), label="Model Dimension"),
    DimensionSweep("heads", (8, 16, 32), label="Number of Heads"),
    DimensionSweep("L", (64, 128, 256, 512), label="Sequence Length"),
)

DEFAULT_LAYER_REFERENCES = (
    ReferenceConfig(
        label="pytorch",
        runner="pytorch",
    ),
    ReferenceConfig(
        label="pytorch_linspace",
        runner="pytorch_linspace",
    ),
)

DEFAULT_LAYER_NUMERICS_CONFIG = NumericsConfig(
    kernel_dtype=torch.bfloat16,
    dimension_sweeps=LAYER_DIMENSION_SWEEPS,
    distributions=(DEFAULT_INPUT_DISTRIBUTION, HEAVY_TAIL_INPUT_DISTRIBUTION),
    references=DEFAULT_LAYER_REFERENCES,
    samples_per_point=3,
    base_seed=5678,
    metric="max_abs",
)

__all__ = [
    "LAYER_DIMENSION_SWEEPS",
    "DEFAULT_LAYER_REFERENCES",
    "DEFAULT_LAYER_NUMERICS_CONFIG",
]
