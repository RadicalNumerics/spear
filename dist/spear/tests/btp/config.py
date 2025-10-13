# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from spear.testing.numerics import (
    DEFAULT_DIMENSION_SWEEPS,
    DEFAULT_INPUT_DISTRIBUTION,
    HEAVY_TAIL_INPUT_DISTRIBUTION,
    DimensionSweep,
    DistributionSpec,
    InputDistributionConfig,
    NumericsConfig,
    ReferenceConfig,
)

DEFAULT_REFERENCES = (
    ReferenceConfig(
        label="two_stage_bf16",
        runner="two_stage",
        coeff_dtype=torch.bfloat16,
        x_dtype=torch.bfloat16,
    ),
    ReferenceConfig(
        label="two_stage_fp32",
        runner="two_stage",
        coeff_dtype=torch.float32,
        x_dtype=torch.float32,
    ),
    ReferenceConfig(
        label="log_two_stage_fp32",
        runner="log_two_stage",
        coeff_dtype=torch.float32,
        x_dtype=torch.float32,
    ),
)

DEFAULT_NUMERICS_CONFIG = NumericsConfig(
    kernel_dtype=torch.bfloat16,
    dimension_sweeps=DEFAULT_DIMENSION_SWEEPS,
    distributions=(DEFAULT_INPUT_DISTRIBUTION, HEAVY_TAIL_INPUT_DISTRIBUTION),
    references=DEFAULT_REFERENCES,
    samples_per_point=3,
    base_seed=1234,
    metric="max_abs",
)

__all__ = [
    "DistributionSpec",
    "InputDistributionConfig",
    "DimensionSweep",
    "ReferenceConfig",
    "NumericsConfig",
    "DEFAULT_DIMENSION_SWEEPS",
    "DEFAULT_INPUT_DISTRIBUTION",
    "HEAVY_TAIL_INPUT_DISTRIBUTION",
    "DEFAULT_REFERENCES",
    "DEFAULT_NUMERICS_CONFIG",
]
