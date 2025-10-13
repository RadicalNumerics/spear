# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from spear.ops.btp.runners import (
    compare_backward_gradients,
    run_kernel_forward,
    run_reference,
    run_reference_logbtp,
    sample_inputs,
)
from spear.testing.numerics import compute_error_metrics, maybe_synchronize

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for BTP numerics tests",
)

FORWARD_LIMITS = {
    "max_abs": 6e-2,
    "mean_abs": 4e-3,
    "max_rel": 1e4,
    "mean_rel": 1e-2,
    "rmse": 1.2e-2,
}

BACKWARD_LIMITS = {
    "max_abs": 5e-1,
    "mean_abs": 4e-2,
    "max_rel": 1e6,
    "mean_rel": 1.5e2,
    "rmse": 1.5e-1,
}


@dataclass
class CompileTestCfg:
    compile: bool = False
    compile_opts: dict | None = None


NO_COMPILE = CompileTestCfg()
DEFAULT_COMPILE = CompileTestCfg(compile=True)
FULLGRAPH = CompileTestCfg(compile=True, compile_opts={"fullgraph": True})
MAX_AUTOTUNE = CompileTestCfg(compile=True, compile_opts={"fullgraph": True, "mode": "max-autotune"})

COMPILE_TEST_CASES = [NO_COMPILE, DEFAULT_COMPILE, FULLGRAPH, MAX_AUTOTUNE]


def _assert_within(metrics: dict, limits: dict, context: str) -> None:
    for key, limit in limits.items():
        value = metrics[key]
        assert value <= limit, f"{context}: {key}={value:.3e} exceeds {limit:.3e}"


def _compile_id(arg: CompileTestCfg):
    should_compile, compile_opts = arg.compile, arg.compile_opts
    if should_compile:
        opt_str = ""
        if compile_opts is not None:
            opt_str = ",".join([f"{k}={v}" for k, v in compile_opts.items()])
            return "_".join(["compile", opt_str])
        return "compile"
    return "no_compile"


@pytest.mark.parametrize("compile_test_cfg", COMPILE_TEST_CASES, ids=_compile_id)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=lambda dt: str(dt).split(".")[-1])
def test_forward_consistency_basic(dtype: torch.dtype, compile_test_cfg: CompileTestCfg) -> None:
    B, H, L = 2, 8, 512
    coeff, x = sample_inputs(B, H, L, dtype=dtype, seed=123)

    kernel_out = run_kernel_forward(
        coeff.to(dtype),
        x.to(dtype),
        compile=compile_test_cfg.compile,
        compile_opts=compile_test_cfg.compile_opts,
    )

    reference_fp64 = run_reference(coeff.to(torch.float64), x.to(torch.float64))
    ref_log_two_stage_fp32 = run_reference_logbtp(coeff.to(torch.float32), x.to(torch.float32))
    _ = run_reference_logbtp(coeff.to(torch.bfloat16), x.to(torch.bfloat16))

    maybe_synchronize(x.device)

    metrics = {
        "two_stage_bf16": compute_error_metrics(kernel_out, reference_fp64),
        "two_stage_fp32": compute_error_metrics(kernel_out, ref_log_two_stage_fp32),
        "reference_float64": compute_error_metrics(kernel_out, reference_fp64),
    }

    for label, stats in metrics.items():
        _assert_within(stats, FORWARD_LIMITS, label)


@pytest.mark.parametrize(
    "B,H,L,stress",
    [
        (1, 1, 64, 3.0),
        (2, 4, 256, 2.0),
        (4, 8, 512, 3.5),
        (4, 16, 1024, 4.0),
    ],
)
def test_forward_stress_across_dimensions(B: int, H: int, L: int, stress: float) -> None:
    coeff, x = sample_inputs(B, H, L, seed=321, stress_scale=stress)

    kernel_out = run_kernel_forward(coeff, x)
    reference_fp64 = run_reference(coeff.to(torch.float64), x.to(torch.float64))

    maybe_synchronize(x.device)

    metrics = compute_error_metrics(kernel_out, reference_fp64)
    _assert_within(metrics, FORWARD_LIMITS, f"reference_fp64 dims=({B},{H},{L})")


@pytest.mark.parametrize("compile_test_cfg", COMPILE_TEST_CASES, ids=_compile_id)
def test_backward_consistency_basic(compile_test_cfg: CompileTestCfg) -> None:
    B, H, L, stress = 2, 8, 512, 1.0
    coeff, x = sample_inputs(B, H, L, seed=456, stress_scale=stress)

    grad_metrics = compare_backward_gradients(
        coeff,
        x,
        compile=compile_test_cfg.compile,
        compile_opts=compile_test_cfg.compile_opts,
    )

    maybe_synchronize(x.device)

    for param_name, metrics in grad_metrics.items():
        _assert_within(metrics, BACKWARD_LIMITS, f"grad_{param_name} dims=({B},{H},{L})")


@pytest.mark.parametrize(
    "B,H,L,stress",
    [
        (2, 8, 256, 2.5),
        (2, 8, 512, 3.0),
        (4, 8, 512, 3.5),
        (4, 8, 8192, 4.0),
    ],
)
@pytest.mark.parametrize("compile_test_cfg", COMPILE_TEST_CASES, ids=_compile_id)
def test_backward_stress_across_dimensions(B: int, H: int, L: int, stress: float, compile_test_cfg: CompileTestCfg) -> None:
    coeff, x = sample_inputs(B, H, L, seed=789, stress_scale=stress)

    grad_metrics = compare_backward_gradients(
        coeff,
        x,
        compile=compile_test_cfg.compile,
        compile_opts=compile_test_cfg.compile_opts,
    )

    maybe_synchronize(x.device)

    for param_name, metrics in grad_metrics.items():
        _assert_within(metrics, BACKWARD_LIMITS, f"grad_{param_name} dims=({B},{H},{L})")


def _get_exact_layer_parametrization_inputs(B: int, H: int, L: int, dim: int = 256):
    """
    Generate inputs using the EXACT same process as the Phalanx layer parametrization.
    This replicates what happens in the layer tests to expose the numerical bug.
    """
    from spear.nn.phalanx.runners import sample_layer_inputs, create_phalanx_layers
    
    x_layer = sample_layer_inputs(B, L, dim, dtype=torch.bfloat16, seed=789, stress_scale=1.0)
    layers = create_phalanx_layers(dim=dim, length=L, methods=("default",), seed=987)
    
    with torch.no_grad():
        A, X, _, _ = layers["default"].param.forward_axcv(x_layer)
    
    return A, X


def test_backward_exact_layer_parametrization() -> None:
    """
    Test backward pass with EXACT layer parametrization inputs.
    
    This uses the actual nn.Linear projections from the Phalanx layer to generate
    A and X coefficients, exactly matching what the layer tests use. 
    
    This test exposes a severe numerical bug in the BTP CUDA backward kernel:
    the kernel produces gradient errors of ~11 (22x over the limit of 0.5) when
    given the constrained distributions from actual layer parametrization.
    
    Expected result: FAIL with grad_coeff max_abs ~11 (limit: 0.5)
    """
    B, H, L, dim = 1, 16, 8192, 256

    A, X = _get_exact_layer_parametrization_inputs(B, H, L, dim)
    X = X.contiguous()

    grad_metrics = compare_backward_gradients(A, X, output_dtype=torch.bfloat16)

    maybe_synchronize(X.device)

    for param_name, metrics in grad_metrics.items():
        _assert_within(metrics, BACKWARD_LIMITS, f"exact_layer_grad_{param_name} dims=({B},{H},{L})")