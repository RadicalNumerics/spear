# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from spear.nn.phalanx.runners import (
    compare_layer_gradients,
    compare_layer_outputs,
    create_phalanx_layers,
    sample_layer_inputs,
)
from spear.testing.numerics import maybe_synchronize

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Phalanx layer numerics tests",
)


@pytest.fixture(scope="module", autouse=True)
def increase_compile_cache_limit():
    """Increase torch.compile cache limit for tests that compile multiple layer variants."""
    import torch._dynamo.config as dynamo_config
    
    old_cache_size_limit = dynamo_config.cache_size_limit
    dynamo_config.cache_size_limit = 64  # Increase from default 8 to 64
    
    yield
    
    dynamo_config.cache_size_limit = old_cache_size_limit

# NOTE: Phalanx layer requires head_dim = 16

LAYER_FORWARD_LIMITS = {
    "max_abs": 1e-1,
    "mean_abs": 1e-2,
    "max_rel": 1e5,
    "mean_rel": 5e-2,
    "rmse": 5e-2,
}

LAYER_BACKWARD_LIMITS = {
    "max_abs": 2.2,
    "mean_abs": 2.2e-1,
    "max_rel": 1e5,
    "mean_rel": 5e1,
    "rmse": 5e-1,
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
def test_layer_forward_consistency_basic(dtype: torch.dtype, compile_test_cfg: CompileTestCfg) -> None:
    B, L, dim = 2, 128, 256
    length = (L + 15) // 16 * 16

    x = sample_layer_inputs(B, L, dim, dtype=dtype, seed=123)

    layers = create_phalanx_layers(
        dim=dim,
        length=length,
        methods=("default", "pytorch", "pytorch_linspace"),
        dtype=dtype,
        seed=456,
    )

    metrics = compare_layer_outputs(
        layers,
        x,
        reference_method="pytorch",
        compile=compile_test_cfg.compile,
        compile_opts=compile_test_cfg.compile_opts,
    )

    maybe_synchronize(x.device)

    for method, stats in metrics.items():
        _assert_within(stats, LAYER_FORWARD_LIMITS, f"{method}_vs_pytorch")


@pytest.mark.parametrize(
    "B,L,dim,stress",
    [
        (1, 64, 128, 1.0),
        (2, 128, 256, 2.0),
        (4, 256, 512, 3.0),
    ],
)
def test_layer_forward_stress_across_dimensions(
    B: int, L: int, dim: int, stress: float
) -> None:
    length = (L + 15) // 16 * 16

    x = sample_layer_inputs(B, L, dim, dtype=torch.bfloat16, seed=321, stress_scale=stress)

    layers = create_phalanx_layers(
        dim=dim,
        length=length,
        methods=("default", "pytorch"),
        dtype=torch.bfloat16,
        seed=654,
    )

    metrics = compare_layer_outputs(layers, x, reference_method="pytorch")

    maybe_synchronize(x.device)

    for method, stats in metrics.items():
        _assert_within(stats, LAYER_FORWARD_LIMITS, f"{method}_vs_pytorch_dims=({B},{L},{dim})")


@pytest.mark.parametrize("compile_test_cfg", COMPILE_TEST_CASES, ids=_compile_id)
def test_layer_backward_consistency_basic(compile_test_cfg: CompileTestCfg) -> None:
    B, L, dim, stress = 2, 128, 256, 1.0

    length = (L + 15) // 16 * 16

    x = sample_layer_inputs(B, L, dim, dtype=torch.bfloat16, seed=789, stress_scale=stress)

    layers = create_phalanx_layers(
        dim=dim,
        length=length,
        methods=("default", "pytorch"),
        dtype=torch.bfloat16,
        seed=987,
    )

    grad_metrics = compare_layer_gradients(
        layers,
        x,
        reference_method="pytorch",
        compile=compile_test_cfg.compile,
        compile_opts=compile_test_cfg.compile_opts,
    )

    maybe_synchronize(x.device)

    failures = []
    for method, param_metrics in grad_metrics.items():
        for param_name, metrics in param_metrics.items():
            for key, limit in LAYER_BACKWARD_LIMITS.items():
                value = metrics[key]
                if value > limit:
                    failures.append(f"{method}_grad_{param_name}: {key}={value:.3e} exceeds {limit:.3e}")
    
    if failures:
        raise AssertionError(f"Gradient checks failed for dims=({B},{L},{dim}):\n" + "\n".join(failures))


@pytest.mark.parametrize(
    "B,L,dim,stress",
    [
        (2, 128, 256, 1.5),
        (2, 256, 512, 2.0),
        (4, 128, 256, 2.5),
    ],
)
@pytest.mark.parametrize("compile_test_cfg", COMPILE_TEST_CASES, ids=_compile_id)
def test_layer_backward_stress_across_dimensions(
    B: int, L: int, dim: int, stress: float, compile_test_cfg: CompileTestCfg
) -> None:
    length = (L + 15) // 16 * 16

    x = sample_layer_inputs(B, L, dim, dtype=torch.bfloat16, seed=111, stress_scale=stress)

    layers = create_phalanx_layers(
        dim=dim,
        length=length,
        methods=("default", "pytorch"),
        dtype=torch.bfloat16,
        seed=222,
    )

    grad_metrics = compare_layer_gradients(
        layers,
        x,
        reference_method="pytorch",
        compile=compile_test_cfg.compile,
        compile_opts=compile_test_cfg.compile_opts,
    )

    maybe_synchronize(x.device)

    failures = []
    for method, param_metrics in grad_metrics.items():
        for param_name, metrics in param_metrics.items():
            for key, limit in LAYER_BACKWARD_LIMITS.items():
                value = metrics[key]
                if value > limit:
                    failures.append(f"{method}_grad_{param_name}: {key}={value:.3e} exceeds {limit:.3e}")
    
    if failures:
        raise AssertionError(f"Gradient checks failed for dims=({B},{L},{dim}), stress={stress}:\n" + "\n".join(failures))


@pytest.mark.parametrize("kv_heads_divisor", [1, 2, 4])
def test_layer_kv_grouping(kv_heads_divisor: int) -> None:
    B, L, dim = 2, 128, 256
    length = (L + 15) // 16 * 16
    heads = dim // 16  # heads is derived from dim
    kv_heads = heads // kv_heads_divisor

    x = sample_layer_inputs(B, L, dim, dtype=torch.bfloat16, seed=333)

    layers = create_phalanx_layers(
        dim=dim,
        length=length,
        methods=("default", "pytorch"),
        dtype=torch.bfloat16,
        kv_heads=kv_heads,
        seed=444,
    )

    metrics = compare_layer_outputs(layers, x, reference_method="pytorch")

    maybe_synchronize(x.device)

    for method, stats in metrics.items():
        _assert_within(stats, LAYER_FORWARD_LIMITS, f"{method}_vs_pytorch_kv_heads={kv_heads}")


def test_layer_weight_sharing() -> None:
    dim, length = 256, 128

    layers = create_phalanx_layers(
        dim=dim,
        length=length,
        methods=("default", "pytorch", "pytorch_linspace"),
        dtype=torch.bfloat16,
        seed=555,
    )

    state_dict_ref = layers["pytorch"].state_dict()

    for method, layer in layers.items():
        if method == "pytorch":
            continue

        state_dict_test = layer.state_dict()

        for key in state_dict_ref.keys():
            assert key in state_dict_test, f"Missing key {key} in {method}"
            assert torch.allclose(
                state_dict_ref[key], state_dict_test[key], rtol=1e-6, atol=1e-6
            ), f"Weight mismatch for {key} in {method}"
