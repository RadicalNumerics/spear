# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""Numerics runners for Phalanx layer."""

from __future__ import annotations

import torch

from spear.testing.numerics import (
    DEFAULT_INPUT_DISTRIBUTION,
    InputDistributionConfig,
    compute_error_metrics,
    make_generator,
    manual_seed,
    maybe_synchronize,
    sample_tensor_from_spec,
)

from .swr import Phalanx

__all__ = [
    "sample_layer_inputs",
    "create_phalanx_layers",
    "run_layer_forward",
    "compare_layer_outputs",
    "compare_layer_gradients",
]


def sample_layer_inputs(
    B: int,
    L: int,
    dim: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    seed: int | None = None,
    stress_scale: float = 1.0,
    distribution: InputDistributionConfig | None = None,
) -> torch.Tensor:
    """Generate input activations for Phalanx layer."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Phalanx layer requires a CUDA device; ensure tests skip earlier.")

    if seed is not None:
        manual_seed(seed, device)

    generator = make_generator(device, seed)
    dist_cfg = distribution or DEFAULT_INPUT_DISTRIBUTION

    x = sample_tensor_from_spec(
        dist_cfg.activation,
        (B, L, dim),
        dtype=dtype,
        generator=generator,
        device=device,
        scale_multiplier=stress_scale,
    )
    return x


def create_phalanx_layers(
    dim: int,
    length: int,
    *,
    methods: tuple[str, ...] = ("default", "pytorch", "pytorch_linspace"),
    dtype: torch.dtype = torch.bfloat16,
    k: int = 2,
    wpb: int = 32,
    output_dtype: torch.dtype | None = None,
    block_size: int = 16,
    kv_heads: int | None = None,
    device: torch.device | None = None,
    seed: int | None = None,
) -> dict[str, Phalanx]:
    """Create Phalanx layers with different methods but identical weights.
    
    Note: heads is automatically derived as dim // 16 (head_dim is fixed at 16).
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    layers = {}
    source_layer = None

    for method in methods:
        layer = Phalanx(
            dim=dim,
            length=length,
            method=method,
            dtype=dtype,
            k=k,
            wpb=wpb,
            output_dtype=output_dtype,
            block_size=block_size,
            kv_heads=kv_heads,
        ).to(device)

        if source_layer is None:
            source_layer = layer
        else:
            layer.load_state_dict(source_layer.state_dict())

        layers[method] = layer

    return layers


def run_layer_forward(
    layer: Phalanx,
    x: torch.Tensor,
    *,
    compile: bool = False,
    compile_opts: dict | None = None,
) -> torch.Tensor:
    """Execute forward pass through Phalanx layer."""

    if compile:
        compile_opts = compile_opts or {}
        layer_fn = torch.compile(layer, **compile_opts)
        return layer_fn(x)
    else:
        return layer(x)


def compare_layer_outputs(
    layers: dict[str, Phalanx],
    x: torch.Tensor,
    *,
    reference_method: str = "pytorch",
    compile: bool = False,
    compile_opts: dict | None = None,
) -> dict[str, dict[str, float]]:
    """Compare outputs from different layer methods against a reference."""

    if reference_method not in layers:
        raise ValueError(f"Reference method '{reference_method}' not in layers")

    outputs = {}
    for method, layer in layers.items():
        outputs[method] = run_layer_forward(layer, x, compile=compile, compile_opts=compile_opts)

    maybe_synchronize(x.device)

    reference_out = outputs[reference_method]
    metrics = {}

    for method, output in outputs.items():
        if method == reference_method:
            continue
        metrics[method] = compute_error_metrics(output, reference_out)

    return metrics


def compare_layer_gradients(
    layers: dict[str, Phalanx],
    x: torch.Tensor,
    *,
    grad_out: torch.Tensor | None = None,
    reference_method: str = "pytorch",
    compile: bool = False,
    compile_opts: dict | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare gradients through the full layer against a reference."""

    if reference_method not in layers:
        raise ValueError(f"Reference method '{reference_method}' not in layers")

    if grad_out is None:
        B, L, D = x.shape
        grad_out = torch.randn(B, L, D, dtype=torch.float32, device=x.device)

    device = x.device
    all_gradients = {}

    for method, layer in layers.items():
        for param in layer.parameters():
            param.grad = None

        x_copy = x.clone().detach().requires_grad_(True)

        y = run_layer_forward(layer, x_copy, compile=compile, compile_opts=compile_opts)
        torch.autograd.backward(y, grad_out)

        grads = {
            "x": x_copy.grad.detach().clone() if x_copy.grad is not None else torch.zeros_like(x_copy),
        }

        for name, param in layer.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().clone()
            else:
                grads[name] = torch.zeros_like(param)

        all_gradients[method] = grads

    maybe_synchronize(device)

    reference_grads = all_gradients[reference_method]
    metrics = {}

    for method, grads in all_gradients.items():
        if method == reference_method:
            continue

        method_metrics = {}
        for param_name, grad in grads.items():
            if param_name in reference_grads:
                method_metrics[param_name] = compute_error_metrics(grad, reference_grads[param_name])

        metrics[method] = method_metrics

    return metrics
