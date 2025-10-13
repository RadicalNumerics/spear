---
hide:
- navigation
- toc
---

<div>
<style type="text/css">
.md-typeset h1,
.md-content__button {
    display: none;
}
</style>
</div><p align="center">
  <img width=500 alt="Spear Logo" src="./assets/spear-logo.svg" />
</p>


SPEAR is a collection of kernels for AI model architectures developed by Radical Numerics.



## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

where `.[dev]` installs all dependencies for development mode; can be simplified to `uv pip install -e .`

### Caching

We use `ccache` by default. To use it and enable faster compilation (see explanation on the [vLLM docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#set-up-using-python-only-build-without-compilation:~:text=%2De%20.-,Tip,-Building%20from%20source)), run:
```bash
CCACHE_NOHASHDIR="true" uv pip install --no-build-isolation -e '.[dev]'
```


## Quick Start

```python
import torch
from spear.nn.phalanx import Phalanx

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

dim = 512  # Must be divisible by 16 (head_dim is fixed at 16)
length = 128
batch_size = 1024
layer = Phalanx(dim=dim, length=length, dtype=dtype).to(device)

x = torch.randn(batch_size, length, dim, dtype=dtype, device=device)
y = layer(x)
print(f"Input: {x.shape} -> Output: {y.shape}")
```

### Development

We include pre-commit hooks for linting and formatting (Python, C++, CUDA). To install:

```bash
uv run pre-commit install
```

To run (note they will be run automatically on commit, so not necessary to run manually):

```bash
uv run pre-commit run --all-files
```

To run tests

```bash
uv run pytest
```

## Structure

```
csrc/        # kernels: CUDA/C++ or other DSLs
spear/
├─ ops/      # low-level wrappers per op family
│  └─ <op>/
└─ nn/       # layers built from ops (parametrized)
   └─ <layer>/
```


## Target Architectures

Currently supported hardware includes compute capabilities 9.0 (Hopper) and 10.0 (Blackwell).

| Kernel Name       |  (NVIDIA) sm9.0 |  (NVIDIA) sm10.0 |  (NVIDIA) sm10.3 |
| ----------------- | :-----: | :-----: | :-----: |
| `swr.btp.fwd.bf16.bdl.hd16-bl16.sm90` | ✔︎ |  ~ |  ⛔| 
| `swr.btp.bwd.bf16.bdl.hd16-bl16.sm90`  | ✔︎ | ~ |  ⛔ | 

* ✔︎: optimized
* ~: working but not fully optimized
* ⛔: not available


---

<p align="center">
  <img width=350 alt="Radical Numerics Logo" src="./assets/rn-logo-desktop-vector.svg" />
</p>

