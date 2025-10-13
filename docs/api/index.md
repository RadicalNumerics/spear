# API Reference

The API mirrors the three layers of the project. Conceptual overviews live in `docs/concepts`; this section links directly to the Python surface area so you can drop kernels and modules into your own models.

## `spear.nn`

High-level modules built on sliding window recurrences. The `Phalanx` layer wraps the BTP kernel with SigmoidA parametrisation, and `SigmoidA` is exposed separately when you only need the projections.

> See [Phalanx](../concepts/phalanx.md) for the design rationale.

::: spear.nn.phalanx.Phalanx
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: spear.nn.phalanx.SigmoidA
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## `spear.ops`

Low-level access to the BTP kernels and reference implementations. Use these when you need fine-grained control over tiling, dtype, or integration with custom autograd.

> Start with [Block Two-Pass](../concepts/btp.md) for an end-to-end explanation.

::: spear.ops.btp.btp
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

