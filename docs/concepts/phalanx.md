# Phalanx Layer

Phalanx turns the BTP kernel into a drop-in PyTorch module. It couples the SigmoidA parametrisation with the sliding window recurrence to deliver a horizon-bounded mixer that works in hybrid transformer stacks.

## Parametrisation

Inputs pass through four projections: `proj_a` produces the per-head gates, `proj_b` and `proj_c` produce key and value gates for the recurrence, and `proj_v` yields the driving term. Gates are applied through elementwise sigmoids, and the key/value projections optionally share parameters through `KVRepeat`, which repeats key-value heads across query groups to reduce memory footprint without shrinking the attention space.

## Execution Paths

Phalanx exposes three forward paths: `method="default"` uses the CUDA BTP kernel (`spear.ops.btp.btp`), `method="pytorch"` runs the log-space reference for convenience or CPU execution, and `method="pytorch_linspace"` uses a masked cumulative-product variant that is useful for debugging numerical issues. All paths compute the same tensors `A`, `X`, `C`, and `V` before invoking the recurrence. After the BTP call, the output is combined with `C` and `V`, permuted back to batch-first layout, and projected to the model dimension via `proj_out`.

## Streaming Inference

`spear.nn.phalanx.inference` implements jagged-band sequential and per-token decoding under the same assumptions as training. A `JagState` stores the local accumulator, cumulative gate, and previous block carry so that decode can step one token at a time while staying consistent with the block layout used in prefill.

## Constraints and Tips

The layer fixes the head dimension to 16, so either supply `heads=dim//16` or leave `heads=None` and let Phalanx infer it. Sequence lengths are padded up to the next multiple of the kernel block size (default 16). Mixed-precision training works by keeping projections in the requested dtype while running the recurrence in `bfloat16` and accumulating in `float32` where needed. For hybrid models, pair Phalanx with attention layers that handle long-range routing; the SWR window deliberately truncates global dependencies so that the layer can run faster at high sequence lengths.

