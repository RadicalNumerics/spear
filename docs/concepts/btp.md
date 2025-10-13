# Block Two-Pass (BTP)

Block Two-Pass (BTP) is the algorithmic core that realises sliding window recurrences on GPUs. It factors the recurrence into two sweeps that match the hardware: a local pass that consumes each tile entirely on chip, and a rank-1 global pass that stitches tiles together without serialising thread blocks.

## First Pass: Local Contractions

Inputs are reshaped into blocks of length `BL`, aligned with warp scheduling. Within a block we form a lower-triangular transfer matrix `L` from the coefficients and apply it to the activations with high-throughput GEMMs. Several numerically equivalent constructions exist; the production kernel uses a stable log-space variant, while the reference paths expose masked cumulative products and explicit double-precision formulations for testing.

## Second Pass: Carry Propagation

Once each block is solved locally, only the final column of `L` is needed to communicate with the next block. BTP multiplies that column by the cumulative gate at the boundary and performs a device-wide parallel rank-1 update. Because the update is diagonal at the top level, every block can run in parallel except for a single synchronisation, eliminating the carry chains that slow traditional scan implementations.

## Autograd and Compilation

`spear.ops.btp.interface` registers the compiled CUDA kernels as `torch.library` operators so that `torch.compile` can inline them. A custom autograd function wraps forward and backward, allocating checkpoints per block in float32 for numerical stability. The PyTorch-only references live alongside the CUDA binding and make it straightforward to validate gradients or experiment with alternative `L` constructions.

## Practical Advice

Choose the warps-per-block (`wpb`) parameter based on the recurrence order `k`: Hopper defaults to 32 warps for second-order recurrences and 8 for first-order. The kernel expects head width `DH=16` and sequence lengths that are multiples of the block size. When those assumptions are violated—in particular the length multiple—the Python wrappers pad internally so that downstream layers can keep simpler shapes.

