# Sliding Window Recurrences

Sliding window recurrences (SWRs) bound the communication horizon of linear recurrences so that they respect the memory hierarchy of modern accelerators. Instead of maintaining global state, an SWR tiles the sequence into warp-sized chunks, computes the dense recurrence locally, and exchanges only the rank-1 carry that the next chunk needs. The window is jagged: its shape mirrors how thread blocks advance across the sequence, giving every warp full bandwidth inside its tile while keeping inter-warp updates minimal.

## Why Locality First?

Classical recurrent operators mix information across the entire prefix of a sequence. Their transfer operator has exponentially decaying off-diagonal bands, which means long paths contribute almost nothing in practice. On GPUs, computing those paths anyway wastes bandwidth because the data has to visit the most expensive memory levels. SWRs embrace that decay. By truncating the operator to a jagged band, they conserve the useful parts of the recurrence while avoiding work that will numerically vanish, even in high precision.

## Tile-Level Semantics

Within each tile a recurrence is computed exactly, using dense tensor cores and a zero-initialised carry. When a tile finishes, it snapshots the terminal state, multiplies it by the cumulative gate for the border positions, and hands off a single rank-1 update to the next tile. This pattern yields depth-one synchronisation across thread blocks and keeps the critical path short enough for `torch.compile` and other graph compilers.

## Relationship to Other Mixers

SWRs sit between pure local operators (sliding-window attention, short convolutions) and global recurrences. They retain dense modelling inside each tile like a convolution, but unlike windowed attention they expose an explicit recurrence that can be truncated or extended depending on hardware limits. In practice SWRs pair well with sparse attention: attention handles long-range routing, SWRs provide efficient token mixing within the receptive field.

## Inference Modes

The same jagged structure works for streaming inference. During prefill, blocks run in the jagged layout described above. For decoding, a `JagState` tracks the local accumulator, the inclusive gate product, and the previous block's terminal state. Both paths share the same numerical assumptions, so training and inference stay aligned.

Refer to the BTP and Phalanx pages for the concrete algorithm and its PyTorch integration.

