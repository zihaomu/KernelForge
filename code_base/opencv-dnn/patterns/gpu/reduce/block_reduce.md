# Block Reduce (GPU)

**Intent**: Efficiently reduce values within a block (sum/max/min/argmax-like) using warp-level primitives and shared memory.

**When It Works**
- Reductions over moderately large contiguous dimensions.
- As a building block for softmax, layernorm, topk, and attention.

**Recognition Signals (Code)**
- CUB: `cub::BlockReduce`, `cub::DeviceReduce`
- Warp shuffles + shared memory staging
- Two-level reduction: per-warp then across warps

**Tradeoffs**
- Register pressure and shared memory usage can cap occupancy.
- For tiny reductions, overhead dominates.
