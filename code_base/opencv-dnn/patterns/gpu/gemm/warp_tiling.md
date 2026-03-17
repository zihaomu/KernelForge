# Warp Tiling (GPU GEMM)

**Intent**: Map a tile of C to each warp to improve locality, reduce synchronization, and control register usage.

**When It Works**
- Large GEMM where a warp-level compute tile balances occupancy and reuse.
- Useful when block tiling alone hits register or shared memory limits.

**Recognition Signals (Code)**
- Warp-level indexing: `warp_id`, `lane_id`, `threadIdx.x / 32`.
- Each warp loads / computes its own sub-tiles.
- Warp shuffles (`__shfl_*`) for reductions or fragment exchange.

**Tradeoffs / Failure Modes**
- Too-large warp tiles inflate registers and reduce occupancy.
- Too-small warp tiles underutilize memory bandwidth.
