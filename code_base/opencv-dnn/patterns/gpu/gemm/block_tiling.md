# Block Tiling (GPU GEMM)

**Intent**: Improve data reuse by loading A/B tiles into shared memory and computing a C tile per CTA.

**When It Works**
- Medium/large GEMM sizes with sufficient arithmetic intensity.
- Input alignment allows vectorized loads.

**Recognition Signals (Code)**
- `__shared__` buffers for A/B tiles.
- Outer loops over K with `tile_k` and `__syncthreads()`.
- Per-thread fragments accumulate into registers.

**Tradeoffs / Failure Modes**
- Small-K or tiny matrices: shared memory overhead dominates.
- Bank conflicts and poor memory coalescing can erase gains.

**Agent Notes**
- Pair with `vectorized_load_store` and `double_buffer` when K is large.
