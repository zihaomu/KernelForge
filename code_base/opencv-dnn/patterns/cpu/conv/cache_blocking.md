# Cache Blocking (CPU ARM)

**Intent**: Improve cache locality by blocking loops over channels/spatial tiles and reusing weights/inputs from cache.

**When It Works**
- Larger convolutions / GEMM-like paths (im2col+GEMM).
- When memory bandwidth is the limiter.

**Recognition Signals (Code)**
- Outer loops with block sizes (e.g., `ic_block`, `oc_block`, `tile_h`, `tile_w`).
- Explicit packing of weights/activations.
- Comments or variables referencing L1/L2.

**Tradeoffs**
- Wrong block sizes can regress due to TLB/cache thrash.
