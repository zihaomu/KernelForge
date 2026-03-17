# Online Softmax (GPU)

**Intent**: Compute `softmax(x)` in a numerically stable way while streaming through data once (or in few passes),
typically using warp/block reductions for `max` and `sum(exp(x - max))`.

**When It Works**
- Long vectors (e.g., attention scores) where memory bandwidth dominates.
- When fusing with surrounding ops (masking, scaling) reduces memory traffic.

**Recognition Signals (Code)**
- Two-stage reduction: compute `max`, then accumulate `exp(x - max)`.
- Warp/block reductions via shuffles or cooperative groups.
- Vectorized loads (`half2`, `float4`) and contiguous memory access.

**Tradeoffs**
- Must handle tails and masking carefully to avoid numerical issues.
- Register pressure can limit occupancy for very wide tiles.
