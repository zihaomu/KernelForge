# Sampling (Top-k / Top-p) (GPU)

**Intent**: Sample next tokens from logits efficiently, often combining top-k/top-p filtering with RNG.

**When It Works**
- Decoding loops with many steps; kernel launch overhead and memory traffic matter.

**Recognition Signals (Code)**
- RNG: `curand`, Philox counters, uniform samples
- Prefix-sum / cumulative probability for top-p
- Integration with topk selection

**Tradeoffs**
- Correct RNG stream management is critical (reproducibility vs speed).
- Numerical stability: softmax temperature and exp overflow.
