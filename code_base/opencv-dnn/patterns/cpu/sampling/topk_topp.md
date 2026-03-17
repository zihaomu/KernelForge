# Sampling (Top-k / Top-p) (CPU ARM)

**Intent**: Efficient sampling from logits on CPU, often combining top-k/top-p filtering with RNG.

**When It Works**
- CPU-only inference or small batch decoding.

**Recognition Signals (Code)**
- RNG and cumulative probability logic
- Partial sort/select for top-k

**Tradeoffs**
- Typically memory/branch bound; batching improves throughput.
