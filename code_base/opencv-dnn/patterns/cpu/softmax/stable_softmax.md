# Stable Softmax (CPU ARM)

**Intent**: Compute softmax stably using `max` subtraction and SIMD reductions.

**When It Works**
- Medium/long vectors (e.g., logits) where vectorization helps.
- When multi-threading across batches is available.

**Recognition Signals (Code)**
- Pass 1: reduce max; pass 2: exp + sum; pass 3: normalize.
- NEON/SVE vector exp approximations or lookup tables.

**Tradeoffs**
- Accurate exp approximations can be expensive.
