# SIMD Reduce (CPU ARM)

**Intent**: Use NEON/SVE to accelerate reductions (sum/max/min/argmax-like) over contiguous arrays.

**When It Works**
- Large contiguous reductions where memory access is predictable.

**Recognition Signals (Code)**
- Vector loads and horizontal reductions
- Two-stage reduce: SIMD lanes then scalar tail

**Tradeoffs**
- Argmax requires tracking indices; increases register pressure.
