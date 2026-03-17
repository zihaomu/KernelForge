# SIMD Vectorization (CPU ARM)

**Intent**: Use SIMD (NEON/SVE or xsimd-like abstraction) to widen inner loops and increase throughput.

**When It Works**
- Regular inner loops with contiguous memory.
- Data can be packed/blocked to match vector width.

**Recognition Signals (Code)**
- NEON intrinsics, or vector abstraction layers.
- Accumulation into vector registers; horizontal reductions.
- Handling tails (remainder) carefully.

**Tradeoffs**
- Tail handling and alignment complicate code.
- Over-vectorization can reduce frequency/occupancy on some cores.
