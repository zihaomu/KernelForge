# Vectorized Elementwise (CPU ARM)

**Intent**: Use NEON/SVE (or xsimd abstraction) to process multiple elements per iteration.

**When It Works**
- Contiguous tensors; simple pointwise ops.
- Fusion at a higher level (operator fusion) improves further.

**Recognition Signals (Code)**
- xsimd/NEON intrinsics, loop unrolling, aligned loads.
- Tail handling logic.
