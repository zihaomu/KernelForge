# LayerNorm (CPU ARM)

**Intent**: Efficient mean/variance computation and normalization with SIMD reductions.

**When It Works**
- Large hidden dimensions; batch*seq parallelism.

**Recognition Signals (Code)**
- Reduce mean/var; vectorized affine transform.
- Welford or two-pass methods.
