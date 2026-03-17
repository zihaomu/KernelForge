# Direct Conv 3x3 (CPU ARM)

**Intent**: Compute 3x3 convolution directly (no transform) with tight inner loops and NEON/SVE vectorization.

**When It Works**
- Small kernels (3x3) and typical mobile batch sizes.
- Depthwise or standard conv depending on layout.

**Recognition Signals (Code)**
- Explicit 3x3 unrolled multiply-adds.
- Intrinsics: `<arm_neon.h>` / NEON types.
- Careful input pointer arithmetic, possible prefetch.

**Tradeoffs**
- Less flexible than im2col; harder to generalize.
- Sensitive to layout (NHWC vs NCHW) and channel blocking.
