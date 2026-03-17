# Winograd (CPU ARM)

**Intent**: Use Winograd transforms to reduce multiply count for small conv kernels (commonly 3x3).

**When It Works**
- 3x3 conv with sufficient spatial size.
- FP16/FP32 depending on numerical tolerance.

**Recognition Signals (Code)**
- Keywords: `winograd`, transform matrices, `Bt * d * B`, `G * g * Gt`.
- Separate transform + GEMM-like multiply + inverse transform.

**Tradeoffs**
- Extra transforms can dominate for small shapes.
- Numerical stability can be worse than direct conv.
