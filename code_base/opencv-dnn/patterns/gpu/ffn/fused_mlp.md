# Fused MLP/FFN (GPU)

**Intent**: Fuse GEMM + activation (GELU/SwiGLU) + GEMM (and possibly bias) to reduce memory traffic.

**When It Works**
- Transformer FFN blocks with large batch*seq and hidden sizes.
- When epilogue fusion can be expressed efficiently (CUTLASS-like).

**Recognition Signals (Code)**
- Two GEMMs with an activation in between, possibly fused epilogues.
- Tensor Core usage and tiled pipelines.

**Tradeoffs**
- Harder scheduling and more intermediate precision choices.
