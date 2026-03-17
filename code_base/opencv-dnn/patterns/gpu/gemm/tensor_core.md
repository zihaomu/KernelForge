# Tensor Core (GPU GEMM)

**Intent**: Use Tensor Cores (WMMA / MMA) to increase throughput for FP16/BF16/INT8-like GEMMs.

**When It Works**
- Shapes compatible with MMA tile sizes (often multiples of 8/16).
- Data layout and alignment can be arranged to feed fragments efficiently.

**Recognition Signals (Code)**
- Tokens like `wmma`, `mma.sync`, `ldmatrix`.
- Fragment-based compute and epilogue scaling.
- Possible use of CUTLASS/CUTE abstractions.

**Constraints**
- Architecture-dependent (often `sm70+`; advanced pipelines `sm80+`).
- Layout constraints are common; expect padding / swizzle.

**Tradeoffs**
- Complex code; fragile performance if alignment/layout not met.
