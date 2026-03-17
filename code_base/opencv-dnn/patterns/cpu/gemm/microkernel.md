# Microkernel GEMM (CPU ARM)

**Intent**: Hand-tuned inner kernels (often in asm) that compute a small MRxNR tile efficiently.

**When It Works**
- Large GEMMs where packing + microkernel amortize overhead.
- Consistent shapes or batched workloads.

**Recognition Signals (Code)**
- Files named `ukernel`/`microkernel`, lots of NEON/SVE/ASM.
- Packing routines for A/B panels.
- Kernel loops unrolled around vector FMA instructions.

**Tradeoffs**
- Many specialized kernels for different shapes/dtypes.
- Hard to maintain; correctness requires extensive tests.
