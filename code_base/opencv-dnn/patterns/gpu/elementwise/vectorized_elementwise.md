# Vectorized Elementwise (GPU)

**Intent**: Maximize memory throughput for elementwise ops via vectorized loads/stores and fusion.

**When It Works**
- Large contiguous tensors with simple pointwise math.
- When multiple elementwise ops can be fused into one kernel.

**Recognition Signals (Code)**
- `half2` / `float4` loads, aligned pointers, stride-1 access.
- Minimal branching; predication for tails.

**Tradeoffs**
- Misalignment and non-contiguous layouts degrade quickly.
