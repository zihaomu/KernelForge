# Welford LayerNorm (GPU)

**Intent**: Use Welford's algorithm to compute mean/variance stably, then normalize and apply affine transform.

**When It Works**
- Large hidden sizes where reduction dominates.
- When fusing bias/residual/dropout reduces memory traffic.

**Recognition Signals (Code)**
- Welford combine steps across threads/warps.
- Separate passes for stats and normalization, or fused when possible.
- Warp reductions (`__shfl_*`) and vectorized loads.

**Tradeoffs**
- Fused variants are complex; numerical parity needs explicit tests.
