# Flash Attention Style (GPU)

**Intent**: Tile attention computation to avoid materializing the full attention matrix, improving memory efficiency.

**When It Works**
- Long context, where `QK^T` is too large to store.
- FP16/BF16 with Tensor Cores and careful accumulation.

**Recognition Signals (Code)**
- Blocking over sequence dimension; compute tiles of `QK` and `PV`.
- Online softmax within tiles (keep running max/sum).
- Heavy use of shared memory and/or `cp.async` pipelines.

**Tradeoffs**
- Complex masking (causal/padding) handling.
- Sensitive to tile sizes and head dimension.
