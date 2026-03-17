# INT8 Dequant / Epilogue Fusion (GPU)

**Intent**: Fuse dequantization (scale/zero-point) with GEMM epilogue or elementwise to reduce memory traffic.

**When It Works**
- INT8 GEMM pipelines where output is immediately consumed by another op.

**Recognition Signals (Code)**
- Per-channel/per-tensor scales applied in epilogue.
- Vectorized int8 loads, accumulation to int32/float.

**Tradeoffs**
- Scale precision and rounding behavior must match reference.
