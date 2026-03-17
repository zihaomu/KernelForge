# INT8 Quant/Dequant (CPU ARM)

**Intent**: Efficient quantization/dequantization and int8 dot-products with NEON/SVE.

**When It Works**
- Edge/mobile inference pipelines with int8 weights/activations.

**Recognition Signals (Code)**
- `int8_t` paths, per-channel scales, saturation/rounding.
- Dot-product instructions or widened accumulators.
