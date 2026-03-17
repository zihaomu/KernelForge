# attention (gpu/cuda)

## Summary
- source_repo: `opencv-dnn`
- platform: `nvidia`
- backend: `cuda`
- style_inference: `gpu_cuda_focus`
- production_grade: `medium`
- quality_signals: tests=yes, bench=no, ci=no
- inferred_dtypes: `['fp16', 'fp32', 'int8']`
- inferred_layouts: `['NCHW', 'NHWC']`
- detection: hits=63, files=8

## Key Files (Entry Points)
### Glue / Operator Integration
- `include/opencv2/dnn/all_layers.hpp`
- `perf/perf_layer.cpp`
- `src/init.cpp`
- `src/layers/attention_layer.cpp`
- `src/onnx/onnx_graph_simplifier.cpp`
- `src/onnx/onnx_importer.cpp`
- `test/test_graph_simplifier.cpp`
- `test/test_onnx_importer.cpp`


## Optimization Signals (Within These Files)
- async_copy_pipeline: `0`
- double_buffer: `0`
- shared_memory_tiling: `0`
- tensor_core: `0`
- vectorized_load_store: `0`
- warp_reduce: `0`

## Implementation Notes (Agent-Facing)
- Look for tiling over sequence and online softmax signals; this is where most speedups come from.
- KV cache layout and memory indirection often dominates for decoding; confirm access patterns.

## Patterns To Read
- [../../patterns/gpu/attention/flash_attention.md](../../patterns/gpu/attention/flash_attention.md)

## See Also (Same Op In Other Repos)
- [MNN](../../../MNN/ops/gpu/attention.md)
- [oneflow](../../../oneflow/ops/gpu/attention.md)
- [sglang](../../../sglang/ops/gpu/attention.md)

## Scope Notes
- Out of scope: non-NVIDIA GPU backends (ROCm/Metal/Vulkan).
