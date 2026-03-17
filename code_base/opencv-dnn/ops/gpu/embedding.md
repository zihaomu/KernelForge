# embedding (gpu/cuda)

## Summary
- source_repo: `opencv-dnn`
- platform: `nvidia`
- backend: `cuda`
- style_inference: `gpu_cuda_focus`
- production_grade: `medium`
- quality_signals: tests=yes, bench=no, ci=no
- inferred_dtypes: `['fp16', 'fp32', 'int8']`
- inferred_layouts: `['NCHW', 'NHWC']`
- detection: hits=238, files=22

## Key Files (Entry Points)
### Glue / Operator Integration
- `include/opencv2/dnn/all_layers.hpp`
- `include/opencv2/dnn/dnn.hpp`
- `misc/tflite/schema_generated.h`
- `perf/perf_layer.cpp`
- `src/cuda/kernel_dispatcher.hpp`
- `src/init.cpp`
- `src/int8layers/softmax_layer.cpp`
- `src/layers/cpu_kernels/convolution.cpp`
- `src/layers/gather_elements_layer.cpp`
- `src/layers/gather_layer.cpp`
- `src/layers/nary_eltwise_layers.cpp`
- `src/layers/permute_layer.cpp`
- `src/onnx/onnx_graph_simplifier.cpp`
- `src/onnx/onnx_importer.cpp`
- `src/op_cuda.hpp`
- `test/test_graph_simplifier.cpp`


## Optimization Signals (Within These Files)
- async_copy_pipeline: `0`
- double_buffer: `0`
- shared_memory_tiling: `0`
- tensor_core: `0`
- vectorized_load_store: `0`
- warp_reduce: `0`

## Implementation Notes (Agent-Facing)
- Embedding/gather is bandwidth-bound; seek coalesced access, caching, and reuse of indices within a block.

## Patterns To Read
- [../../patterns/gpu/embedding/gather.md](../../patterns/gpu/embedding/gather.md)

## See Also (Same Op In Other Repos)
- [MNN](../../../MNN/ops/gpu/embedding.md)
- [oneflow](../../../oneflow/ops/gpu/embedding.md)
- [sglang](../../../sglang/ops/gpu/embedding.md)

## Scope Notes
- Out of scope: non-NVIDIA GPU backends (ROCm/Metal/Vulkan).
