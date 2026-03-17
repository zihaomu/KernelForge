# topk (gpu/cuda)

## Summary
- source_repo: `opencv-dnn`
- platform: `nvidia`
- backend: `cuda`
- style_inference: `gpu_cuda_focus`
- production_grade: `medium`
- quality_signals: tests=yes, bench=no, ci=no
- inferred_dtypes: `['fp16', 'fp32', 'int8']`
- inferred_layouts: `['NCHW', 'NHWC']`
- detection: hits=326, files=24

## Key Files (Entry Points)
### CUDA Sources
- `src/cuda/detection_output.cu`
- `src/cuda/grid_nms.cu`
### Glue / Operator Integration
- `include/opencv2/dnn/all_layers.hpp`
- `include/opencv2/dnn/dnn.hpp`
- `misc/caffe/opencv-caffe.pb.cc`
- `misc/caffe/opencv-caffe.pb.h`
- `misc/tflite/schema_generated.h`
- `perf/perf_layer.cpp`
- `src/cuda4dnn/kernels/detection_output.hpp`
- `src/cuda4dnn/primitives/detection_output.hpp`
- `src/init.cpp`
- `src/layers/detection_output_layer.cpp`
- `src/layers/proposal_layer.cpp`
- `src/layers/topk_layer.cpp`
- `src/model.cpp`
- `src/nms.cpp`
- `src/nms.inl.hpp`
- `src/onnx/onnx_importer.cpp`


## Optimization Signals (Within These Files)
- async_copy_pipeline: `0`
- double_buffer: `0`
- shared_memory_tiling: `10`
- tensor_core: `0`
- vectorized_load_store: `0`
- warp_reduce: `4`

## Repo-Specific Hints (Inferred)
- TopK-related CUDA sources detected; check whether it uses per-thread candidates + shared merge or full sort networks.

## Implementation Notes (Agent-Facing)
- Top-k is selection, not full sort; keep k small and use hierarchical candidate reduction.
- For vocab-sized topk, a two-stage approach is typical: block-local topk then merge.

## Patterns To Read
- [../../patterns/gpu/topk/radix_select.md](../../patterns/gpu/topk/radix_select.md)

## See Also (Same Op In Other Repos)
- [MNN](../../../MNN/ops/gpu/topk.md)
- [oneflow](../../../oneflow/ops/gpu/topk.md)
- [sglang](../../../sglang/ops/gpu/topk.md)

## Scope Notes
- Out of scope: non-NVIDIA GPU backends (ROCm/Metal/Vulkan).
