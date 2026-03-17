# reduce (gpu/cuda)

## Summary
- source_repo: `opencv-dnn`
- platform: `nvidia`
- backend: `cuda`
- style_inference: `gpu_cuda_focus`
- production_grade: `medium`
- quality_signals: tests=yes, bench=no, ci=no
- inferred_dtypes: `['fp16', 'fp32', 'int8']`
- inferred_layouts: `['NCHW', 'NHWC']`
- detection: hits=220, files=24

## Key Files (Entry Points)
### CUDA Sources
- `src/cuda/grid_nms.cu`
- `src/cuda/permute.cu`
- `src/cuda/resize.cu`
- `src/cuda/slice.cu`
### Glue / Operator Integration
- `include/opencv2/dnn/dnn.hpp`
- `misc/caffe/opencv-caffe.pb.cc`
- `misc/caffe/opencv-caffe.pb.h`
- `perf/perf_convolution.cpp`
- `perf/perf_convolution1d.cpp`
- `perf/perf_convolution3d.cpp`
- `perf/perf_gemm.cpp`
- `perf/perf_layer.cpp`
- `src/int8layers/pooling_layer.cpp`
- `src/layers/einsum_layer.cpp`
- `src/layers/normalize_bbox_layer.cpp`
- `src/layers/pooling_layer.cpp`
- `src/layers/reduce_layer.cpp`
- `src/layers/scale_layer.cpp`
- `src/layers/scatterND_layer.cpp`
- `src/layers/scatter_layer.cpp`


## Optimization Signals (Within These Files)
- async_copy_pipeline: `0`
- double_buffer: `0`
- shared_memory_tiling: `9`
- tensor_core: `0`
- vectorized_load_store: `0`
- warp_reduce: `0`

## Implementation Notes (Agent-Facing)
- Reductions are often the bottleneck; prefer warp-level reductions + one final block reduction.
- If available, CUB primitives are a strong baseline; custom kernels win when fused into a larger op.

## Patterns To Read
- [../../patterns/gpu/reduce/block_reduce.md](../../patterns/gpu/reduce/block_reduce.md)

## See Also (Same Op In Other Repos)
- [MatmulTutorial](../../../MatmulTutorial/ops/gpu/reduce.md)
- [MNN](../../../MNN/ops/gpu/reduce.md)
- [oneflow](../../../oneflow/ops/gpu/reduce.md)
- [sglang](../../../sglang/ops/gpu/reduce.md)

## Scope Notes
- Out of scope: non-NVIDIA GPU backends (ROCm/Metal/Vulkan).
