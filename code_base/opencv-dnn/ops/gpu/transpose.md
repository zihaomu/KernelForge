# transpose (gpu/cuda)

## Summary
- source_repo: `opencv-dnn`
- platform: `nvidia`
- backend: `cuda`
- style_inference: `gpu_cuda_focus`
- production_grade: `medium`
- quality_signals: tests=yes, bench=no, ci=no
- inferred_dtypes: `['fp16', 'fp32', 'int8']`
- inferred_layouts: `['NCHW', 'NHWC']`
- detection: hits=979, files=24

## Key Files (Entry Points)
### CUDA Sources
- `src/cuda/detection_output.cu`
- `src/cuda/permute.cu`
### Glue / Operator Integration
- `include/opencv2/dnn/all_layers.hpp`
- `misc/caffe/opencv-caffe.pb.cc`
- `misc/caffe/opencv-caffe.pb.h`
- `misc/tflite/schema_generated.h`
- `perf/perf_gemm.cpp`
- `perf/perf_layer.cpp`
- `perf/perf_net.cpp`
- `src/cuda4dnn/csl/cublas.hpp`
- `src/cuda4dnn/csl/cudnn/transpose_convolution.hpp`
- `src/cuda4dnn/csl/tensor_ops.hpp`
- `src/cuda4dnn/kernels/detection_output.hpp`
- `src/cuda4dnn/kernels/permute.hpp`
- `src/cuda4dnn/primitives/depth_space_ops.hpp`
- `src/cuda4dnn/primitives/detection_output.hpp`
- `src/cuda4dnn/primitives/permute.hpp`
- `src/cuda4dnn/primitives/reorg.hpp`


## Optimization Signals (Within These Files)
- async_copy_pipeline: `0`
- double_buffer: `0`
- shared_memory_tiling: `5`
- tensor_core: `0`
- vectorized_load_store: `0`
- warp_reduce: `4`

## Implementation Notes (Agent-Facing)
- Naive transpose is uncoalesced; shared-memory tiled transpose is the standard fix.

## Patterns To Read
- [../../patterns/gpu/transpose/transpose_tiling.md](../../patterns/gpu/transpose/transpose_tiling.md)

## See Also (Same Op In Other Repos)
- [MatmulTutorial](../../../MatmulTutorial/ops/gpu/transpose.md)
- [MNN](../../../MNN/ops/gpu/transpose.md)
- [oneflow](../../../oneflow/ops/gpu/transpose.md)
- [sglang](../../../sglang/ops/gpu/transpose.md)

## Scope Notes
- Out of scope: non-NVIDIA GPU backends (ROCm/Metal/Vulkan).
