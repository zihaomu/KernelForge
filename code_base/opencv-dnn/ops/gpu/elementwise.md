# elementwise (gpu/cuda)

## Summary
- source_repo: `opencv-dnn`
- platform: `nvidia`
- backend: `cuda`
- style_inference: `gpu_cuda_focus`
- production_grade: `medium`
- quality_signals: tests=yes, bench=no, ci=no
- inferred_dtypes: `['fp16', 'fp32', 'int8']`
- inferred_layouts: `['NCHW', 'NHWC']`
- detection: hits=1310, files=24

## Key Files (Entry Points)
### CUDA Sources
- `src/cuda/activation_eltwise.cu`
- `src/cuda/bias_activation_eltwise.cu`
- `src/cuda/bias_eltwise_activation.cu`
- `src/cuda/eltwise_activation.cu`
- `src/cuda/eltwise_ops.cu`
- `src/cuda/grid_nms.cu`
### Glue / Operator Integration
- `misc/caffe/opencv-caffe.pb.cc`
- `misc/caffe/opencv-caffe.pb.h`
- `perf/perf_layer.cpp`
- `src/caffe/caffe_io.cpp`
- `src/cuda4dnn/csl/cudnn/convolution.hpp`
- `src/cuda4dnn/csl/tensor_ops.hpp`
- `src/cuda4dnn/kernels/activation_eltwise.hpp`
- `src/cuda4dnn/kernels/bias_activation_eltwise.hpp`
- `src/cuda4dnn/kernels/bias_eltwise_activation.hpp`
- `src/cuda4dnn/kernels/eltwise_activation.hpp`
- `src/cuda4dnn/kernels/eltwise_ops.hpp`
- `src/cuda4dnn/primitives/convolution.hpp`
- `src/cuda4dnn/primitives/eltwise.hpp`
- `src/cuda4dnn/primitives/matmul.hpp`
- `src/cuda4dnn/primitives/matmul_broadcast.hpp`
- `src/cuda4dnn/primitives/scale_shift.hpp`


## Optimization Signals (Within These Files)
- async_copy_pipeline: `0`
- double_buffer: `0`
- shared_memory_tiling: `7`
- tensor_core: `0`
- vectorized_load_store: `0`
- warp_reduce: `0`

## Implementation Notes (Agent-Facing)
- Vectorize loads/stores (half2/float4) and fuse multiple pointwise ops into one kernel when possible.
- Avoid divergence; handle tails with predication.

## Patterns To Read
- [../../patterns/gpu/elementwise/vectorized_elementwise.md](../../patterns/gpu/elementwise/vectorized_elementwise.md)

## See Also (Same Op In Other Repos)
- [MatmulTutorial](../../../MatmulTutorial/ops/gpu/elementwise.md)
- [MNN](../../../MNN/ops/gpu/elementwise.md)
- [oneflow](../../../oneflow/ops/gpu/elementwise.md)
- [sglang](../../../sglang/ops/gpu/elementwise.md)

## Scope Notes
- Out of scope: non-NVIDIA GPU backends (ROCm/Metal/Vulkan).
