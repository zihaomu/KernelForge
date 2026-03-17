# activation (gpu/cuda)

## Summary
- source_repo: `opencv-dnn`
- platform: `nvidia`
- backend: `cuda`
- style_inference: `gpu_cuda_focus`
- production_grade: `medium`
- quality_signals: tests=yes, bench=no, ci=no
- inferred_dtypes: `['fp16', 'fp32', 'int8']`
- inferred_layouts: `['NCHW', 'NHWC']`
- detection: hits=2416, files=24

## Key Files (Entry Points)
### CUDA Sources
- `src/cuda/activation_eltwise.cu`
- `src/cuda/activations.cu`
- `src/cuda/bias_activation.cu`
- `src/cuda/bias_activation_eltwise.cu`
- `src/cuda/bias_eltwise_activation.cu`
- `src/cuda/eltwise_activation.cu`
- `src/cuda/region.cu`
### Glue / Operator Integration
- `include/opencv2/dnn/all_layers.hpp`
- `include/opencv2/dnn/dnn.hpp`
- `misc/caffe/opencv-caffe.pb.cc`
- `misc/caffe/opencv-caffe.pb.h`
- `misc/tflite/schema_generated.h`
- `perf/perf_layer.cpp`
- `src/caffe/caffe_io.cpp`
- `src/cuda/functors.hpp`
- `src/cuda/math.hpp`
- `src/cuda4dnn/csl/cudnn/activation.hpp`
- `src/cuda4dnn/csl/cudnn/recurrent.hpp`
- `src/cuda4dnn/csl/tensor_ops.hpp`
- `src/cuda4dnn/kernels/activation_eltwise.hpp`
- `src/cuda4dnn/kernels/activations.hpp`
- `src/cuda4dnn/kernels/bias_activation.hpp`
- `src/cuda4dnn/kernels/bias_activation_eltwise.hpp`


## Optimization Signals (Within These Files)
- async_copy_pipeline: `0`
- double_buffer: `0`
- shared_memory_tiling: `0`
- tensor_core: `0`
- vectorized_load_store: `0`
- warp_reduce: `0`

## Implementation Notes (Agent-Facing)
- Activation kernels are usually bandwidth-bound; prioritize fusion and vectorization.
- For GELU/SILU, polynomial/approx variants change accuracy; ensure tests lock behavior.

## See Also (Same Op In Other Repos)
- [MNN](../../../MNN/ops/gpu/activation.md)
- [oneflow](../../../oneflow/ops/gpu/activation.md)
- [sglang](../../../sglang/ops/gpu/activation.md)

## Scope Notes
- Out of scope: non-NVIDIA GPU backends (ROCm/Metal/Vulkan).
