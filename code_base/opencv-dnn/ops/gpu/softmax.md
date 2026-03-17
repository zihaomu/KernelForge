# softmax (gpu/cuda)

## Summary
- source_repo: `opencv-dnn`
- platform: `nvidia`
- backend: `cuda`
- style_inference: `gpu_cuda_focus`
- production_grade: `medium`
- quality_signals: tests=yes, bench=no, ci=no
- inferred_dtypes: `['fp16', 'fp32', 'int8']`
- inferred_layouts: `['NCHW', 'NHWC']`
- detection: hits=422, files=24

## Key Files (Entry Points)
### CUDA Sources
- `src/cuda/region.cu`
### Glue / Operator Integration
- `include/opencv2/dnn/all_layers.hpp`
- `include/opencv2/dnn/dnn.hpp`
- `misc/caffe/opencv-caffe.pb.cc`
- `misc/caffe/opencv-caffe.pb.h`
- `perf/perf_layer.cpp`
- `perf/perf_net.cpp`
- `src/caffe/caffe_io.cpp`
- `src/cuda4dnn/csl/cudnn/softmax.hpp`
- `src/cuda4dnn/csl/tensor_ops.hpp`
- `src/cuda4dnn/kernels/region.hpp`
- `src/cuda4dnn/primitives/region.hpp`
- `src/cuda4dnn/primitives/softmax.hpp`
- `src/darknet/darknet_io.cpp`
- `src/int8layers/softmax_layer.cpp`
- `src/layers/attention_layer.cpp`
- `src/layers/cpu_kernels/softmax.cpp`


## Optimization Signals (Within These Files)
- async_copy_pipeline: `0`
- double_buffer: `0`
- shared_memory_tiling: `0`
- tensor_core: `0`
- vectorized_load_store: `0`
- warp_reduce: `0`

## Implementation Notes (Agent-Facing)
- Use stable softmax: subtract max, then exp+sum, then normalize; avoid extra global passes when possible.
- Warp/block reductions are the core; vectorized loads help when inputs are aligned and contiguous.
- For attention scores, prefer online softmax within tiles (FlashAttention-style).

## Patterns To Read
- [../../patterns/gpu/softmax/online_softmax.md](../../patterns/gpu/softmax/online_softmax.md)

## See Also (Same Op In Other Repos)
- [MNN](../../../MNN/ops/gpu/softmax.md)
- [oneflow](../../../oneflow/ops/gpu/softmax.md)
- [sglang](../../../sglang/ops/gpu/softmax.md)

## Scope Notes
- Out of scope: non-NVIDIA GPU backends (ROCm/Metal/Vulkan).
