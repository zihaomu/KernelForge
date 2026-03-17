# conv (gpu/cuda)

## Summary
- source_repo: `opencv-dnn`
- platform: `nvidia`
- backend: `cuda`
- style_inference: `gpu_cuda_focus`
- production_grade: `medium`
- quality_signals: tests=yes, bench=no, ci=no
- inferred_dtypes: `['fp16', 'fp32', 'int8']`
- inferred_layouts: `['NCHW', 'NHWC']`
- detection: hits=1599, files=24

## Key Files (Entry Points)
### Glue / Operator Integration
- `include/opencv2/dnn/all_layers.hpp`
- `include/opencv2/dnn/dnn.hpp`
- `misc/caffe/opencv-caffe.pb.cc`
- `misc/caffe/opencv-caffe.pb.h`
- `misc/tflite/schema_generated.h`
- `perf/perf_convolution.cpp`
- `perf/perf_convolution1d.cpp`
- `perf/perf_convolution3d.cpp`
- `src/caffe/caffe_importer.cpp`
- `src/caffe/caffe_io.cpp`
- `src/caffe/caffe_shrinker.cpp`
- `src/cuda4dnn/csl/cudnn/convolution.hpp`
- `src/cuda4dnn/csl/cudnn/transform.hpp`
- `src/cuda4dnn/csl/cudnn/transpose_convolution.hpp`
- `src/cuda4dnn/csl/tensor_ops.hpp`
- `src/cuda4dnn/primitives/convolution.hpp`


## Optimization Signals (Within These Files)
- async_copy_pipeline: `0`
- double_buffer: `0`
- shared_memory_tiling: `0`
- tensor_core: `0`
- vectorized_load_store: `0`
- warp_reduce: `0`

## Implementation Notes (Agent-Facing)
- (no curated notes yet)

## See Also (Same Op In Other Repos)
- [MNN](../../../MNN/ops/gpu/conv.md)
- [oneflow](../../../oneflow/ops/gpu/conv.md)

## Scope Notes
- Out of scope: non-NVIDIA GPU backends (ROCm/Metal/Vulkan).
