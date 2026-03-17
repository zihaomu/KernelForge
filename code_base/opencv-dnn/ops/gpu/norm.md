# norm (gpu/cuda)

## Summary
- source_repo: `opencv-dnn`
- platform: `nvidia`
- backend: `cuda`
- style_inference: `gpu_cuda_focus`
- production_grade: `medium`
- quality_signals: tests=yes, bench=no, ci=no
- inferred_dtypes: `['fp16', 'fp32', 'int8']`
- inferred_layouts: `['NCHW', 'NHWC']`
- detection: hits=383, files=24

## Key Files (Entry Points)
### CUDA Sources
- `src/cuda/mvn.cu`
### Glue / Operator Integration
- `include/opencv2/dnn/all_layers.hpp`
- `misc/caffe/opencv-caffe.pb.cc`
- `misc/caffe/opencv-caffe.pb.h`
- `perf/perf_layer.cpp`
- `src/caffe/caffe_importer.cpp`
- `src/caffe/caffe_io.cpp`
- `src/cuda4dnn/kernels/mvn.hpp`
- `src/cuda4dnn/primitives/batch_norm.hpp`
- `src/cuda4dnn/primitives/group_norm.hpp`
- `src/cuda4dnn/primitives/instance_norm.hpp`
- `src/cuda4dnn/primitives/layer_norm.hpp`
- `src/darknet/darknet_io.cpp`
- `src/init.cpp`
- `src/int8layers/batch_norm_layer.cpp`
- `src/layers/batch_norm_layer.cpp`
- `src/layers/cpu_kernels/fast_norm.hpp`


## Optimization Signals (Within These Files)
- async_copy_pipeline: `0`
- double_buffer: `0`
- shared_memory_tiling: `0`
- tensor_core: `0`
- vectorized_load_store: `0`
- warp_reduce: `0`

## Implementation Notes (Agent-Facing)
- Stats reduction (mean/var) is the bottleneck; use Welford or two-pass depending on numerical needs.
- Fuse affine transform (gamma/beta) and nearby elementwise ops to reduce memory traffic.

## Patterns To Read
- [../../patterns/gpu/norm/welford_layernorm.md](../../patterns/gpu/norm/welford_layernorm.md)

## See Also (Same Op In Other Repos)
- [MNN](../../../MNN/ops/gpu/norm.md)
- [oneflow](../../../oneflow/ops/gpu/norm.md)
- [sglang](../../../sglang/ops/gpu/norm.md)

## Scope Notes
- Out of scope: non-NVIDIA GPU backends (ROCm/Metal/Vulkan).
