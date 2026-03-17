# rope (gpu/cuda)

## Summary
- source_repo: `opencv-dnn`
- platform: `nvidia`
- backend: `cuda`
- style_inference: `gpu_cuda_focus`
- production_grade: `medium`
- quality_signals: tests=yes, bench=no, ci=no
- inferred_dtypes: `['fp16', 'fp32', 'int8']`
- inferred_layouts: `['NCHW', 'NHWC']`
- detection: hits=327, files=24

## Key Files (Entry Points)
### Glue / Operator Integration
- `include/opencv2/dnn.hpp`
- `include/opencv2/dnn/all_layers.hpp`
- `include/opencv2/dnn/dict.hpp`
- `include/opencv2/dnn/dnn.hpp`
- `include/opencv2/dnn/dnn.inl.hpp`
- `include/opencv2/dnn/layer.hpp`
- `include/opencv2/dnn/shape_utils.hpp`
- `include/opencv2/dnn/utils/inference_engine.hpp`
- `perf/perf_caffe.cpp`
- `perf/perf_net.cpp`
- `perf/perf_utils.cpp`
- `src/caffe/caffe_importer.cpp`
- `src/caffe/caffe_io.cpp`
- `src/caffe/caffe_io.hpp`
- `src/caffe/caffe_shrinker.cpp`
- `src/caffe/glog_emulator.hpp`


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
- [MatmulTutorial](../../../MatmulTutorial/ops/gpu/rope.md)
- [MNN](../../../MNN/ops/gpu/rope.md)
- [oneflow](../../../oneflow/ops/gpu/rope.md)
- [sglang](../../../sglang/ops/gpu/rope.md)

## Scope Notes
- Out of scope: non-NVIDIA GPU backends (ROCm/Metal/Vulkan).
