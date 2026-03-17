# quantization (gpu/cuda)

## Summary
- source_repo: `opencv-dnn`
- platform: `nvidia`
- backend: `cuda`
- style_inference: `gpu_cuda_focus`
- production_grade: `medium`
- quality_signals: tests=yes, bench=no, ci=no
- inferred_dtypes: `['fp16', 'fp32', 'int8']`
- inferred_layouts: `['NCHW', 'NHWC']`
- detection: hits=2379, files=24

## Key Files (Entry Points)
### Glue / Operator Integration
- `include/opencv2/dnn/all_layers.hpp`
- `include/opencv2/dnn/dnn.hpp`
- `misc/caffe/opencv-caffe.pb.cc`
- `misc/caffe/opencv-caffe.pb.h`
- `misc/onnx/opencv-onnx.pb.cc`
- `misc/onnx/opencv-onnx.pb.h`
- `misc/tensorflow/attr_value.pb.cc`
- `misc/tensorflow/attr_value.pb.h`
- `misc/tensorflow/function.pb.cc`
- `misc/tensorflow/function.pb.h`
- `misc/tensorflow/graph.pb.cc`
- `misc/tensorflow/graph.pb.h`
- `misc/tensorflow/op_def.pb.cc`
- `misc/tensorflow/op_def.pb.h`
- `misc/tensorflow/tensor.pb.cc`
- `misc/tensorflow/tensor.pb.h`


## Optimization Signals (Within These Files)
- async_copy_pipeline: `0`
- double_buffer: `0`
- shared_memory_tiling: `0`
- tensor_core: `0`
- vectorized_load_store: `0`
- warp_reduce: `0`

## Implementation Notes (Agent-Facing)
- Quant/dequant is often best fused into producers/consumers (e.g., GEMM epilogue).

## Patterns To Read
- [../../patterns/gpu/quantization/int8_dequant_epilogue.md](../../patterns/gpu/quantization/int8_dequant_epilogue.md)

## See Also (Same Op In Other Repos)
- [MatmulTutorial](../../../MatmulTutorial/ops/gpu/quantization.md)
- [MNN](../../../MNN/ops/gpu/quantization.md)
- [oneflow](../../../oneflow/ops/gpu/quantization.md)
- [sglang](../../../sglang/ops/gpu/quantization.md)

## Scope Notes
- Out of scope: non-NVIDIA GPU backends (ROCm/Metal/Vulkan).
