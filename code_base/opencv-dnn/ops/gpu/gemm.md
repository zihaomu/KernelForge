# gemm (gpu/cuda)

## Summary
- source_repo: `opencv-dnn`
- platform: `nvidia`
- backend: `cuda`
- style_inference: `gpu_cuda_focus`
- production_grade: `medium`
- quality_signals: tests=yes, bench=no, ci=no
- inferred_dtypes: `['fp16', 'fp32', 'int8']`
- inferred_layouts: `['NCHW', 'NHWC']`
- detection: hits=75, files=24

## Key Files (Entry Points)
### Glue / Operator Integration
- `perf/perf_gemm.cpp`
- `perf/perf_layer.cpp`
- `src/cuda4dnn/csl/cublas.hpp`
- `src/cuda4dnn/csl/tensor_ops.hpp`
- `src/cuda4dnn/primitives/inner_product.hpp`
- `src/cuda4dnn/primitives/matmul.hpp`
- `src/int8layers/fully_connected_layer.cpp`
- `src/layers/attention_layer.cpp`
- `src/layers/convolution_layer.cpp`
- `src/layers/cpu_kernels/fast_gemm.cpp`
- `src/layers/cpu_kernels/fast_gemm.hpp`
- `src/layers/cpu_kernels/fast_gemm_kernels.default.hpp`
- `src/layers/cpu_kernels/fast_gemm_kernels.simd.hpp`
- `src/layers/fully_connected_layer.cpp`
- `src/layers/gemm_layer.cpp`
- `src/layers/matmul_layer.cpp`


## Optimization Signals (Within These Files)
- async_copy_pipeline: `0`
- double_buffer: `0`
- shared_memory_tiling: `0`
- tensor_core: `0`
- vectorized_load_store: `0`
- warp_reduce: `0`

## Implementation Notes (Agent-Facing)
- Prefer block tiling + shared memory to increase A/B reuse; tune CTA tile sizes against shared/reg limits.
- If Tensor Core tokens are present, check alignment/layout constraints and whether epilogue fusion exists.
- Pipeline global->shared copies (e.g., `cp.async`) and consider double-buffering for large-K.

## Patterns To Read
- [../../patterns/gpu/gemm/block_tiling.md](../../patterns/gpu/gemm/block_tiling.md)
- [../../patterns/gpu/gemm/warp_tiling.md](../../patterns/gpu/gemm/warp_tiling.md)
- [../../patterns/gpu/gemm/tensor_core.md](../../patterns/gpu/gemm/tensor_core.md)
- [../../patterns/gpu/gemm/double_buffer.md](../../patterns/gpu/gemm/double_buffer.md)

## See Also (Same Op In Other Repos)
- [MatmulTutorial](../../../MatmulTutorial/ops/gpu/gemm.md)
- [MNN](../../../MNN/ops/gpu/gemm.md)
- [oneflow](../../../oneflow/ops/gpu/gemm.md)
- [sglang](../../../sglang/ops/gpu/gemm.md)

## Scope Notes
- Out of scope: non-NVIDIA GPU backends (ROCm/Metal/Vulkan).
