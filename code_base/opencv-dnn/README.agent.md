# opencv-dnn - README.agent.md

**AUTO-GENERATED** (heuristic static scan). If you edit this file manually, re-run the generator with `--overwrite-readme` to refresh it (a timestamped backup will be created).

## Project Summary
- source_repo: `opencv-dnn`
- scanned_at: `2026-03-17`
- scope_for_agent:
  - gpu: `nvidia` + `cuda` only
  - cpu: `arm` only (NEON/SVE/ASM)
- upstream_description: (not found)

## Quick Signals
- cu_files: `25`
- asm_files: `0`
- tests_present: `yes`
- benchmark_present: `no`
- ci_present: `no`
- style_inference: `gpu_cuda_focus`

## Kernel Inventory (Agent-Facing)
### GPU (NVIDIA CUDA)
- [activation][op_opencv_dnn_gpu_activation] (see also: [MNN][see_mnn_gpu_activation], [oneflow][see_oneflow_gpu_activation], [sglang][see_sglang_gpu_activation])
- [elementwise][op_opencv_dnn_gpu_elementwise] (see also: [MatmulTutorial][see_matmultutorial_gpu_elementwise], [MNN][see_mnn_gpu_elementwise], [oneflow][see_oneflow_gpu_elementwise], [sglang][see_sglang_gpu_elementwise])
- [norm][op_opencv_dnn_gpu_norm] (see also: [MNN][see_mnn_gpu_norm], [oneflow][see_oneflow_gpu_norm], [sglang][see_sglang_gpu_norm])
- [reduce][op_opencv_dnn_gpu_reduce] (see also: [MatmulTutorial][see_matmultutorial_gpu_reduce], [MNN][see_mnn_gpu_reduce], [oneflow][see_oneflow_gpu_reduce], [sglang][see_sglang_gpu_reduce])
- [softmax][op_opencv_dnn_gpu_softmax] (see also: [MNN][see_mnn_gpu_softmax], [oneflow][see_oneflow_gpu_softmax], [sglang][see_sglang_gpu_softmax])
- [topk][op_opencv_dnn_gpu_topk] (see also: [MNN][see_mnn_gpu_topk], [oneflow][see_oneflow_gpu_topk], [sglang][see_sglang_gpu_topk])
- [transpose][op_opencv_dnn_gpu_transpose] (see also: [MatmulTutorial][see_matmultutorial_gpu_transpose], [MNN][see_mnn_gpu_transpose], [oneflow][see_oneflow_gpu_transpose], [sglang][see_sglang_gpu_transpose])

## Kernel Details (Heuristic Metadata)
Notes:
- `dtype/layout/constraints` are inferred from token presence and may be incomplete.
- Treat missing fields as "unknown" instead of assuming defaults.

### activation (gpu/cuda)

```yaml
op_type: activation
backend: cuda
source_repo: opencv-dnn
platform: nvidia
dtype: [fp16, fp32, int8]
layout: [NCHW, NHWC]
optimization_tags:
  - shared_memory_tiling
  - vectorized_load_store
  - warp_reduce
applicable_shapes:
  - (unknown)
constraints:
  - alignment_16 (inferred: vectorized load/store tokens present)
quality_signals:
  correctness_test: yes
  benchmark_present: no
production_grade: medium
notes:
  - GPU metadata is scoped to NVIDIA CUDA only; other GPU backends (e.g., Vulkan/Metal/ROCm) are ignored.
```

### elementwise (gpu/cuda)

```yaml
op_type: elementwise
backend: cuda
source_repo: opencv-dnn
platform: nvidia
dtype: [fp16, fp32, int8]
layout: [NCHW, NHWC]
optimization_tags:
  - shared_memory_tiling
  - vectorized_load_store
  - warp_reduce
applicable_shapes:
  - (unknown)
constraints:
  - alignment_16 (inferred: vectorized load/store tokens present)
quality_signals:
  correctness_test: yes
  benchmark_present: no
production_grade: medium
notes:
  - GPU metadata is scoped to NVIDIA CUDA only; other GPU backends (e.g., Vulkan/Metal/ROCm) are ignored.
```

### norm (gpu/cuda)

```yaml
op_type: norm
backend: cuda
source_repo: opencv-dnn
platform: nvidia
dtype: [fp16, fp32, int8]
layout: [NCHW, NHWC]
optimization_tags:
  - shared_memory_tiling
  - vectorized_load_store
  - warp_reduce
applicable_shapes:
  - large_batch_seq
  - long_context
constraints:
  - alignment_16 (inferred: vectorized load/store tokens present)
quality_signals:
  correctness_test: yes
  benchmark_present: no
production_grade: medium
notes:
  - GPU metadata is scoped to NVIDIA CUDA only; other GPU backends (e.g., Vulkan/Metal/ROCm) are ignored.
```

### reduce (gpu/cuda)

```yaml
op_type: reduce
backend: cuda
source_repo: opencv-dnn
platform: nvidia
dtype: [fp16, fp32, int8]
layout: [NCHW, NHWC]
optimization_tags:
  - shared_memory_tiling
  - vectorized_load_store
  - warp_reduce
applicable_shapes:
  - (unknown)
constraints:
  - alignment_16 (inferred: vectorized load/store tokens present)
quality_signals:
  correctness_test: yes
  benchmark_present: no
production_grade: medium
notes:
  - GPU metadata is scoped to NVIDIA CUDA only; other GPU backends (e.g., Vulkan/Metal/ROCm) are ignored.
```

### softmax (gpu/cuda)

```yaml
op_type: softmax
backend: cuda
source_repo: opencv-dnn
platform: nvidia
dtype: [fp16, fp32, int8]
layout: [NCHW, NHWC]
optimization_tags:
  - shared_memory_tiling
  - vectorized_load_store
  - warp_reduce
applicable_shapes:
  - large_batch_seq
  - long_context
constraints:
  - alignment_16 (inferred: vectorized load/store tokens present)
quality_signals:
  correctness_test: yes
  benchmark_present: no
production_grade: medium
notes:
  - GPU metadata is scoped to NVIDIA CUDA only; other GPU backends (e.g., Vulkan/Metal/ROCm) are ignored.
```

### topk (gpu/cuda)

```yaml
op_type: topk
backend: cuda
source_repo: opencv-dnn
platform: nvidia
dtype: [fp16, fp32, int8]
layout: [NCHW, NHWC]
optimization_tags:
  - shared_memory_tiling
  - vectorized_load_store
  - warp_reduce
applicable_shapes:
  - (unknown)
constraints:
  - alignment_16 (inferred: vectorized load/store tokens present)
quality_signals:
  correctness_test: yes
  benchmark_present: no
production_grade: medium
notes:
  - GPU metadata is scoped to NVIDIA CUDA only; other GPU backends (e.g., Vulkan/Metal/ROCm) are ignored.
```

### transpose (gpu/cuda)

```yaml
op_type: transpose
backend: cuda
source_repo: opencv-dnn
platform: nvidia
dtype: [fp16, fp32, int8]
layout: [NCHW, NHWC]
optimization_tags:
  - shared_memory_tiling
  - vectorized_load_store
  - warp_reduce
applicable_shapes:
  - (unknown)
constraints:
  - alignment_16 (inferred: vectorized load/store tokens present)
quality_signals:
  correctness_test: yes
  benchmark_present: no
production_grade: medium
notes:
  - GPU metadata is scoped to NVIDIA CUDA only; other GPU backends (e.g., Vulkan/Metal/ROCm) are ignored.
```

## Patterns
- This repo contains a `patterns/` folder generated by the doc tool.
- These files are intended as a shared vocabulary for optimization-strategy search (not code search).

## Out Of Scope
- Any GPU backend other than NVIDIA CUDA.
- Any CPU architecture other than ARM (including x86).

## Link References

[op_opencv_dnn_gpu_activation]: ops/gpu/activation.md
[op_opencv_dnn_gpu_elementwise]: ops/gpu/elementwise.md
[op_opencv_dnn_gpu_norm]: ops/gpu/norm.md
[op_opencv_dnn_gpu_reduce]: ops/gpu/reduce.md
[op_opencv_dnn_gpu_softmax]: ops/gpu/softmax.md
[op_opencv_dnn_gpu_topk]: ops/gpu/topk.md
[op_opencv_dnn_gpu_transpose]: ops/gpu/transpose.md
[see_matmultutorial_gpu_elementwise]: ../MatmulTutorial/ops/gpu/elementwise.md
[see_matmultutorial_gpu_reduce]: ../MatmulTutorial/ops/gpu/reduce.md
[see_matmultutorial_gpu_transpose]: ../MatmulTutorial/ops/gpu/transpose.md
[see_mnn_gpu_activation]: ../MNN/ops/gpu/activation.md
[see_mnn_gpu_elementwise]: ../MNN/ops/gpu/elementwise.md
[see_mnn_gpu_norm]: ../MNN/ops/gpu/norm.md
[see_mnn_gpu_reduce]: ../MNN/ops/gpu/reduce.md
[see_mnn_gpu_softmax]: ../MNN/ops/gpu/softmax.md
[see_mnn_gpu_topk]: ../MNN/ops/gpu/topk.md
[see_mnn_gpu_transpose]: ../MNN/ops/gpu/transpose.md
[see_oneflow_gpu_activation]: ../oneflow/ops/gpu/activation.md
[see_oneflow_gpu_elementwise]: ../oneflow/ops/gpu/elementwise.md
[see_oneflow_gpu_norm]: ../oneflow/ops/gpu/norm.md
[see_oneflow_gpu_reduce]: ../oneflow/ops/gpu/reduce.md
[see_oneflow_gpu_softmax]: ../oneflow/ops/gpu/softmax.md
[see_oneflow_gpu_topk]: ../oneflow/ops/gpu/topk.md
[see_oneflow_gpu_transpose]: ../oneflow/ops/gpu/transpose.md
[see_sglang_gpu_activation]: ../sglang/ops/gpu/activation.md
[see_sglang_gpu_elementwise]: ../sglang/ops/gpu/elementwise.md
[see_sglang_gpu_norm]: ../sglang/ops/gpu/norm.md
[see_sglang_gpu_reduce]: ../sglang/ops/gpu/reduce.md
[see_sglang_gpu_softmax]: ../sglang/ops/gpu/softmax.md
[see_sglang_gpu_topk]: ../sglang/ops/gpu/topk.md
[see_sglang_gpu_transpose]: ../sglang/ops/gpu/transpose.md
