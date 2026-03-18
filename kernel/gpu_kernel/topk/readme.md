# topk (GPU)

v1: last-dim topk（CUDA extension），Python API 调用。

## Python API
文件：`python/kc_topk.py`

```python
from kernel.gpu_kernel.topk.python.kc_topk import topk
values, indices = topk(x, k=8, dim=-1, largest=True)
```

约束（v1）：
- `x` 必须是 CUDA tensor
- `dim=-1` only
- dtype: fp16/fp32
- `k <= 32`（当前实现是 baseline）
- contiguous（内部会 `contiguous()`）

## Accuracy / Benchmark / Example
```bash
conda run -n py12_sgl python kernel/gpu_kernel/topk/test/test_accuracy.py
conda run -n py12_sgl python kernel/gpu_kernel/topk/test/bench.py
conda run -n py12_sgl python kernel/gpu_kernel/topk/example/run_topk.py
```

