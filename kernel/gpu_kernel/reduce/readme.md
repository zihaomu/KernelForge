# reduce (GPU)

v1: last-dim reduce（sum / max），CUDA extension，通过 Python API 调用。

## Python API
文件：`python/kc_reduce.py`

```python
from kernel.gpu_kernel.reduce.python.kc_reduce import reduce_sum, reduce_max

y1 = reduce_sum(x, dim=-1)
y2 = reduce_max(x, dim=-1)
```

约束（v1）：
- `x` 必须是 CUDA tensor
- `dim=-1` only
- dtype: fp16/fp32
- contiguous（内部会 `contiguous()`）

## Accuracy / Benchmark / Example
```bash
conda run -n py12_sgl python kernel/gpu_kernel/reduce/test/test_accuracy.py
conda run -n py12_sgl python kernel/gpu_kernel/reduce/test/bench.py
conda run -n py12_sgl python kernel/gpu_kernel/reduce/example/run_reduce.py
```

