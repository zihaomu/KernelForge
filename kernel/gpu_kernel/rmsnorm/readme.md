# rmsnorm (GPU)

v1: RMSNorm along last dim（CUDA extension），Python API 调用。

## Python API
文件：`python/kc_rmsnorm.py`

```python
from kernel.gpu_kernel.rmsnorm.python.kc_rmsnorm import rmsnorm
y = rmsnorm(x, weight, eps=1e-5)
```

约束（v1）：
- `x/weight` 必须是 CUDA tensor
- `x` last-dim = `weight.numel()`
- dtype: fp16/fp32（`x` 与 `weight` dtype 需一致）
- contiguous（内部会 `contiguous()`）

## Accuracy / Benchmark / Example
```bash
conda run -n py12_sgl python kernel/gpu_kernel/rmsnorm/test/test_accuracy.py
conda run -n py12_sgl python kernel/gpu_kernel/rmsnorm/test/bench.py
conda run -n py12_sgl python kernel/gpu_kernel/rmsnorm/example/run_rmsnorm.py
```

