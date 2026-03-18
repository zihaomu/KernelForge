# elementwise_add (GPU)

v1: `y = a + b`（CUDA, torch extension）

## 目录结构
```
elementwise_add/
  python/kc_elementwise_add.py
  cuda_kernel/
    binding.cpp
    kernel.cu
  test/
    test_accuracy.py
    bench.py
  example/
    run_add.py
```

## Python API
```python
import torch
from kernel.gpu_kernel.elementwise_add.python.kc_elementwise_add import add

y = add(a, b)
```

约束（v1）：
- `a/b` 必须是 CUDA tensor、同 shape、同 dtype
- dtype: `torch.float16` / `torch.float32`
- contiguous（内部会调用 `contiguous()`）

## Accuracy / Benchmark / Example
```bash
conda run -n py12_sgl python kernel/gpu_kernel/elementwise_add/test/test_accuracy.py
conda run -n py12_sgl python kernel/gpu_kernel/elementwise_add/test/bench.py
conda run -n py12_sgl python kernel/gpu_kernel/elementwise_add/example/run_add.py
```

