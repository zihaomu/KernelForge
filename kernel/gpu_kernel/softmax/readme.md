# softmax (GPU)

目标：实现一个可被 PyTorch 调用的 CUDA softmax，并提供：
- Python 接口（输入输出都是 `torch.Tensor`）
- accuracy 测试（对比 `torch.softmax`）
- benchmark（对比 `torch.softmax`）
- example

## 目录结构
```
softmax/
  python/kc_softmax.py
  cuda_kernel/
    binding.cpp
    kernel.cu
    setup.py
  test/
    test_accuracy.py
    bench.py
  example/
    run_softmax.py
```

## Python API
文件：`python/kc_softmax.py`

```python
import torch
from kernel.gpu_kernel.softmax.python.kc_softmax import softmax

y = softmax(x)  # dim=-1 only (v1)
```

约束（v1）：
- `x` 必须是 CUDA tensor
- `dim=-1`（只支持 last-dim softmax）
- `dtype`: `torch.float16` / `torch.float32`
- `x` 要求 contiguous（内部会调用 `contiguous()`）

## 编译方式
使用 `torch.utils.cpp_extension.load` 动态编译（生成 `.so` 并缓存到 `~/.cache/torch_extensions`）。

环境要求：
- `conda activate py12_sgl`
- 有可用 CUDA（本机已验证 `torch 2.9.1+cu128`, CUDA 12.8）

## Accuracy
```bash
conda run -n py12_sgl python kernel/gpu_kernel/softmax/test/test_accuracy.py
```

## Benchmark
```bash
conda run -n py12_sgl python kernel/gpu_kernel/softmax/test/bench.py
```

## Example
```bash
conda run -n py12_sgl python kernel/gpu_kernel/softmax/example/run_softmax.py
```

