# softmax (CPU)

本算子用于验证 CPU 侧的标准交付物：
- `extern "C"` API
- numpy reference 生成 `.npy`
- C++/gtest accuracy
- benchmark
- example

## 接口协议
文件：`include/kc_softmax.h`

v1:
- 输入：2D `[rows, cols]`，float32
- softmax 维度：last dim（`cols`）

## Accuracy
1. 生成数据
```bash
conda run -n py12_sgl python kernel/cpu_kernel/softmax/test/gen_data.py
```

2. 编译 + 运行测试
```bash
cmake -S . -B build
cmake --build build -j
./build/kernel/cpu_kernel/softmax/kc_softmax_accuracy_test
```

## Benchmark
```bash
./build/kernel/cpu_kernel/softmax/kc_softmax_bench
```

## Example
```bash
./build/kernel/cpu_kernel/softmax/kc_softmax_example
```

