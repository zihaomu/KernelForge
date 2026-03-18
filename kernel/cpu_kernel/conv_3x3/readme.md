# conv_3x3 (CPU)

本算子用于验证你项目里 **CPU 算子** 的完整工程链路：
- C 接口暴露
- numpy 生成 reference + 保存 `.npy`
- C++/gtest 读取 `.npy` 做 accuracy 测试
- C++ benchmark
- C++ example

## 目录结构
```
conv_3x3/
  include/kc_conv3x3.h
  src/kc_conv3x3.cc
  test/
    gen_data.py
    test_conv3x3_accuracy.cc
    data/...
  perf/bench_conv3x3.cc
  example/run_conv3x3.cc
```

## 接口协议 (C API)
文件：`include/kc_conv3x3.h`

- 数据布局：
  - `input`: NCHW `[N, Cin, H, W]`
  - `weight`: OIHW `[Cout, Cin, 3, 3]`
  - `bias`: `[Cout]`（可为 `NULL`）
  - `output`: NCHW `[N, Cout, Hout, Wout]`

- 输出 shape：
  - `Hout = floor((H + 2*pad_h - 3) / stride_h) + 1`
  - `Wout = floor((W + 2*pad_w - 3) / stride_w) + 1`

当前实现包含一个优化 fast-path：
- `stride_h=stride_w=1` 且 `pad_h=pad_w=1`：内部对 `W` 维做 xsimd 向量化（其余边界用标量处理）

## Accuracy 测试
1. 生成测试数据（numpy reference + 写入 `.npy`）：
```bash
conda run -n py12_sgl python kernel/cpu_kernel/conv_3x3/test/gen_data.py
```

2. C++ 测试（gtest + libnpy 读 `.npy`）：
```bash
cmake -S . -B build
cmake --build build -j
./build/kernel/cpu_kernel/conv_3x3/kc_conv3x3_accuracy_test
```

## Benchmark
```bash
cmake -S . -B build
cmake --build build -j
./build/kernel/cpu_kernel/conv_3x3/kc_conv3x3_bench
```

## Example
```bash
./build/kernel/cpu_kernel/conv_3x3/kc_conv3x3_example
```
