# rmsnorm (CPU)

## 接口协议
文件：`include/kc_rmsnorm.h`

v1:
- 输入 `x: [rows, cols]`、`weight: [cols]`
- 输出 `y: [rows, cols]`
- 公式：`y = x * rsqrt(mean(x^2) + eps) * weight`

## Accuracy
```bash
conda run -n py12_sgl python kernel/cpu_kernel/rmsnorm/test/gen_data.py

cmake -S . -B build -DKC_BUILD_TESTS=ON -DKC_BUILD_BENCH=ON -DKC_BUILD_EXAMPLES=ON
cmake --build build -j
./build/kernel/cpu_kernel/rmsnorm/kc_rmsnorm_accuracy_test
```

## Benchmark
```bash
./build/kernel/cpu_kernel/rmsnorm/kc_rmsnorm_bench
```

## Example
```bash
./build/kernel/cpu_kernel/rmsnorm/kc_rmsnorm_example
```

