# reduce (CPU)

## 接口协议
文件：`include/kc_reduce.h`

v1:
- `kc_reduce_sum_lastdim_f32(input, output, rows, cols)` 输出 `[rows]`
- `kc_reduce_max_lastdim_f32(input, output, rows, cols)` 输出 `[rows]`

## Accuracy
```bash
conda run -n py12_sgl python kernel/cpu_kernel/reduce/test/gen_data.py

cmake -S . -B build -DKC_BUILD_TESTS=ON -DKC_BUILD_BENCH=ON -DKC_BUILD_EXAMPLES=ON
cmake --build build -j
./build/kernel/cpu_kernel/reduce/kc_reduce_accuracy_test
```

## Benchmark
```bash
./build/kernel/cpu_kernel/reduce/kc_reduce_bench
```

## Example
```bash
./build/kernel/cpu_kernel/reduce/kc_reduce_example
```

