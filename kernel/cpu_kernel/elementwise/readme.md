# elementwise (CPU)

## 接口协议
文件：`include/kc_elementwise.h`

v1:
- `kc_add_f32(a, b, out, n)`: `out[i] = a[i] + b[i]`

## Accuracy
```bash
conda run -n py12_sgl python kernel/cpu_kernel/elementwise/test/gen_data.py

cmake -S . -B build -DKC_BUILD_TESTS=ON -DKC_BUILD_BENCH=ON -DKC_BUILD_EXAMPLES=ON
cmake --build build -j

./build/kernel/cpu_kernel/elementwise/kc_elementwise_accuracy_test
```

## Benchmark
```bash
./build/kernel/cpu_kernel/elementwise/kc_elementwise_bench
```

## Example
```bash
./build/kernel/cpu_kernel/elementwise/kc_elementwise_example
```

