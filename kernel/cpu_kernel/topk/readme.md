# topk (CPU)

## 接口协议
文件：`include/kc_topk.h`

v1:
- 只支持 2D `[rows, cols]`，在 last-dim 做 topk
- 输出 `values: [rows, k]`，`indices: [rows, k]`

## Accuracy
```bash
conda run -n py12_sgl python kernel/cpu_kernel/topk/test/gen_data.py

cmake -S . -B build -DKC_BUILD_TESTS=ON -DKC_BUILD_BENCH=ON -DKC_BUILD_EXAMPLES=ON
cmake --build build -j
./build/kernel/cpu_kernel/topk/kc_topk_accuracy_test
```

## Benchmark
```bash
./build/kernel/cpu_kernel/topk/kc_topk_bench
```

## Example
```bash
./build/kernel/cpu_kernel/topk/kc_topk_example
```

