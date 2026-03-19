# GEMM Kernel 对比实验（kernel_autoresearch vs OpenBLAS）

本目录用于对比 `kernel_autoresearch` 当前最优 GEMM kernel 与 `3rdparty/OpenBLAS` 的性能，现已覆盖三组 dtype：

- `f32 -> f32`
- `f16 -> f16`（OpenBLAS 侧使用 `sgemm` 代理基线：`f16` 输入转 `f32` 后计算）
- `i8 -> i32`（OpenBLAS 侧使用 `sgemm` 代理基线：`i8` 输入转 `f32` 后计算）

## 1. 实验方案

- 数据来源：
  - `f32_default`: `kernel_autoresearch/workspace`
  - `f16_quick`: `kernel_autoresearch/workspace_fp16_quick`
  - `i8_quick`: `kernel_autoresearch/workspace_int8_quick`
- 每个 workspace 从 `orchestration_state.json` 读取各 bucket 最优候选。
- shape 集合来自 `kernel_autoresearch/configs/shapes.yaml`。
- 统计口径：`warmup=5`, `iters=30`, 取 `latency p50`。
- 指标：
  - `gflops = 2*M*N*K / (latency_ms_p50 * 1e6)`
  - `kc/openblas = kc_gflops / openblas_gflops`
  - `slowdown% = (慢的一方延迟 / 快的一方延迟 - 1) * 100`

## 2. 一键运行

```bash
./kernel_autoresearch/speed_compare/run_compare.sh
```

输出目录：

- `kernel_autoresearch/speed_compare/results_all/compare.tsv`
- `kernel_autoresearch/speed_compare/results_all/report.md`
- `kernel_autoresearch/speed_compare/results_all/summary.json`

## 3. 对比表（2026-03-19）

来源：`results_all/report.md` 与 `results_all/compare.tsv`

### 3.1 各 dtype 平均结果

| case | dtype | OpenBLAS baseline | mean kc/openblas (GFLOPS) | 平均更慢的一方 | 平均慢多少 |
|---|---|---|---:|---|---:|
| f32_default | f32->f32 | native_f32 | 0.1483 | kernel_autoresearch | 538.01% |
| f16_quick | f16->f16 | proxy_f32_from_f16 | 0.0880 | kernel_autoresearch | 6338.73% |
| i8_quick | i8->i32 | proxy_f32_from_i8 | 0.1928 | kernel_autoresearch | 626.63% |

### 3.2 各 shape 详细结果

| case | shape | kc_gflops | openblas_gflops | kc/openblas | 更慢的一方 | 慢多少 |
|---|---|---:|---:|---:|---|---:|
| f32_default | s128 | 24.128 | 370.980 | 0.0650 | kernel_autoresearch | 1437.58% |
| f32_default | s256 | 91.710 | 754.490 | 0.1216 | kernel_autoresearch | 722.69% |
| f32_default | m512x1024x256 | 239.621 | 941.378 | 0.2545 | kernel_autoresearch | 292.86% |
| f32_default | m1024 | 223.163 | 1579.500 | 0.1413 | kernel_autoresearch | 607.78% |
| f32_default | l2048x1024x2048 | 293.085 | 1842.680 | 0.1591 | kernel_autoresearch | 528.72% |
| f16_quick | s128 | 45.839 | 259.549 | 0.1766 | kernel_autoresearch | 466.21% |
| f16_quick | s256 | 92.968 | 752.713 | 0.1235 | kernel_autoresearch | 709.64% |
| f16_quick | m512x1024x256 | 102.078 | 934.335 | 0.1093 | kernel_autoresearch | 815.32% |
| f16_quick | m1024 | 8.913 | 590.314 | 0.0151 | kernel_autoresearch | 6523.00% |
| f16_quick | l2048x1024x2048 | 8.974 | 585.505 | 0.0153 | kernel_autoresearch | 6424.77% |
| i8_quick | s128 | 48.140 | 254.046 | 0.1895 | kernel_autoresearch | 427.73% |
| i8_quick | s256 | 123.817 | 753.981 | 0.1642 | kernel_autoresearch | 508.95% |
| i8_quick | m512x1024x256 | 268.297 | 814.838 | 0.3293 | kernel_autoresearch | 203.71% |
| i8_quick | m1024 | 87.660 | 594.398 | 0.1475 | kernel_autoresearch | 578.07% |
| i8_quick | l2048x1024x2048 | 79.689 | 596.949 | 0.1335 | kernel_autoresearch | 649.10% |

## 4. 明确结论（谁慢，慢多少）

- 这轮实验里，**全部 15/15 个 shape 上都是 `kernel_autoresearch` 比 OpenBLAS 慢**。
- 平均慢多少：
  - `f32->f32`: `kernel_autoresearch` 平均慢 **538.01%**（约 **6.38x** 延迟）。
  - `f16->f16`（代理基线）: `kernel_autoresearch` 平均慢 **6338.73%**（约 **64.39x** 延迟）。
  - `i8->i32`（代理基线）: `kernel_autoresearch` 平均慢 **626.63%**（约 **7.27x** 延迟）。
- 在当前配置中，最接近 OpenBLAS 的点是 `i8_quick / m512x1024x256`，仍慢 **203.71%**（约 **3.04x**）。

## 5. 注意事项

- OpenBLAS 在你当前构建中仅暴露 `cblas_sgemm`，没有可直接调用的 `int8->int32` GEMM CBLAS 符号。
- 因此 `f16/i8` 使用的是 `OpenBLAS sgemm` 代理基线（输入按对应 dtype 量化后再转 `f32` 计算）。
- 所以上述 `f16/i8` 结论代表“当前 KC 实现 vs OpenBLAS-f32 代理基线”的结果，不是 OpenBLAS 原生 int8/fp16 内核结果。
