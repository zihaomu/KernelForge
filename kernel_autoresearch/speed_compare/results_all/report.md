# GEMM Speed Compare: kernel_autoresearch vs OpenBLAS

- Rows: 15

## Mean Summary By Case

| case | dtype | openblas baseline | mean kc/openblas (GFLOPS) | mean slower side | mean slowdown |
|---|---|---|---:|---|---:|
| f16_quick | f16->f16 | proxy_f32_from_f16 | 0.0880 | kernel_autoresearch | 6338.73% |
| f32_default | f32->f32 | native_f32 | 0.1483 | kernel_autoresearch | 538.01% |
| i8_quick | i8->i32 | proxy_f32_from_i8 | 0.1928 | kernel_autoresearch | 626.63% |

## Detail: f16_quick (f16->f16)

| shape | bucket | threads | kc_gflops | openblas_gflops | kc/openblas | slower side | slowdown |
|---|---:|---:|---:|---:|---:|---|---:|
| s128 | medium | 16 | 45.839 | 259.549 | 0.1766 | kernel_autoresearch | 466.21% |
| s256 | medium | 16 | 92.968 | 752.713 | 0.1235 | kernel_autoresearch | 709.64% |
| m512x1024x256 | medium | 16 | 102.078 | 934.335 | 0.1093 | kernel_autoresearch | 815.32% |
| m1024 | large | 4 | 8.913 | 590.314 | 0.0151 | kernel_autoresearch | 6523.00% |
| l2048x1024x2048 | large | 4 | 8.974 | 585.505 | 0.0153 | kernel_autoresearch | 6424.77% |

## Detail: f32_default (f32->f32)

| shape | bucket | threads | kc_gflops | openblas_gflops | kc/openblas | slower side | slowdown |
|---|---:|---:|---:|---:|---:|---|---:|
| s128 | medium | 16 | 24.128 | 370.980 | 0.0650 | kernel_autoresearch | 1437.58% |
| s256 | medium | 16 | 91.710 | 754.490 | 0.1216 | kernel_autoresearch | 722.69% |
| m512x1024x256 | medium | 16 | 239.621 | 941.378 | 0.2545 | kernel_autoresearch | 292.86% |
| m1024 | large | 16 | 223.163 | 1579.500 | 0.1413 | kernel_autoresearch | 607.78% |
| l2048x1024x2048 | large | 16 | 293.085 | 1842.680 | 0.1591 | kernel_autoresearch | 528.72% |

## Detail: i8_quick (i8->i32)

| shape | bucket | threads | kc_gflops | openblas_gflops | kc/openblas | slower side | slowdown |
|---|---:|---:|---:|---:|---:|---|---:|
| s128 | medium | 16 | 48.140 | 254.046 | 0.1895 | kernel_autoresearch | 427.73% |
| s256 | medium | 16 | 123.817 | 753.981 | 0.1642 | kernel_autoresearch | 508.95% |
| m512x1024x256 | medium | 16 | 268.297 | 814.838 | 0.3293 | kernel_autoresearch | 203.71% |
| m1024 | large | 4 | 87.660 | 594.398 | 0.1475 | kernel_autoresearch | 578.07% |
| l2048x1024x2048 | large | 4 | 79.689 | 596.949 | 0.1335 | kernel_autoresearch | 649.10% |
