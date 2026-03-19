# GEMM Speed Compare: kernel_autoresearch vs OpenBLAS

- Shapes: 5
- Mean GFLOPS ratio (KC/OpenBLAS): 0.1550
- Best ratio: m512x1024x256 -> 0.2253
- Worst ratio: s128 -> 0.0949

| shape | bucket | kc_gflops | openblas_gflops | kc/openblas | kc_latency_speedup |
|---|---:|---:|---:|---:|---:|
| s128 | medium | 25.677 | 270.705 | 0.0949 | 0.0949 |
| s256 | medium | 103.366 | 730.143 | 0.1416 | 0.1416 |
| m512x1024x256 | medium | 212.803 | 944.640 | 0.2253 | 0.2253 |
| m1024 | large | 255.030 | 1536.130 | 0.1660 | 0.1660 |
| l2048x1024x2048 | large | 277.750 | 1882.750 | 0.1475 | 0.1475 |
