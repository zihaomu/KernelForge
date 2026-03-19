# GEMM Speed Compare: kernel_autoresearch vs OpenBLAS

- Shapes: 5
- Mean GFLOPS ratio (KC/OpenBLAS): 0.1573
- Best ratio: m512x1024x256 -> 0.2394
- Worst ratio: s128 -> 0.0792

| shape | bucket | kc_gflops | openblas_gflops | kc/openblas | kc_latency_speedup |
|---|---:|---:|---:|---:|---:|
| s128 | medium | 23.327 | 294.616 | 0.0792 | 0.0792 |
| s256 | medium | 75.056 | 666.556 | 0.1126 | 0.1126 |
| m512x1024x256 | medium | 221.712 | 925.980 | 0.2394 | 0.2394 |
| m1024 | large | 260.424 | 1092.770 | 0.2383 | 0.2383 |
| l2048x1024x2048 | large | 213.921 | 1827.200 | 0.1171 | 0.1171 |
