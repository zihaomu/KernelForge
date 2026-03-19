# GEMM Autoresearch Report

## Top Pattern Signals
- tiling: 117.4850
- vectorize: 94.7040
- pack: 49.9270
- fuse: 22.5000
- jit: 6.5000
- reorder: 0.7140

## Best Per Shape
- s128: variant=blocked_pack bm/bn/bk=64/96/64 threads=16 simd=True gflops=52.937 lat_p50=0.079ms score=0.9700
- s256: variant=blocked bm/bn/bk=96/64/128 threads=16 simd=True gflops=147.990 lat_p50=0.227ms score=0.9753
- m512x1024x256: variant=blocked_pack bm/bn/bk=128/64/96 threads=16 simd=True gflops=331.549 lat_p50=0.810ms score=0.9700
- m1024: variant=blocked bm/bn/bk=128/128/128 threads=16 simd=True gflops=393.347 lat_p50=5.460ms score=1.0000
- l2048x1024x2048: variant=blocked_pack bm/bn/bk=128/128/64 threads=16 simd=True gflops=445.374 lat_p50=19.287ms score=0.9700

## Bucket Policy
- medium: blocked|96|64|128|False|False|True|16|2 avg_score=0.9753
- large: blocked|128|128|128|False|False|True|16|1 avg_score=1.0000

## Trial Stats
- total_trials: 120
- failed_trials: 1
