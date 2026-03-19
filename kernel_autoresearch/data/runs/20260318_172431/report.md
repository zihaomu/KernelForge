# GEMM Autoresearch Report

## Top Pattern Signals
- tiling: 117.2810
- vectorize: 94.7040
- pack: 49.9270
- fuse: 22.5000
- jit: 6.5000
- reorder: 0.7140

## Best Per Shape
- s128: variant=blocked_pack bm/bn/bk=64/96/96 threads=16 simd=True gflops=60.246 lat_p50=0.070ms score=0.9700
- s256: variant=blocked_pack bm/bn/bk=96/64/128 threads=16 simd=True gflops=137.661 lat_p50=0.244ms score=0.9700
- m512x1024x256: variant=blocked_pack bm/bn/bk=128/64/96 threads=16 simd=True gflops=334.385 lat_p50=0.803ms score=0.9700
- m1024: variant=blocked bm/bn/bk=128/128/128 threads=16 simd=True gflops=414.963 lat_p50=5.175ms score=1.0000
- l2048x1024x2048: variant=blocked_pack bm/bn/bk=64/128/128 threads=16 simd=True gflops=453.041 lat_p50=18.961ms score=0.9700

## Bucket Policy
- medium: blocked_pack|64|96|96|True|True|True|16|1 avg_score=0.9700
- large: blocked|128|128|128|False|False|True|16|1 avg_score=1.0000

## Trial Stats
- total_trials: 120
- failed_trials: 1
