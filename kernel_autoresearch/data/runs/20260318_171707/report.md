# GEMM Autoresearch Report

## Top Pattern Signals
- tiling: 117.4850
- vectorize: 94.5000
- pack: 49.9270
- fuse: 22.5000
- jit: 6.5000
- reorder: 0.7140

## Best Per Shape
- s128: variant=blocked_pack bm/bn/bk=64/128/96 threads=4 simd=True gflops=57.749 lat_p50=0.073ms score=0.9200
- s256: variant=blocked bm/bn/bk=96/64/128 threads=16 simd=True gflops=153.391 lat_p50=0.219ms score=0.9200
- m512x1024x256: variant=blocked_pack bm/bn/bk=128/64/96 threads=16 simd=True gflops=321.654 lat_p50=0.835ms score=0.9200
- m1024: variant=blocked bm/bn/bk=128/128/96 threads=16 simd=False gflops=388.179 lat_p50=5.532ms score=0.9200
- l2048x1024x2048: variant=blocked_pack bm/bn/bk=128/128/64 threads=16 simd=True gflops=446.078 lat_p50=19.257ms score=0.9200

## Bucket Policy
- medium: blocked_pack|64|128|96|True|True|True|4|1 avg_score=0.9200
- large: blocked|128|128|96|False|False|False|16|1 avg_score=0.9200

## Trial Stats
- total_trials: 120
- failed_trials: 1
