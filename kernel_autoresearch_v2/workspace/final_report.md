# CPU GEMM Autoresearch Final Report

- generated_at: 2026-03-19T18:07:27

## Bucket Summary

| bucket | baseline_latency_us | baseline_gflops | best_score | best_candidate |
|---|---:|---:|---:|---|
| large | 103197.000 | 20.809600 | 19.403706 | `blocked_pack_simd/bm128/bn64/bk128/th16/uk2/simd1` |
| medium | 5019.841 | 18.734333 | 7.157048 | `blocked_pack/bm128/bn64/bk128/th16/uk1/simd1` |

## Notes

- correctness gate is always prior to performance gate
- keep/revert follows min_improve_ratio policy
