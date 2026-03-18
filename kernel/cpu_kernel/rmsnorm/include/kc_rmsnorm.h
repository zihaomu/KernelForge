#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// v1: RMSNorm along last dim for a 2D tensor [rows, cols] (row-major).
//
// y[r, c] = x[r, c] * rsqrt(mean(x[r,:]^2) + eps) * weight[c]
//
// input: rows*cols float32
// weight: cols float32
// output: rows*cols float32
//
// Returns 0 on success.
int kc_rmsnorm_lastdim_f32(const float* input,
                           const float* weight,
                           float* output,
                           int32_t rows,
                           int32_t cols,
                           float eps);

#ifdef __cplusplus
}  // extern "C"
#endif

