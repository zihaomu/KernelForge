#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// v1: reduce sum / max over last dim for a 2D tensor [rows, cols] (row-major).
//
// input:  rows*cols float32
// output: rows float32
//
// Returns 0 on success.
int kc_reduce_sum_lastdim_f32(const float* input, float* output, int32_t rows, int32_t cols);
int kc_reduce_max_lastdim_f32(const float* input, float* output, int32_t rows, int32_t cols);

#ifdef __cplusplus
}  // extern "C"
#endif

