#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// v1 API: softmax along last dimension for a 2D tensor [rows, cols].
//
// input:  rows*cols float32
// output: rows*cols float32
//
// Returns 0 on success.
int kc_softmax_lastdim_f32(const float* input, float* output, int32_t rows, int32_t cols);

#ifdef __cplusplus
}  // extern "C"
#endif

