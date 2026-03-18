#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// v1: TopK along last dim for a 2D tensor [rows, cols] (row-major).
//
// - input:  rows*cols float32
// - out_values: rows*k float32
// - out_indices: rows*k int32 (indices in [0, cols))
//
// descend=1 means largest first.
// Returns 0 on success.
int kc_topk_lastdim_f32(const float* input,
                        float* out_values,
                        int32_t* out_indices,
                        int32_t rows,
                        int32_t cols,
                        int32_t k,
                        int32_t descend);

#ifdef __cplusplus
}  // extern "C"
#endif

