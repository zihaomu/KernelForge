#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// v1: elementwise add for float32 arrays.
// out[i] = a[i] + b[i]
// Returns 0 on success.
int kc_add_f32(const float* a, const float* b, float* out, int32_t n);

#ifdef __cplusplus
}  // extern "C"
#endif

