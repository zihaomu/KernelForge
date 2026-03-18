#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// v1 API: NCHW, weight OIHW, bias optional.
//
// input:  [N, Cin, H, W]
// weight: [Cout, Cin, 3, 3]
// bias:   [Cout] (nullable -> treated as zeros)
// output: [N, Cout, Hout, Wout]
//
// Returns 0 on success, non-zero on invalid arguments.
int kc_conv3x3_nchw_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int32_t n,
    int32_t cin,
    int32_t h,
    int32_t w,
    int32_t cout,
    int32_t stride_h,
    int32_t stride_w,
    int32_t pad_h,
    int32_t pad_w);

#ifdef __cplusplus
}  // extern "C"
#endif

