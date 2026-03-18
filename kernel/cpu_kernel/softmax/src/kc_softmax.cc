#include "kc_softmax.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

int kc_softmax_lastdim_f32(const float* input, float* output, int32_t rows, int32_t cols) {
  if (!input || !output) return 1;
  if (rows <= 0 || cols <= 0) return 2;

  for (int32_t r = 0; r < rows; ++r) {
    const float* x = input + (int64_t)r * cols;
    float* y = output + (int64_t)r * cols;

    float maxv = x[0];
    for (int32_t c = 1; c < cols; ++c) maxv = std::max(maxv, x[c]);

    float sum = 0.0f;
    for (int32_t c = 0; c < cols; ++c) sum += std::exp(x[c] - maxv);
    const float inv = 1.0f / sum;

    for (int32_t c = 0; c < cols; ++c) y[c] = std::exp(x[c] - maxv) * inv;
  }

  return 0;
}

