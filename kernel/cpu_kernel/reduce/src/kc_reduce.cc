#include "kc_reduce.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <xsimd/xsimd.hpp>

int kc_reduce_sum_lastdim_f32(const float* input, float* output, int32_t rows, int32_t cols) {
  if (!input || !output) return 1;
  if (rows <= 0 || cols <= 0) return 2;

  using batch = xsimd::batch<float>;
  constexpr int V = (int)batch::size;

  for (int32_t r = 0; r < rows; ++r) {
    const float* x = input + (int64_t)r * cols;
    batch acc = batch(0.0f);
    int32_t c = 0;
    const int32_t cols_vec = (cols / V) * V;
    for (; c < cols_vec; c += V) acc += xsimd::load_unaligned(x + c);
    float sum = xsimd::reduce_add(acc);
    for (; c < cols; ++c) sum += x[c];
    output[r] = sum;
  }
  return 0;
}

int kc_reduce_max_lastdim_f32(const float* input, float* output, int32_t rows, int32_t cols) {
  if (!input || !output) return 1;
  if (rows <= 0 || cols <= 0) return 2;

  using batch = xsimd::batch<float>;
  constexpr int V = (int)batch::size;

  for (int32_t r = 0; r < rows; ++r) {
    const float* x = input + (int64_t)r * cols;
    int32_t c = 0;
    float maxv = x[0];
    if (cols >= V) {
      batch vmax = xsimd::load_unaligned(x);
      c = V;
      const int32_t cols_vec = (cols / V) * V;
      for (; c < cols_vec; c += V) vmax = xsimd::max(vmax, xsimd::load_unaligned(x + c));
      maxv = xsimd::reduce_max(vmax);
    } else {
      c = 1;
    }
    for (; c < cols; ++c) maxv = std::max(maxv, x[c]);
    output[r] = maxv;
  }
  return 0;
}

