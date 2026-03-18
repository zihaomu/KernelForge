#include "kc_rmsnorm.h"

#include <cmath>
#include <cstdint>

#include <xsimd/xsimd.hpp>

int kc_rmsnorm_lastdim_f32(const float* input,
                           const float* weight,
                           float* output,
                           int32_t rows,
                           int32_t cols,
                           float eps) {
  if (!input || !weight || !output) return 1;
  if (rows <= 0 || cols <= 0) return 2;
  if (!(eps >= 0.0f)) return 3;

  using batch = xsimd::batch<float>;
  constexpr int V = (int)batch::size;

  for (int32_t r = 0; r < rows; ++r) {
    const float* x = input + (int64_t)r * cols;
    float* y = output + (int64_t)r * cols;

    // sumsq
    batch acc = batch(0.0f);
    int32_t c = 0;
    const int32_t cols_vec = (cols / V) * V;
    for (; c < cols_vec; c += V) {
      const batch vx = xsimd::load_unaligned(x + c);
      acc += vx * vx;
    }
    float sumsq = xsimd::reduce_add(acc);
    for (; c < cols; ++c) sumsq += x[c] * x[c];

    const float mean = sumsq / (float)cols;
    const float inv = 1.0f / std::sqrt(mean + eps);

    // y = x * inv * weight
    const batch vinv = batch(inv);
    c = 0;
    for (; c < cols_vec; c += V) {
      const batch vx = xsimd::load_unaligned(x + c);
      const batch vw = xsimd::load_unaligned(weight + c);
      xsimd::store_unaligned(y + c, vx * vinv * vw);
    }
    for (; c < cols; ++c) y[c] = x[c] * inv * weight[c];
  }

  return 0;
}

