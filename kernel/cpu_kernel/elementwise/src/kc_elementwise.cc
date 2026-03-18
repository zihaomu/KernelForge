#include "kc_elementwise.h"

#include <cstdint>

#include <xsimd/xsimd.hpp>

int kc_add_f32(const float* a, const float* b, float* out, int32_t n) {
  if (!a || !b || !out) return 1;
  if (n < 0) return 2;
  if (n == 0) return 0;

  using batch = xsimd::batch<float>;
  constexpr int V = (int)batch::size;

  int32_t i = 0;
  const int32_t n_vec = (n / V) * V;
  for (; i < n_vec; i += V) {
    const batch va = xsimd::load_unaligned(a + i);
    const batch vb = xsimd::load_unaligned(b + i);
    xsimd::store_unaligned(out + i, va + vb);
  }
  for (; i < n; ++i) out[i] = a[i] + b[i];

  return 0;
}

