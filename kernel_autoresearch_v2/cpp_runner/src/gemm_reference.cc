#include "gemm_kernels.h"

#include <algorithm>

namespace kc {

void gemm_reference_f32(const float* a, const float* b, float* c, const GemmConfig& cfg) {
  const int m = cfg.m;
  const int n = cfg.n;
  const int k = cfg.k;
  std::fill(c, c + static_cast<size_t>(m) * n, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int p = 0; p < k; ++p) {
      const float av = a[i * k + p];
      for (int j = 0; j < n; ++j) {
        c[i * n + j] += av * b[p * n + j];
      }
    }
  }
}

}  // namespace kc

