#include "gemm_kernels.h"

#include <algorithm>

#if KC_HAS_OPENMP
#include <omp.h>
#endif

namespace kc {

void gemm_naive_f32(const float* a, const float* b, float* c, const GemmConfig& cfg) {
  const int m = cfg.m;
  const int n = cfg.n;
  const int k = cfg.k;
  std::fill(c, c + static_cast<size_t>(m) * n, 0.0f);

#if KC_HAS_OPENMP
  if (cfg.threads > 0) omp_set_num_threads(cfg.threads);
#pragma omp parallel for schedule(static) if (cfg.threads > 1)
#endif
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

