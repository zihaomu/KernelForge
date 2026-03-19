#include "gemm_kernels.h"

#include <algorithm>

#if KC_HAS_OPENMP
#include <omp.h>
#endif

namespace kc {

void gemm_blocked_f32(const float* a, const float* b, float* c, const GemmConfig& cfg) {
  const int m = cfg.m;
  const int n = cfg.n;
  const int k = cfg.k;
  const int bm = std::max(1, cfg.bm);
  const int bn = std::max(1, cfg.bn);
  const int bk = std::max(1, cfg.bk);

  std::fill(c, c + static_cast<size_t>(m) * n, 0.0f);

#if KC_HAS_OPENMP
  if (cfg.threads > 0) omp_set_num_threads(cfg.threads);
#pragma omp parallel for schedule(static) collapse(2) if (cfg.threads > 1)
#endif
  for (int ii = 0; ii < m; ii += bm) {
    for (int jj = 0; jj < n; jj += bn) {
      const int i_end = std::min(ii + bm, m);
      const int j_end = std::min(jj + bn, n);
      for (int kk = 0; kk < k; kk += bk) {
        const int k_end = std::min(kk + bk, k);
        for (int i = ii; i < i_end; ++i) {
          for (int p = kk; p < k_end; ++p) {
            const float av = a[i * k + p];
            if (cfg.simd) {
#if defined(__clang__) || defined(__GNUC__)
#pragma omp simd
#endif
              for (int j = jj; j < j_end; ++j) {
                c[i * n + j] += av * b[p * n + j];
              }
            } else {
              for (int j = jj; j < j_end; ++j) {
                c[i * n + j] += av * b[p * n + j];
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace kc

