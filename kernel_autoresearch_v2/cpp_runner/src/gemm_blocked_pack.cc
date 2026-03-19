#include "gemm_kernels.h"

#include <algorithm>
#include <vector>

#if KC_HAS_OPENMP
#include <omp.h>
#endif

namespace kc {

void gemm_blocked_pack_f32(const float* a, const float* b, float* c, const GemmConfig& cfg) {
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
      const int cur_m = i_end - ii;
      const int cur_n = j_end - jj;

      for (int kk = 0; kk < k; kk += bk) {
        const int k_end = std::min(kk + bk, k);
        const int cur_k = k_end - kk;

        std::vector<float> packed_b(static_cast<size_t>(cur_k) * cur_n);
        for (int p = 0; p < cur_k; ++p) {
          for (int j = 0; j < cur_n; ++j) {
            packed_b[p * cur_n + j] = b[(kk + p) * n + (jj + j)];
          }
        }

        std::vector<float> packed_a;
        if (cfg.pack_a) {
          packed_a.resize(static_cast<size_t>(cur_m) * cur_k);
          for (int i = 0; i < cur_m; ++i) {
            for (int p = 0; p < cur_k; ++p) {
              packed_a[i * cur_k + p] = a[(ii + i) * k + (kk + p)];
            }
          }
        }

        for (int i = 0; i < cur_m; ++i) {
          for (int p = 0; p < cur_k; ++p) {
            const float av = cfg.pack_a ? packed_a[i * cur_k + p] : a[(ii + i) * k + (kk + p)];
            if (cfg.simd) {
#if defined(__clang__) || defined(__GNUC__)
#pragma omp simd
#endif
              for (int j = 0; j < cur_n; ++j) {
                c[(ii + i) * n + (jj + j)] += av * packed_b[p * cur_n + j];
              }
            } else {
              for (int j = 0; j < cur_n; ++j) {
                c[(ii + i) * n + (jj + j)] += av * packed_b[p * cur_n + j];
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace kc

