#include "gemm_kernels.h"
#include "fp16_utils.h"

#include <algorithm>
#include <vector>

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
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      if (cfg.simd) {
#if defined(__clang__) || defined(__GNUC__)
#pragma omp simd reduction(+ : sum)
#endif
        for (int p = 0; p < k; ++p) {
          sum += a[i * k + p] * b[p * n + j];
        }
      } else {
        for (int p = 0; p < k; ++p) {
          sum += a[i * k + p] * b[p * n + j];
        }
      }
      c[i * n + j] = sum;
    }
  }
}

void gemm_naive_f16_f16(const uint16_t* a, const uint16_t* b, uint16_t* c, const GemmConfig& cfg) {
  const int m = cfg.m;
  const int n = cfg.n;
  const int k = cfg.k;
  std::vector<float> c_acc(static_cast<size_t>(m) * n, 0.0f);

#if KC_HAS_OPENMP
  if (cfg.threads > 0) omp_set_num_threads(cfg.threads);
#pragma omp parallel for schedule(static) if (cfg.threads > 1)
#endif
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int p = 0; p < k; ++p) {
        sum += fp16_to_float(a[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(p)]) *
               fp16_to_float(b[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(j)]);
      }
      c_acc[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] = sum;
    }
  }

  for (size_t idx = 0; idx < c_acc.size(); ++idx) {
    c[idx] = float_to_fp16(c_acc[idx]);
  }
}

void gemm_naive_i8_i32(const int8_t* a, const int8_t* b, int32_t* c, const GemmConfig& cfg) {
  const int m = cfg.m;
  const int n = cfg.n;
  const int k = cfg.k;
  std::fill(c, c + static_cast<size_t>(m) * n, int32_t{0});

#if KC_HAS_OPENMP
  if (cfg.threads > 0) omp_set_num_threads(cfg.threads);
#pragma omp parallel for schedule(static) if (cfg.threads > 1)
#endif
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      int32_t sum = 0;
      for (int p = 0; p < k; ++p) {
        sum += static_cast<int32_t>(a[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(p)]) *
               static_cast<int32_t>(b[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(j)]);
      }
      c[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] = sum;
    }
  }
}

}  // namespace kc
