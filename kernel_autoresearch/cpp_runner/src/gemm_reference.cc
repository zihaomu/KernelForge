#include "gemm_kernels.h"
#include "fp16_utils.h"

#include <algorithm>
#include <vector>

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

void gemm_reference_f16_f16(const uint16_t* a, const uint16_t* b, uint16_t* c, const GemmConfig& cfg) {
  const int m = cfg.m;
  const int n = cfg.n;
  const int k = cfg.k;
  std::vector<float> c_acc(static_cast<size_t>(m) * n, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int p = 0; p < k; ++p) {
      const float av = fp16_to_float(a[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(p)]);
      for (int j = 0; j < n; ++j) {
        c_acc[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] +=
            av * fp16_to_float(b[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(j)]);
      }
    }
  }
  for (size_t idx = 0; idx < c_acc.size(); ++idx) {
    c[idx] = float_to_fp16(c_acc[idx]);
  }
}

void gemm_reference_i8_i32(const int8_t* a, const int8_t* b, int32_t* c, const GemmConfig& cfg) {
  const int m = cfg.m;
  const int n = cfg.n;
  const int k = cfg.k;
  std::fill(c, c + static_cast<size_t>(m) * n, int32_t{0});
  for (int i = 0; i < m; ++i) {
    for (int p = 0; p < k; ++p) {
      const int32_t av = static_cast<int32_t>(a[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(p)]);
      for (int j = 0; j < n; ++j) {
        c[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] +=
            av * static_cast<int32_t>(b[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(j)]);
      }
    }
  }
}

}  // namespace kc
