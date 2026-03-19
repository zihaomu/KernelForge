#include "gemm_kernels.h"
#include "fp16_utils.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include <xsimd/xsimd.hpp>

#if KC_HAS_OPENMP
#include <omp.h>
#endif

namespace kc {

namespace {

inline void axpy_f32_xsimd(float* c_row, const float* b_row, float av, int len) {
  using batch = xsimd::batch<float>;
  constexpr int lanes = batch::size;
  const batch av_vec(av);
  int j = 0;
  for (; j + lanes <= len; j += lanes) {
    batch c_vec = batch::load_unaligned(c_row + j);
    const batch b_vec = batch::load_unaligned(b_row + j);
    c_vec = xsimd::fma(av_vec, b_vec, c_vec);
    c_vec.store_unaligned(c_row + j);
  }
  for (; j < len; ++j) {
    c_row[j] += av * b_row[j];
  }
}

inline void axpy_i8_i32_xsimd(int32_t* c_row, const int8_t* b_row, int8_t av, int len) {
  using b8 = xsimd::batch<int8_t>;
  using b16 = xsimd::batch<int16_t>;
  using b32 = xsimd::batch<int32_t>;
  constexpr int lanes8 = b8::size;
  constexpr int lanes16 = b16::size;
  constexpr int lanes32 = b32::size;

  const b8 av8(av);
  int j = 0;
  for (; j + lanes8 <= len; j += lanes8) {
    const b8 b8v = b8::load_unaligned(b_row + j);
    const auto b16pair = xsimd::widen(b8v);
    const auto av16pair = xsimd::widen(av8);
    for (int h = 0; h < 2; ++h) {
      const b16 prod16 = av16pair[static_cast<size_t>(h)] * b16pair[static_cast<size_t>(h)];
      const auto prod32pair = xsimd::widen(prod16);
      for (int q = 0; q < 2; ++q) {
        const int offset = j + h * lanes16 + q * lanes32;
        b32 c32 = b32::load_unaligned(c_row + offset);
        c32 += prod32pair[static_cast<size_t>(q)];
        c32.store_unaligned(c_row + offset);
      }
    }
  }

  const int32_t av32 = static_cast<int32_t>(av);
  for (; j < len; ++j) {
    c_row[j] += av32 * static_cast<int32_t>(b_row[j]);
  }
}

}  // namespace

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
            packed_b[static_cast<size_t>(p) * static_cast<size_t>(cur_n) + static_cast<size_t>(j)] =
                b[static_cast<size_t>(kk + p) * static_cast<size_t>(n) + static_cast<size_t>(jj + j)];
          }
        }

        std::vector<float> packed_a;
        if (cfg.pack_a) {
          packed_a.resize(static_cast<size_t>(cur_m) * cur_k);
          for (int i = 0; i < cur_m; ++i) {
            for (int p = 0; p < cur_k; ++p) {
              packed_a[static_cast<size_t>(i) * static_cast<size_t>(cur_k) + static_cast<size_t>(p)] =
                  a[static_cast<size_t>(ii + i) * static_cast<size_t>(k) + static_cast<size_t>(kk + p)];
            }
          }
        }

        for (int i = 0; i < cur_m; ++i) {
          float* c_row =
              c + static_cast<size_t>(ii + i) * static_cast<size_t>(n) + static_cast<size_t>(jj);
          for (int p = 0; p < cur_k; ++p) {
            const float av = cfg.pack_a
                                 ? packed_a[static_cast<size_t>(i) * static_cast<size_t>(cur_k) +
                                            static_cast<size_t>(p)]
                                 : a[static_cast<size_t>(ii + i) * static_cast<size_t>(k) +
                                     static_cast<size_t>(kk + p)];
            const float* b_row =
                packed_b.data() + static_cast<size_t>(p) * static_cast<size_t>(cur_n);
            if (cfg.simd) {
              axpy_f32_xsimd(c_row, b_row, av, cur_n);
            } else {
              for (int j = 0; j < cur_n; ++j) {
                c_row[j] += av * b_row[j];
              }
            }
          }
        }
      }
    }
  }
}

void gemm_blocked_pack_f16_f16(const uint16_t* a, const uint16_t* b, uint16_t* c, const GemmConfig& cfg) {
  const int m = cfg.m;
  const int n = cfg.n;
  const int k = cfg.k;
  const int bm = std::max(1, cfg.bm);
  const int bn = std::max(1, cfg.bn);
  const int bk = std::max(1, cfg.bk);

  std::vector<float> c_acc(static_cast<size_t>(m) * n, 0.0f);

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

        std::vector<uint16_t> packed_b(static_cast<size_t>(cur_k) * cur_n);
        for (int p = 0; p < cur_k; ++p) {
          for (int j = 0; j < cur_n; ++j) {
            packed_b[static_cast<size_t>(p) * static_cast<size_t>(cur_n) + static_cast<size_t>(j)] =
                b[static_cast<size_t>(kk + p) * static_cast<size_t>(n) + static_cast<size_t>(jj + j)];
          }
        }

        std::vector<uint16_t> packed_a;
        if (cfg.pack_a) {
          packed_a.resize(static_cast<size_t>(cur_m) * cur_k);
          for (int i = 0; i < cur_m; ++i) {
            for (int p = 0; p < cur_k; ++p) {
              packed_a[static_cast<size_t>(i) * static_cast<size_t>(cur_k) + static_cast<size_t>(p)] =
                  a[static_cast<size_t>(ii + i) * static_cast<size_t>(k) + static_cast<size_t>(kk + p)];
            }
          }
        }

        std::vector<float> packed_b_f(static_cast<size_t>(cur_k) * cur_n);
        for (int p = 0; p < cur_k; ++p) {
          for (int j = 0; j < cur_n; ++j) {
            packed_b_f[static_cast<size_t>(p) * static_cast<size_t>(cur_n) + static_cast<size_t>(j)] =
                fp16_to_float(packed_b[static_cast<size_t>(p) * static_cast<size_t>(cur_n) + static_cast<size_t>(j)]);
          }
        }

        for (int i = 0; i < cur_m; ++i) {
          float* c_row =
              c_acc.data() + static_cast<size_t>(ii + i) * static_cast<size_t>(n) + static_cast<size_t>(jj);
          for (int p = 0; p < cur_k; ++p) {
            const float av = cfg.pack_a
                                 ? fp16_to_float(
                                       packed_a[static_cast<size_t>(i) * static_cast<size_t>(cur_k) +
                                                static_cast<size_t>(p)])
                                 : fp16_to_float(
                                       a[static_cast<size_t>(ii + i) * static_cast<size_t>(k) +
                                         static_cast<size_t>(kk + p)]);
            const float* b_row =
                packed_b_f.data() + static_cast<size_t>(p) * static_cast<size_t>(cur_n);
            if (cfg.simd) {
              axpy_f32_xsimd(c_row, b_row, av, cur_n);
            } else {
              for (int j = 0; j < cur_n; ++j) {
                c_row[j] += av * b_row[j];
              }
            }
          }
        }
      }
    }
  }

  for (size_t idx = 0; idx < c_acc.size(); ++idx) {
    c[idx] = float_to_fp16(c_acc[idx]);
  }
}

void gemm_blocked_pack_i8_i32(const int8_t* a, const int8_t* b, int32_t* c, const GemmConfig& cfg) {
  const int m = cfg.m;
  const int n = cfg.n;
  const int k = cfg.k;
  const int bm = std::max(1, cfg.bm);
  const int bn = std::max(1, cfg.bn);
  const int bk = std::max(1, cfg.bk);

  std::fill(c, c + static_cast<size_t>(m) * n, int32_t{0});

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

        std::vector<int8_t> packed_b(static_cast<size_t>(cur_k) * cur_n);
        for (int p = 0; p < cur_k; ++p) {
          for (int j = 0; j < cur_n; ++j) {
            packed_b[static_cast<size_t>(p) * static_cast<size_t>(cur_n) + static_cast<size_t>(j)] =
                b[static_cast<size_t>(kk + p) * static_cast<size_t>(n) + static_cast<size_t>(jj + j)];
          }
        }

        std::vector<int8_t> packed_a;
        if (cfg.pack_a) {
          packed_a.resize(static_cast<size_t>(cur_m) * cur_k);
          for (int i = 0; i < cur_m; ++i) {
            for (int p = 0; p < cur_k; ++p) {
              packed_a[static_cast<size_t>(i) * static_cast<size_t>(cur_k) + static_cast<size_t>(p)] =
                  a[static_cast<size_t>(ii + i) * static_cast<size_t>(k) + static_cast<size_t>(kk + p)];
            }
          }
        }

        for (int i = 0; i < cur_m; ++i) {
          int32_t* c_row =
              c + static_cast<size_t>(ii + i) * static_cast<size_t>(n) + static_cast<size_t>(jj);
          for (int p = 0; p < cur_k; ++p) {
            const int8_t av = cfg.pack_a
                                  ? packed_a[static_cast<size_t>(i) * static_cast<size_t>(cur_k) +
                                             static_cast<size_t>(p)]
                                  : a[static_cast<size_t>(ii + i) * static_cast<size_t>(k) +
                                      static_cast<size_t>(kk + p)];
            const int8_t* b_row =
                packed_b.data() + static_cast<size_t>(p) * static_cast<size_t>(cur_n);
            if (cfg.simd) {
              axpy_i8_i32_xsimd(c_row, b_row, av, cur_n);
            } else {
              const int32_t av32 = static_cast<int32_t>(av);
              for (int j = 0; j < cur_n; ++j) {
                c_row[j] += av32 * static_cast<int32_t>(b_row[j]);
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace kc
