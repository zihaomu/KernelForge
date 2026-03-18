#include "kc_conv3x3.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <xsimd/xsimd.hpp>

namespace {

inline int32_t out_size_1d(int32_t in, int32_t pad, int32_t k, int32_t stride) {
  // floor((in + 2*pad - k)/stride) + 1
  return (in + 2 * pad - k) / stride + 1;
}

void conv3x3_nchw_f32_scalar(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int32_t n,
    int32_t cin,
    int32_t h,
    int32_t w,
    int32_t cout,
    int32_t stride_h,
    int32_t stride_w,
    int32_t pad_h,
    int32_t pad_w) {
  const int32_t hout = out_size_1d(h, pad_h, /*k=*/3, stride_h);
  const int32_t wout = out_size_1d(w, pad_w, /*k=*/3, stride_w);

  auto in_at = [&](int32_t ni, int32_t ci, int32_t yi, int32_t xi) -> float {
    if (yi < 0 || yi >= h || xi < 0 || xi >= w) return 0.0f;
    const int64_t idx = ((int64_t)ni * cin + ci) * (int64_t)h * w + (int64_t)yi * w + xi;
    return input[idx];
  };

  auto w_at = [&](int32_t co, int32_t ci, int32_t ky, int32_t kx) -> float {
    const int64_t idx = ((int64_t)co * cin + ci) * 9 + ky * 3 + kx;
    return weight[idx];
  };

  for (int32_t ni = 0; ni < n; ++ni) {
    for (int32_t co = 0; co < cout; ++co) {
      for (int32_t yo = 0; yo < hout; ++yo) {
        for (int32_t xo = 0; xo < wout; ++xo) {
          float acc = bias ? bias[co] : 0.0f;
          const int32_t y0 = yo * stride_h - pad_h;
          const int32_t x0 = xo * stride_w - pad_w;
          for (int32_t ci = 0; ci < cin; ++ci) {
            // unroll 3x3
            acc += in_at(ni, ci, y0 + 0, x0 + 0) * w_at(co, ci, 0, 0);
            acc += in_at(ni, ci, y0 + 0, x0 + 1) * w_at(co, ci, 0, 1);
            acc += in_at(ni, ci, y0 + 0, x0 + 2) * w_at(co, ci, 0, 2);
            acc += in_at(ni, ci, y0 + 1, x0 + 0) * w_at(co, ci, 1, 0);
            acc += in_at(ni, ci, y0 + 1, x0 + 1) * w_at(co, ci, 1, 1);
            acc += in_at(ni, ci, y0 + 1, x0 + 2) * w_at(co, ci, 1, 2);
            acc += in_at(ni, ci, y0 + 2, x0 + 0) * w_at(co, ci, 2, 0);
            acc += in_at(ni, ci, y0 + 2, x0 + 1) * w_at(co, ci, 2, 1);
            acc += in_at(ni, ci, y0 + 2, x0 + 2) * w_at(co, ci, 2, 2);
          }
          const int64_t out_idx = ((int64_t)ni * cout + co) * (int64_t)hout * wout + (int64_t)yo * wout + xo;
          output[out_idx] = acc;
        }
      }
    }
  }
}

// Fast path: stride=1, pad=1, so hout=h, wout=w. Vectorize across output X for interior region.
void conv3x3_nchw_f32_s1p1_xsimd(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int32_t n,
    int32_t cin,
    int32_t h,
    int32_t w,
    int32_t cout) {
  using b = xsimd::batch<float>;
  constexpr int V = (int)b::size;

  const int32_t hout = h;
  const int32_t wout = w;

  auto in_ptr = [&](int32_t ni, int32_t ci, int32_t yi, int32_t xi) -> const float* {
    const int64_t idx = ((int64_t)ni * cin + ci) * (int64_t)h * w + (int64_t)yi * w + xi;
    return input + idx;
  };

  auto out_ptr = [&](int32_t ni, int32_t co, int32_t yo, int32_t xo) -> float* {
    const int64_t idx = ((int64_t)ni * cout + co) * (int64_t)hout * wout + (int64_t)yo * wout + xo;
    return output + idx;
  };

  auto w_at = [&](int32_t co, int32_t ci, int32_t ky, int32_t kx) -> float {
    const int64_t idx = ((int64_t)co * cin + ci) * 9 + ky * 3 + kx;
    return weight[idx];
  };

  // Edges (padding) scalar; interior vectorized.
  for (int32_t ni = 0; ni < n; ++ni) {
    for (int32_t co = 0; co < cout; ++co) {
      // Top row / bottom row
      for (int32_t yo : {0, h - 1}) {
        if (yo < 0 || yo >= h) continue;
        for (int32_t xo = 0; xo < w; ++xo) {
          float acc = bias ? bias[co] : 0.0f;
          for (int32_t ci = 0; ci < cin; ++ci) {
            for (int32_t ky = 0; ky < 3; ++ky) {
              const int32_t yi = yo + ky - 1;
              if (yi < 0 || yi >= h) continue;
              for (int32_t kx = 0; kx < 3; ++kx) {
                const int32_t xi = xo + kx - 1;
                if (xi < 0 || xi >= w) continue;
                acc += input[((int64_t)ni * cin + ci) * (int64_t)h * w + (int64_t)yi * w + xi] * w_at(co, ci, ky, kx);
              }
            }
          }
          *out_ptr(ni, co, yo, xo) = acc;
        }
      }

      // Middle rows
      for (int32_t yo = 1; yo < h - 1; ++yo) {
        // Left / right edge scalar
        for (int32_t xo : {0, w - 1}) {
          if (xo < 0 || xo >= w) continue;
          float acc = bias ? bias[co] : 0.0f;
          for (int32_t ci = 0; ci < cin; ++ci) {
            for (int32_t ky = 0; ky < 3; ++ky) {
              const int32_t yi = yo + ky - 1;
              for (int32_t kx = 0; kx < 3; ++kx) {
                const int32_t xi = xo + kx - 1;
                if (xi < 0 || xi >= w) continue;
                acc += input[((int64_t)ni * cin + ci) * (int64_t)h * w + (int64_t)yi * w + xi] * w_at(co, ci, ky, kx);
              }
            }
          }
          *out_ptr(ni, co, yo, xo) = acc;
        }

        // Interior vectorized: xo in [1, w-2]
        int32_t xo = 1;
        const int32_t xo_end = w - 1;  // exclusive of last edge
        const int32_t xo_vec_end = std::max<int32_t>(xo, xo_end - V);  // last start that keeps loads in bounds

        for (; xo <= xo_vec_end; xo += V) {
          b acc = b(bias ? bias[co] : 0.0f);
          for (int32_t ci = 0; ci < cin; ++ci) {
            // ky=0..2, kx=0..2
            for (int32_t ky = 0; ky < 3; ++ky) {
              const int32_t yi = yo + ky - 1;
              const float* row = in_ptr(ni, ci, yi, /*xi=*/0);
              const float w0 = w_at(co, ci, ky, 0);
              const float w1 = w_at(co, ci, ky, 1);
              const float w2 = w_at(co, ci, ky, 2);
              const b v0 = xsimd::load_unaligned(row + (xo - 1));
              const b v1 = xsimd::load_unaligned(row + (xo + 0));
              const b v2 = xsimd::load_unaligned(row + (xo + 1));
              acc = xsimd::fma(v0, b(w0), acc);
              acc = xsimd::fma(v1, b(w1), acc);
              acc = xsimd::fma(v2, b(w2), acc);
            }
          }
          xsimd::store_unaligned(out_ptr(ni, co, yo, xo), acc);
        }

        // Tail (still interior but not vectorizable)
        for (; xo < w - 1; ++xo) {
          float acc = bias ? bias[co] : 0.0f;
          for (int32_t ci = 0; ci < cin; ++ci) {
            for (int32_t ky = 0; ky < 3; ++ky) {
              const int32_t yi = yo + ky - 1;
              for (int32_t kx = 0; kx < 3; ++kx) {
                const int32_t xi = xo + kx - 1;
                acc += input[((int64_t)ni * cin + ci) * (int64_t)h * w + (int64_t)yi * w + xi] * w_at(co, ci, ky, kx);
              }
            }
          }
          *out_ptr(ni, co, yo, xo) = acc;
        }
      }
    }
  }
}

}  // namespace

int kc_conv3x3_nchw_f32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int32_t n,
    int32_t cin,
    int32_t h,
    int32_t w,
    int32_t cout,
    int32_t stride_h,
    int32_t stride_w,
    int32_t pad_h,
    int32_t pad_w) {
  if (!input || !weight || !output) return 1;
  if (n <= 0 || cin <= 0 || h <= 0 || w <= 0 || cout <= 0) return 2;
  if (stride_h <= 0 || stride_w <= 0) return 3;
  if (pad_h < 0 || pad_w < 0) return 4;

  const int32_t hout = out_size_1d(h, pad_h, /*k=*/3, stride_h);
  const int32_t wout = out_size_1d(w, pad_w, /*k=*/3, stride_w);
  if (hout <= 0 || wout <= 0) return 5;

  if (stride_h == 1 && stride_w == 1 && pad_h == 1 && pad_w == 1) {
    conv3x3_nchw_f32_s1p1_xsimd(input, weight, bias, output, n, cin, h, w, cout);
    return 0;
  }

  conv3x3_nchw_f32_scalar(input, weight, bias, output, n, cin, h, w, cout, stride_h, stride_w, pad_h, pad_w);
  return 0;
}

