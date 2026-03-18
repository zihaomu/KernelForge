#include "kc_conv3x3.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "npy.hpp"

namespace {

template <typename T>
static std::vector<int64_t> to_i64(const std::vector<size_t>& shape) {
  std::vector<int64_t> out;
  out.reserve(shape.size());
  for (size_t v : shape) out.push_back(static_cast<int64_t>(v));
  return out;
}

static std::string data_path(const std::string& name) {
  // CTest runs from build dir; keep path relative to source tree.
  return std::string("kernel/cpu_kernel/conv_3x3/test/data/") + name;
}

}  // namespace

TEST(kc_conv3x3, accuracy_basic_nchw_f32) {
  const auto params_npy = npy::read_npy<int32_t>(data_path("params.npy"));
  ASSERT_EQ(params_npy.shape.size(), 1u);
  ASSERT_EQ(params_npy.shape[0], 9u);
  const int32_t* p = params_npy.data.data();
  const int32_t n = p[0];
  const int32_t cin = p[1];
  const int32_t h = p[2];
  const int32_t w = p[3];
  const int32_t cout = p[4];
  const int32_t stride_h = p[5];
  const int32_t stride_w = p[6];
  const int32_t pad_h = p[7];
  const int32_t pad_w = p[8];

  const auto x_npy = npy::read_npy<float>(data_path("input.npy"));
  const auto wt_npy = npy::read_npy<float>(data_path("weight.npy"));
  const auto b_npy = npy::read_npy<float>(data_path("bias.npy"));
  const auto y_ref_npy = npy::read_npy<float>(data_path("output.npy"));

  ASSERT_EQ(to_i64<float>(x_npy.shape), (std::vector<int64_t>{n, cin, h, w}));
  ASSERT_EQ(to_i64<float>(wt_npy.shape), (std::vector<int64_t>{cout, cin, 3, 3}));
  ASSERT_EQ(to_i64<float>(b_npy.shape), (std::vector<int64_t>{cout}));

  const int32_t hout = (h + 2 * pad_h - 3) / stride_h + 1;
  const int32_t wout = (w + 2 * pad_w - 3) / stride_w + 1;
  ASSERT_EQ(to_i64<float>(y_ref_npy.shape), (std::vector<int64_t>{n, cout, hout, wout}));

  std::vector<float> y_out(static_cast<size_t>(n) * cout * hout * wout, 0.0f);
  const int rc = kc_conv3x3_nchw_f32(
      x_npy.data.data(),
      wt_npy.data.data(),
      b_npy.data.data(),
      y_out.data(),
      n,
      cin,
      h,
      w,
      cout,
      stride_h,
      stride_w,
      pad_h,
      pad_w);
  ASSERT_EQ(rc, 0);

  const float* y_ref = y_ref_npy.data.data();
  double max_abs = 0.0;
  double max_rel = 0.0;
  for (size_t i = 0; i < y_out.size(); ++i) {
    const double a = static_cast<double>(y_out[i]);
    const double b = static_cast<double>(y_ref[i]);
    const double abs_err = std::abs(a - b);
    const double rel_err = abs_err / (std::abs(b) + 1e-9);
    max_abs = std::max(max_abs, abs_err);
    max_rel = std::max(max_rel, rel_err);
  }

  // xsimd fast-path is still float32; tolerate small FP differences.
  EXPECT_LE(max_abs, 5e-4);
  EXPECT_LE(max_rel, 5e-4);
}

