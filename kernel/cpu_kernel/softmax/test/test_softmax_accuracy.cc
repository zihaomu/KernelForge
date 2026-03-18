#include "kc_softmax.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "npy.hpp"

namespace {

static std::string data_path(const std::string& name) {
  return std::string("kernel/cpu_kernel/softmax/test/data/") + name;
}

}  // namespace

TEST(kc_softmax, accuracy_lastdim_f32) {
  const auto params_npy = npy::read_npy<int32_t>(data_path("params.npy"));
  ASSERT_EQ(params_npy.shape.size(), 1u);
  ASSERT_EQ(params_npy.shape[0], 2u);
  const int32_t rows = params_npy.data[0];
  const int32_t cols = params_npy.data[1];

  const auto x_npy = npy::read_npy<float>(data_path("input.npy"));
  const auto y_ref_npy = npy::read_npy<float>(data_path("output.npy"));
  ASSERT_EQ((int32_t)x_npy.shape[0], rows);
  ASSERT_EQ((int32_t)x_npy.shape[1], cols);
  ASSERT_EQ((int32_t)y_ref_npy.shape[0], rows);
  ASSERT_EQ((int32_t)y_ref_npy.shape[1], cols);

  std::vector<float> y(rows * (int64_t)cols, 0.0f);
  const int rc = kc_softmax_lastdim_f32(x_npy.data.data(), y.data(), rows, cols);
  ASSERT_EQ(rc, 0);

  double max_abs = 0.0;
  double max_rel = 0.0;
  for (size_t i = 0; i < y.size(); ++i) {
    const double a = (double)y[i];
    const double b = (double)y_ref_npy.data[i];
    const double abs_err = std::abs(a - b);
    const double rel_err = abs_err / (std::abs(b) + 1e-9);
    max_abs = std::max(max_abs, abs_err);
    max_rel = std::max(max_rel, rel_err);
  }

  EXPECT_LE(max_abs, 1e-6);
  EXPECT_LE(max_rel, 1e-5);
}

