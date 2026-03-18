#include "kc_rmsnorm.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "npy.hpp"

namespace {

static std::string data_path(const std::string& name) {
  return std::string("kernel/cpu_kernel/rmsnorm/test/data/") + name;
}

}  // namespace

TEST(kc_rmsnorm, accuracy_lastdim_f32) {
  const auto params = npy::read_npy<int32_t>(data_path("params.npy"));
  ASSERT_EQ(params.shape.size(), 1u);
  ASSERT_EQ(params.shape[0], 2u);
  const int32_t rows = params.data[0];
  const int32_t cols = params.data[1];

  const auto eps_npy = npy::read_npy<float>(data_path("eps.npy"));
  ASSERT_EQ(eps_npy.shape.size(), 1u);
  ASSERT_EQ(eps_npy.shape[0], 1u);
  const float eps = eps_npy.data[0];

  const auto x = npy::read_npy<float>(data_path("input.npy"));
  const auto w = npy::read_npy<float>(data_path("weight.npy"));
  const auto ref = npy::read_npy<float>(data_path("output.npy"));
  ASSERT_EQ((int32_t)x.shape[0], rows);
  ASSERT_EQ((int32_t)x.shape[1], cols);
  ASSERT_EQ((int32_t)w.shape[0], cols);
  ASSERT_EQ((int32_t)ref.shape[0], rows);
  ASSERT_EQ((int32_t)ref.shape[1], cols);

  std::vector<float> y((int64_t)rows * cols, 0.0f);
  ASSERT_EQ(kc_rmsnorm_lastdim_f32(x.data.data(), w.data.data(), y.data(), rows, cols, eps), 0);

  double max_abs = 0.0;
  double max_rel = 0.0;
  for (size_t i = 0; i < y.size(); ++i) {
    const double a = (double)y[i];
    const double b = (double)ref.data[i];
    const double abs_err = std::abs(a - b);
    const double rel_err = abs_err / (std::abs(b) + 1e-9);
    max_abs = std::max(max_abs, abs_err);
    max_rel = std::max(max_rel, rel_err);
  }

  EXPECT_LE(max_abs, 5e-5);
  EXPECT_LE(max_rel, 5e-5);
}

