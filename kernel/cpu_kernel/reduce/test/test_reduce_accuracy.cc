#include "kc_reduce.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "npy.hpp"

namespace {

static std::string data_path(const std::string& name) {
  return std::string("kernel/cpu_kernel/reduce/test/data/") + name;
}

}  // namespace

TEST(kc_reduce, accuracy_sum_max_lastdim_f32) {
  const auto params = npy::read_npy<int32_t>(data_path("params.npy"));
  ASSERT_EQ(params.shape.size(), 1u);
  ASSERT_EQ(params.shape[0], 2u);
  const int32_t rows = params.data[0];
  const int32_t cols = params.data[1];

  const auto x = npy::read_npy<float>(data_path("input.npy"));
  const auto ref_sum = npy::read_npy<float>(data_path("sum.npy"));
  const auto ref_max = npy::read_npy<float>(data_path("max.npy"));
  ASSERT_EQ((int32_t)x.shape[0], rows);
  ASSERT_EQ((int32_t)x.shape[1], cols);
  ASSERT_EQ((int32_t)ref_sum.shape[0], rows);
  ASSERT_EQ((int32_t)ref_max.shape[0], rows);

  std::vector<float> out_sum(rows, 0.0f);
  std::vector<float> out_max(rows, 0.0f);
  ASSERT_EQ(kc_reduce_sum_lastdim_f32(x.data.data(), out_sum.data(), rows, cols), 0);
  ASSERT_EQ(kc_reduce_max_lastdim_f32(x.data.data(), out_max.data(), rows, cols), 0);

  double max_abs_sum = 0.0;
  double max_abs_max = 0.0;
  for (int32_t r = 0; r < rows; ++r) {
    max_abs_sum = std::max(max_abs_sum, std::abs((double)out_sum[r] - (double)ref_sum.data[r]));
    max_abs_max = std::max(max_abs_max, std::abs((double)out_max[r] - (double)ref_max.data[r]));
  }
  // Different summation orders (SIMD reduction) can produce tiny FP32 diffs.
  EXPECT_LE(max_abs_sum, 2e-5);
  EXPECT_LE(max_abs_max, 1e-6);
}
