#include "kc_elementwise.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "npy.hpp"

namespace {

static std::string data_path(const std::string& name) {
  return std::string("kernel/cpu_kernel/elementwise/test/data/") + name;
}

}  // namespace

TEST(kc_elementwise, accuracy_add_f32) {
  const auto params = npy::read_npy<int32_t>(data_path("params.npy"));
  ASSERT_EQ(params.shape.size(), 1u);
  ASSERT_EQ(params.shape[0], 1u);
  const int32_t n = params.data[0];

  const auto a = npy::read_npy<float>(data_path("a.npy"));
  const auto b = npy::read_npy<float>(data_path("b.npy"));
  const auto ref = npy::read_npy<float>(data_path("out.npy"));

  ASSERT_EQ((int32_t)a.shape[0], n);
  ASSERT_EQ((int32_t)b.shape[0], n);
  ASSERT_EQ((int32_t)ref.shape[0], n);

  std::vector<float> out((size_t)n, 0.0f);
  const int rc = kc_add_f32(a.data.data(), b.data.data(), out.data(), n);
  ASSERT_EQ(rc, 0);

  double max_abs = 0.0;
  for (int32_t i = 0; i < n; ++i) {
    max_abs = std::max(max_abs, std::abs((double)out[i] - (double)ref.data[i]));
  }
  EXPECT_LE(max_abs, 1e-6);
}

