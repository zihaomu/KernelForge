#include "kc_topk.h"

#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "npy.hpp"

namespace {

static std::string data_path(const std::string& name) {
  return std::string("kernel/cpu_kernel/topk/test/data/") + name;
}

}  // namespace

TEST(kc_topk, accuracy_lastdim_f32) {
  const auto params = npy::read_npy<int32_t>(data_path("params.npy"));
  ASSERT_EQ(params.shape.size(), 1u);
  ASSERT_EQ(params.shape[0], 4u);
  const int32_t rows = params.data[0];
  const int32_t cols = params.data[1];
  const int32_t k = params.data[2];
  const int32_t descend = params.data[3];

  const auto x = npy::read_npy<float>(data_path("input.npy"));
  const auto ref_vals = npy::read_npy<float>(data_path("values.npy"));
  const auto ref_idx = npy::read_npy<int32_t>(data_path("indices.npy"));
  ASSERT_EQ((int32_t)x.shape[0], rows);
  ASSERT_EQ((int32_t)x.shape[1], cols);
  ASSERT_EQ((int32_t)ref_vals.shape[0], rows);
  ASSERT_EQ((int32_t)ref_vals.shape[1], k);
  ASSERT_EQ((int32_t)ref_idx.shape[0], rows);
  ASSERT_EQ((int32_t)ref_idx.shape[1], k);

  std::vector<float> out_vals((int64_t)rows * k, 0.0f);
  std::vector<int32_t> out_idx((int64_t)rows * k, 0);
  ASSERT_EQ(kc_topk_lastdim_f32(x.data.data(), out_vals.data(), out_idx.data(), rows, cols, k, descend), 0);

  for (int32_t r = 0; r < rows; ++r) {
    for (int32_t t = 0; t < k; ++t) {
      const int64_t off = (int64_t)r * k + t;
      // We enforce same order as numpy argsort-based reference in gen_data.py.
      EXPECT_EQ(out_idx[off], ref_idx.data[off]);
      EXPECT_EQ(out_vals[off], ref_vals.data[off]);
    }
  }
}

