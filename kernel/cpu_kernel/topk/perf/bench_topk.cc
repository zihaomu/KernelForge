#include "kc_topk.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

static void fill_rand(std::vector<float>& v, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  for (float& x : v) x = dist(rng);
}

}  // namespace

int main() {
  const int32_t rows = 2048;
  const int32_t cols = 4096;
  const int32_t k = 8;
  const int32_t descend = 1;

  std::vector<float> x((int64_t)rows * cols);
  std::vector<float> vals((int64_t)rows * k);
  std::vector<int32_t> idx((int64_t)rows * k);
  fill_rand(x, 0);

  for (int i = 0; i < 2; ++i) kc_topk_lastdim_f32(x.data(), vals.data(), idx.data(), rows, cols, k, descend);

  const int iters = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) kc_topk_lastdim_f32(x.data(), vals.data(), idx.data(), rows, cols, k, descend);
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

  std::printf("kc_topk_lastdim_f32 rows=%d cols=%d k=%d time: %.3f ms  vals0=%.6f idx0=%d\n",
              rows, cols, k, ms, vals[0], idx[0]);
  return 0;
}

