#include "kc_softmax.h"

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
  const int32_t rows = 4096;
  const int32_t cols = 1024;
  std::vector<float> x((int64_t)rows * cols);
  std::vector<float> y((int64_t)rows * cols);
  fill_rand(x, 0);

  for (int i = 0; i < 5; ++i) kc_softmax_lastdim_f32(x.data(), y.data(), rows, cols);

  const int iters = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) kc_softmax_lastdim_f32(x.data(), y.data(), rows, cols);
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

  std::printf("kc_softmax_lastdim_f32 rows=%d cols=%d\n", rows, cols);
  std::printf("time: %.3f ms\n", ms);
  std::printf("y_mean: %.6f\n", (double)y[0]);
  return 0;
}

