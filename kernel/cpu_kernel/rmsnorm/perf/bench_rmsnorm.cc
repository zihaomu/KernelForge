#include "kc_rmsnorm.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

static void fill_rand(std::vector<float>& v, uint32_t seed, float scale) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  for (float& x : v) x = dist(rng);
}

}  // namespace

int main() {
  const int32_t rows = 4096;
  const int32_t cols = 1024;
  const float eps = 1e-5f;

  std::vector<float> x((int64_t)rows * cols);
  std::vector<float> w(cols);
  std::vector<float> y((int64_t)rows * cols);
  fill_rand(x, 0, 3.0f);
  fill_rand(w, 1, 0.1f);
  for (float& t : w) t += 1.0f;

  for (int i = 0; i < 5; ++i) kc_rmsnorm_lastdim_f32(x.data(), w.data(), y.data(), rows, cols, eps);

  const int iters = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) kc_rmsnorm_lastdim_f32(x.data(), w.data(), y.data(), rows, cols, eps);
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

  std::printf("kc_rmsnorm_lastdim_f32 rows=%d cols=%d time: %.3f ms y0=%.6f\n", rows, cols, ms, y[0]);
  return 0;
}

