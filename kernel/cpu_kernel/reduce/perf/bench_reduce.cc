#include "kc_reduce.h"

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
  std::vector<float> out(rows);
  fill_rand(x, 0);

  for (int i = 0; i < 5; ++i) kc_reduce_sum_lastdim_f32(x.data(), out.data(), rows, cols);

  const int iters = 50;
  auto run = [&](const char* name, int (*fn)(const float*, float*, int32_t, int32_t)) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn(x.data(), out.data(), rows, cols);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    std::printf("%s rows=%d cols=%d  time: %.3f ms  out0=%.6f\n", name, rows, cols, ms, out[0]);
  };

  run("kc_reduce_sum_lastdim_f32", kc_reduce_sum_lastdim_f32);
  run("kc_reduce_max_lastdim_f32", kc_reduce_max_lastdim_f32);
  return 0;
}

