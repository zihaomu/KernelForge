#include "kc_elementwise.h"

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
  const int32_t n = 1 << 24;  // ~16M
  std::vector<float> a((size_t)n), b((size_t)n), out((size_t)n);
  fill_rand(a, 0);
  fill_rand(b, 1);

  for (int i = 0; i < 10; ++i) kc_add_f32(a.data(), b.data(), out.data(), n);

  const int iters = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) kc_add_f32(a.data(), b.data(), out.data(), n);
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

  const double bytes = (double)n * sizeof(float) * 3.0;  // load a,b and store out
  const double gbps = bytes / (ms * 1e-3) / 1e9;

  std::printf("kc_add_f32 n=%d\n", n);
  std::printf("time: %.3f ms, bandwidth: %.2f GB/s\n", ms, gbps);
  std::printf("out[0]=%.6f\n", out[0]);
  return 0;
}

