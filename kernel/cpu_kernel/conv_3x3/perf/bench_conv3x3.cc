#include "kc_conv3x3.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

static void fill_rand(std::vector<float>& v, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
  for (float& x : v) x = dist(rng);
}

}  // namespace

int main() {
  const int32_t n = 1;
  const int32_t cin = 32;
  const int32_t cout = 64;
  const int32_t h = 112;
  const int32_t w = 112;
  const int32_t stride_h = 1;
  const int32_t stride_w = 1;
  const int32_t pad_h = 1;
  const int32_t pad_w = 1;

  const int32_t hout = (h + 2 * pad_h - 3) / stride_h + 1;
  const int32_t wout = (w + 2 * pad_w - 3) / stride_w + 1;

  std::vector<float> input(static_cast<size_t>(n) * cin * h * w);
  std::vector<float> weight(static_cast<size_t>(cout) * cin * 3 * 3);
  std::vector<float> bias(static_cast<size_t>(cout));
  std::vector<float> output(static_cast<size_t>(n) * cout * hout * wout);

  fill_rand(input, 0);
  fill_rand(weight, 1);
  fill_rand(bias, 2);

  // Warmup
  for (int i = 0; i < 3; ++i) {
    kc_conv3x3_nchw_f32(input.data(), weight.data(), bias.data(), output.data(), n, cin, h, w, cout, stride_h, stride_w,
                       pad_h, pad_w);
  }

  const int iters = 20;
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    kc_conv3x3_nchw_f32(input.data(), weight.data(), bias.data(), output.data(), n, cin, h, w, cout, stride_h, stride_w,
                       pad_h, pad_w);
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

  // FLOPs: per output element: cin * 9 mul + cin * 9 add ~= 2*cin*9
  const double flops = static_cast<double>(n) * cout * hout * wout * (2.0 * cin * 9.0);
  const double gflops = flops / (ms * 1e-3) / 1e9;

  std::printf("kc_conv3x3_nchw_f32  n=%d cin=%d cout=%d h=%d w=%d  -> hout=%d wout=%d\n", n, cin, cout, h, w, hout, wout);
  std::printf("time: %.3f ms, throughput: %.2f GFLOP/s\n", ms, gflops);
  std::printf("output[0]=%.6f\n", output[0]);
  return 0;
}

