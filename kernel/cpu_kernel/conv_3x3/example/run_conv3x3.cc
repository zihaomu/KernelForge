#include "kc_conv3x3.h"

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
  const int32_t n = 1;
  const int32_t cin = 1;
  const int32_t cout = 1;
  const int32_t h = 5;
  const int32_t w = 5;
  const int32_t stride_h = 1;
  const int32_t stride_w = 1;
  const int32_t pad_h = 1;
  const int32_t pad_w = 1;

  const int32_t hout = (h + 2 * pad_h - 3) / stride_h + 1;
  const int32_t wout = (w + 2 * pad_w - 3) / stride_w + 1;

  std::vector<float> input(static_cast<size_t>(n) * cin * h * w, 0.0f);
  std::vector<float> weight(static_cast<size_t>(cout) * cin * 3 * 3, 0.0f);
  std::vector<float> bias(static_cast<size_t>(cout), 0.0f);
  std::vector<float> output(static_cast<size_t>(n) * cout * hout * wout, 0.0f);

  // Simple impulse-like input
  input[2 * w + 2] = 1.0f;
  // All-ones kernel
  for (float& x : weight) x = 1.0f;

  const int rc = kc_conv3x3_nchw_f32(
      input.data(),
      weight.data(),
      bias.data(),
      output.data(),
      n,
      cin,
      h,
      w,
      cout,
      stride_h,
      stride_w,
      pad_h,
      pad_w);

  std::printf("kc_conv3x3_nchw_f32 rc=%d\n", rc);
  for (int32_t yo = 0; yo < hout; ++yo) {
    for (int32_t xo = 0; xo < wout; ++xo) {
      const float v = output[yo * wout + xo];
      std::printf("%6.2f ", v);
    }
    std::printf("\n");
  }
  return 0;
}

