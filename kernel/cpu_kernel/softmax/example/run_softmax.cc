#include "kc_softmax.h"

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
  const int32_t rows = 2;
  const int32_t cols = 4;
  const float x[rows * cols] = {
      1.0f, 2.0f, 3.0f, 4.0f,
      4.0f, 3.0f, 2.0f, 1.0f,
  };
  std::vector<float> y(rows * cols, 0.0f);
  const int rc = kc_softmax_lastdim_f32(x, y.data(), rows, cols);
  std::printf("kc_softmax_lastdim_f32 rc=%d\n", rc);
  for (int32_t r = 0; r < rows; ++r) {
    for (int32_t c = 0; c < cols; ++c) {
      std::printf("%0.6f ", y[r * cols + c]);
    }
    std::printf("\n");
  }
  return 0;
}

