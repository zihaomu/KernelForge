#include "kc_rmsnorm.h"

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
  const int32_t rows = 1;
  const int32_t cols = 4;
  const float eps = 1e-5f;
  const float x[cols] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float w[cols] = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> y(rows * cols, 0.0f);

  const int rc = kc_rmsnorm_lastdim_f32(x, w, y.data(), rows, cols, eps);
  std::printf("kc_rmsnorm rc=%d\n", rc);
  for (int i = 0; i < cols; ++i) std::printf("%0.6f ", y[i]);
  std::printf("\n");
  return 0;
}

