#include "kc_reduce.h"

#include <cstdint>
#include <cstdio>

int main() {
  const int32_t rows = 2;
  const int32_t cols = 4;
  const float x[rows * cols] = {
      1, 2, 3, 4,
      -1, -2, -3, -4,
  };
  float out_sum[rows] = {};
  float out_max[rows] = {};
  const int rc1 = kc_reduce_sum_lastdim_f32(x, out_sum, rows, cols);
  const int rc2 = kc_reduce_max_lastdim_f32(x, out_max, rows, cols);
  std::printf("sum rc=%d: %.1f %.1f\n", rc1, out_sum[0], out_sum[1]);
  std::printf("max rc=%d: %.1f %.1f\n", rc2, out_max[0], out_max[1]);
  return 0;
}

