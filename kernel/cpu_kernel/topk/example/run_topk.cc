#include "kc_topk.h"

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
  const int32_t rows = 1;
  const int32_t cols = 8;
  const int32_t k = 3;
  const int32_t descend = 1;
  const float x[cols] = {1.0f, 9.0f, 2.0f, -1.0f, 3.0f, 7.0f, 0.0f, 4.0f};
  std::vector<float> vals(rows * k, 0.0f);
  std::vector<int32_t> idx(rows * k, 0);
  const int rc = kc_topk_lastdim_f32(x, vals.data(), idx.data(), rows, cols, k, descend);
  std::printf("kc_topk rc=%d\n", rc);
  for (int i = 0; i < k; ++i) std::printf("(%d, %.1f) ", idx[i], vals[i]);
  std::printf("\n");
  return 0;
}

