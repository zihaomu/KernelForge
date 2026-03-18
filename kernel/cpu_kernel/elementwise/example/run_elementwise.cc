#include "kc_elementwise.h"

#include <cstdint>
#include <cstdio>

int main() {
  const int32_t n = 8;
  float a[n] = {0, 1, 2, 3, 4, 5, 6, 7};
  float b[n] = {10, 10, 10, 10, 10, 10, 10, 10};
  float out[n] = {};

  const int rc = kc_add_f32(a, b, out, n);
  std::printf("kc_add_f32 rc=%d\n", rc);
  for (int i = 0; i < n; ++i) std::printf("%.1f ", out[i]);
  std::printf("\n");
  return 0;
}

