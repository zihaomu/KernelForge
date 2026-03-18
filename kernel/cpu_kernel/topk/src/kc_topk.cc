#include "kc_topk.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

struct Pair {
  float v;
  int32_t i;
};

inline bool better(float a, float b, int32_t descend) {
  return descend ? (a > b) : (a < b);
}

}  // namespace

int kc_topk_lastdim_f32(const float* input,
                        float* out_values,
                        int32_t* out_indices,
                        int32_t rows,
                        int32_t cols,
                        int32_t k,
                        int32_t descend) {
  if (!input || !out_values || !out_indices) return 1;
  if (rows <= 0 || cols <= 0) return 2;
  if (k <= 0 || k > cols) return 3;
  descend = descend ? 1 : 0;

  std::vector<Pair> buf((size_t)k);
  const float worst_init = descend ? -std::numeric_limits<float>::infinity()
                                   : +std::numeric_limits<float>::infinity();

  for (int32_t r = 0; r < rows; ++r) {
    const float* x = input + (int64_t)r * cols;
    // init
    for (int32_t t = 0; t < k; ++t) {
      buf[(size_t)t] = Pair{worst_init, -1};
    }

    // insertion into topk buffer
    for (int32_t c = 0; c < cols; ++c) {
      const float v = x[c];
      // quick reject: compare with current worst element (last after sorting)
      // We maintain buf sorted best->worst.
      if (!better(v, buf[(size_t)k - 1].v, descend)) continue;

      // find insertion pos
      int32_t pos = k - 1;
      while (pos > 0 && better(v, buf[(size_t)pos - 1].v, descend)) {
        buf[(size_t)pos] = buf[(size_t)pos - 1];
        --pos;
      }
      buf[(size_t)pos] = Pair{v, c};
    }

    // write
    float* ov = out_values + (int64_t)r * k;
    int32_t* oi = out_indices + (int64_t)r * k;
    for (int32_t t = 0; t < k; ++t) {
      ov[t] = buf[(size_t)t].v;
      oi[t] = buf[(size_t)t].i;
    }
  }

  return 0;
}

