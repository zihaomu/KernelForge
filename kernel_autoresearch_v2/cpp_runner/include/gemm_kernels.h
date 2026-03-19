#pragma once

#include <cstdint>

namespace kc {

struct GemmConfig {
  int32_t m = 0;
  int32_t n = 0;
  int32_t k = 0;
  int32_t bm = 64;
  int32_t bn = 64;
  int32_t bk = 64;
  bool pack_a = false;
  bool pack_b = false;
  bool simd = false;
  int32_t threads = 1;
  int32_t unroll_k = 1;
};

void gemm_reference_f32(const float* a, const float* b, float* c, const GemmConfig& cfg);
void gemm_naive_f32(const float* a, const float* b, float* c, const GemmConfig& cfg);
void gemm_blocked_f32(const float* a, const float* b, float* c, const GemmConfig& cfg);
void gemm_blocked_pack_f32(const float* a, const float* b, float* c, const GemmConfig& cfg);

}  // namespace kc

