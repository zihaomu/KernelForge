#pragma once

#include <cstdint>
#include <cstring>

namespace kc {

inline uint16_t float_to_fp16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const uint32_t sign = (bits >> 16) & 0x8000u;
  uint32_t mantissa = bits & 0x007FFFFFu;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;

  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mantissa = (mantissa | 0x00800000u) >> (1 - exp);
    if (mantissa & 0x00001000u) mantissa += 0x00002000u;
    return static_cast<uint16_t>(sign | (mantissa >> 13));
  }

  if (exp >= 31) {
    if (mantissa == 0) return static_cast<uint16_t>(sign | 0x7C00u);
    mantissa >>= 13;
    return static_cast<uint16_t>(sign | 0x7C00u | mantissa | (mantissa == 0));
  }

  if (mantissa & 0x00001000u) {
    mantissa += 0x00002000u;
    if (mantissa & 0x00800000u) {
      mantissa = 0;
      exp += 1;
      if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    }
  }

  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mantissa >> 13));
}

inline float fp16_to_float(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mantissa = h & 0x03FFu;

  uint32_t bits = 0;
  if (exp == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int32_t e = -1;
      do {
        e++;
        mantissa <<= 1;
      } while ((mantissa & 0x0400u) == 0);
      mantissa &= 0x03FFu;
      bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (mantissa << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (mantissa << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
  }

  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

}  // namespace kc
