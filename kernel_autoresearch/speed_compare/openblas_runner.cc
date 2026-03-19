#include <cblas.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using ArgMap = std::unordered_map<std::string, std::string>;

ArgMap parse_args(int argc, char** argv) {
  ArgMap args;
  for (int i = 1; i + 1 < argc; i += 2) {
    args[argv[i]] = argv[i + 1];
  }
  return args;
}

int get_i(const ArgMap& args, const std::string& key, int def) {
  const auto it = args.find(key);
  if (it == args.end()) return def;
  return std::stoi(it->second);
}

std::string get_s(const ArgMap& args, const std::string& key, const std::string& def) {
  const auto it = args.find(key);
  if (it == args.end()) return def;
  return it->second;
}

std::string normalize_dtype(std::string v) {
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (v == "fp16" || v == "half") return "f16";
  if (v == "int8") return "i8";
  if (v == "float32" || v == "fp32") return "f32";
  return v;
}

double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = q * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = static_cast<size_t>(std::ceil(idx));
  const double t = idx - static_cast<double>(lo);
  return v[lo] * (1.0 - t) + v[hi] * t;
}

uint16_t float_to_fp16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  uint32_t mantissa = bits & 0x007FFFFFu;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;

  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mantissa = (mantissa | 0x00800000u) >> static_cast<uint32_t>(1 - exp);
    return static_cast<uint16_t>(sign | ((mantissa + 0x00001000u) >> 13));
  }
  if (exp >= 31) {
    if (mantissa == 0) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | 0x7C00u | (mantissa >> 13));
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mantissa >> 13));
}

float fp16_to_float(uint16_t h) {
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
        ++e;
        mantissa <<= 1;
      } while ((mantissa & 0x0400u) == 0u);
      mantissa &= 0x03FFu;
      bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (mantissa << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mantissa << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (mantissa << 13);
  }

  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

float deterministic_value_f32(uint64_t idx, uint32_t salt) {
  const uint64_t x = (idx * 1315423911ULL + static_cast<uint64_t>(salt) * 2654435761ULL) & 0xFFFFFFFFULL;
  const int v = static_cast<int>(x % 1024ULL) - 512;
  return static_cast<float>(v) / 128.0f;
}

int8_t deterministic_value_i8(uint64_t idx, uint32_t salt) {
  const uint64_t x = (idx * 1315423911ULL + static_cast<uint64_t>(salt) * 2654435761ULL) & 0xFFFFFFFFULL;
  const int v = static_cast<int>(x % 255ULL) - 127;
  return static_cast<int8_t>(v);
}

uint16_t deterministic_value_f16(uint64_t idx, uint32_t salt) {
  return float_to_fp16(deterministic_value_f32(idx, salt));
}

void fill_deterministic_f32(std::vector<float>& vec, uint32_t salt) {
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = deterministic_value_f32(static_cast<uint64_t>(i), salt);
  }
}

void fill_deterministic_i8(std::vector<int8_t>& vec, uint32_t salt) {
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = deterministic_value_i8(static_cast<uint64_t>(i), salt);
  }
}

void fill_deterministic_f16(std::vector<uint16_t>& vec, uint32_t salt) {
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = deterministic_value_f16(static_cast<uint64_t>(i), salt);
  }
}

void fill_random_f32(std::vector<float>& vec, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& x : vec) x = dist(rng);
}

void fill_random_i8(std::vector<int8_t>& vec, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-127, 127);
  for (int8_t& x : vec) x = static_cast<int8_t>(dist(rng));
}

void fill_random_f16(std::vector<uint16_t>& vec, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (uint16_t& x : vec) x = float_to_fp16(dist(rng));
}

void gemm_reference_f32(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int m, int n, int k) {
  std::fill(c.begin(), c.end(), 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int p = 0; p < k; ++p) {
      const float av = a[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(p)];
      for (int j = 0; j < n; ++j) {
        c[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] +=
            av * b[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(j)];
      }
    }
  }
}

void gemm_reference_i8_i32(const std::vector<int8_t>& a, const std::vector<int8_t>& b, std::vector<int32_t>& c, int m, int n, int k) {
  std::fill(c.begin(), c.end(), int32_t{0});
  for (int i = 0; i < m; ++i) {
    for (int p = 0; p < k; ++p) {
      const int32_t av = static_cast<int32_t>(a[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(p)]);
      for (int j = 0; j < n; ++j) {
        c[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] +=
            av * static_cast<int32_t>(b[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(j)]);
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parse_args(argc, argv);
    const int m = get_i(args, "--m", 256);
    const int n = get_i(args, "--n", 256);
    const int k = get_i(args, "--k", 256);
    const int warmup = std::max(0, get_i(args, "--warmup", 2));
    const int iters = std::max(1, get_i(args, "--iters", 6));
    const int threads = std::max(1, get_i(args, "--threads", 1));
    const std::string input_mode = get_s(args, "--input_mode", "random");
    const std::string input_dtype = normalize_dtype(get_s(args, "--input_dtype", "f32"));

    std::string baseline_mode = "native_f32";
    if (input_dtype == "f16") {
      baseline_mode = "proxy_f32_from_f16";
    } else if (input_dtype == "i8") {
      baseline_mode = "proxy_f32_from_i8";
    } else if (input_dtype != "f32") {
      throw std::runtime_error("unsupported input_dtype for openblas runner: " + input_dtype);
    }

    openblas_set_num_threads(threads);

    const size_t a_size = static_cast<size_t>(m) * static_cast<size_t>(k);
    const size_t b_size = static_cast<size_t>(k) * static_cast<size_t>(n);
    const size_t c_size = static_cast<size_t>(m) * static_cast<size_t>(n);

    std::vector<float> a_f32(a_size), b_f32(b_size), c_f32(c_size), cref_f32(c_size);
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;

    if (input_dtype == "f32") {
      if (input_mode == "deterministic") {
        fill_deterministic_f32(a_f32, 123);
        fill_deterministic_f32(b_f32, 321);
      } else {
        fill_random_f32(a_f32, 123);
        fill_random_f32(b_f32, 321);
      }
      gemm_reference_f32(a_f32, b_f32, cref_f32, m, n, k);
    } else if (input_dtype == "f16") {
      std::vector<uint16_t> a_f16(a_size), b_f16(b_size);
      if (input_mode == "deterministic") {
        fill_deterministic_f16(a_f16, 123);
        fill_deterministic_f16(b_f16, 321);
      } else {
        fill_random_f16(a_f16, 123);
        fill_random_f16(b_f16, 321);
      }
      for (size_t i = 0; i < a_size; ++i) a_f32[i] = fp16_to_float(a_f16[i]);
      for (size_t i = 0; i < b_size; ++i) b_f32[i] = fp16_to_float(b_f16[i]);
      gemm_reference_f32(a_f32, b_f32, cref_f32, m, n, k);
    } else if (input_dtype == "i8") {
      std::vector<int8_t> a_i8(a_size), b_i8(b_size);
      if (input_mode == "deterministic") {
        fill_deterministic_i8(a_i8, 123);
        fill_deterministic_i8(b_i8, 321);
      } else {
        fill_random_i8(a_i8, 123);
        fill_random_i8(b_i8, 321);
      }
      for (size_t i = 0; i < a_size; ++i) a_f32[i] = static_cast<float>(a_i8[i]);
      for (size_t i = 0; i < b_size; ++i) b_f32[i] = static_cast<float>(b_i8[i]);
      std::vector<int32_t> cref_i32(c_size);
      gemm_reference_i8_i32(a_i8, b_i8, cref_i32, m, n, k);
      for (size_t i = 0; i < c_size; ++i) cref_f32[i] = static_cast<float>(cref_i32[i]);
    }

    for (int i = 0; i < warmup; ++i) {
      std::fill(c_f32.begin(), c_f32.end(), 0.0f);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a_f32.data(), k, b_f32.data(), n, 0.0f, c_f32.data(), n);
    }

    std::vector<double> lats;
    lats.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
      std::fill(c_f32.begin(), c_f32.end(), 0.0f);
      const auto t0 = std::chrono::high_resolution_clock::now();
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a_f32.data(), k, b_f32.data(), n, 0.0f, c_f32.data(), n);
      const auto t1 = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double, std::milli> dt = t1 - t0;
      lats.push_back(dt.count());
    }

    for (size_t i = 0; i < c_size; ++i) {
      const double diff = std::abs(static_cast<double>(c_f32[i]) - static_cast<double>(cref_f32[i]));
      max_abs_err = std::max(max_abs_err, diff);
      const double denom = std::max(std::abs(static_cast<double>(cref_f32[i])), 1e-9);
      max_rel_err = std::max(max_rel_err, diff / denom);
    }

    bool valid = false;
    if (input_dtype == "i8") {
      valid = (max_abs_err <= 8.0) || (max_rel_err <= 1e-3);
    } else {
      valid = (max_abs_err <= 2e-3) || (max_rel_err <= 2e-3);
    }

    double output_sum = 0.0;
    double output_l2 = 0.0;
    for (float v : c_f32) {
      const double dv = static_cast<double>(v);
      output_sum += dv;
      output_l2 += dv * dv;
    }

    const double p50 = percentile(lats, 0.50);
    const double p95 = percentile(lats, 0.95);
    const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const double gflops = flops / (std::max(p50, 1e-9) * 1e6);

    std::ostringstream oss;
    oss << "{";
    oss << "\"valid\":" << (valid ? "true" : "false") << ",";
    oss << "\"engine\":\"openblas\",";
    oss << "\"baseline_mode\":\"" << baseline_mode << "\",";
    oss << "\"input_dtype\":\"" << input_dtype << "\",";
    oss << "\"output_dtype\":\"f32\",";
    oss << "\"m\":" << m << ",";
    oss << "\"n\":" << n << ",";
    oss << "\"k\":" << k << ",";
    oss << "\"threads\":" << threads << ",";
    oss << "\"latency_ms_p50\":" << p50 << ",";
    oss << "\"latency_ms_p95\":" << p95 << ",";
    oss << "\"gflops\":" << gflops << ",";
    oss << "\"max_abs_err\":" << max_abs_err << ",";
    oss << "\"max_rel_err\":" << max_rel_err << ",";
    oss << "\"output_sum\":" << output_sum << ",";
    oss << "\"output_l2\":" << output_l2;
    oss << "}";
    std::cout << oss.str() << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cout << "{\"valid\":false,\"error\":\"" << ex.what() << "\"}\n";
    return 1;
  }
}
