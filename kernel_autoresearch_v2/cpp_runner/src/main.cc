#include "gemm_kernels.h"
#include "json_print.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using ArgMap = std::unordered_map<std::string, std::string>;

ArgMap parse_args(int argc, char** argv) {
  ArgMap args;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key(argv[i]);
    const std::string val(argv[i + 1]);
    args[key] = val;
  }
  return args;
}

int get_i(const ArgMap& args, const std::string& key, int def) {
  const auto it = args.find(key);
  if (it == args.end()) return def;
  return std::stoi(it->second);
}

bool get_b(const ArgMap& args, const std::string& key, bool def) {
  const auto it = args.find(key);
  if (it == args.end()) return def;
  return std::stoi(it->second) != 0;
}

std::string get_s(const ArgMap& args, const std::string& key, const std::string& def) {
  const auto it = args.find(key);
  if (it == args.end()) return def;
  return it->second;
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

float deterministic_value(uint64_t idx, uint32_t salt) {
  const uint64_t x = (idx * 1315423911ULL + static_cast<uint64_t>(salt) * 2654435761ULL) & 0xFFFFFFFFULL;
  const int v = static_cast<int>(x % 1024ULL) - 512;
  return static_cast<float>(v) / 128.0f;
}

void fill_deterministic(std::vector<float>& vec, uint32_t salt) {
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = deterministic_value(static_cast<uint64_t>(i), salt);
  }
}

void fill_rand(std::vector<float>& vec, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& x : vec) x = dist(rng);
}

void run_variant(const std::string& variant, const float* a, const float* b, float* c, kc::GemmConfig cfg) {
  if (variant == "naive") {
    cfg.pack_a = false;
    cfg.pack_b = false;
    cfg.simd = false;
    kc::gemm_naive_f32(a, b, c, cfg);
    return;
  }
  if (variant == "blocked") {
    cfg.pack_a = false;
    cfg.pack_b = false;
    kc::gemm_blocked_f32(a, b, c, cfg);
    return;
  }
  if (variant == "blocked_pack") {
    cfg.pack_a = true;
    cfg.pack_b = true;
    kc::gemm_blocked_pack_f32(a, b, c, cfg);
    return;
  }
  if (variant == "blocked_pack_simd") {
    cfg.pack_a = true;
    cfg.pack_b = true;
    cfg.simd = true;
    kc::gemm_blocked_pack_f32(a, b, c, cfg);
    return;
  }
  // Fallback to naive for unknown variants.
  cfg.pack_a = false;
  cfg.pack_b = false;
  cfg.simd = false;
  kc::gemm_naive_f32(a, b, c, cfg);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parse_args(argc, argv);
    kc::GemmConfig cfg;
    cfg.m = get_i(args, "--m", 256);
    cfg.n = get_i(args, "--n", 256);
    cfg.k = get_i(args, "--k", 256);
    cfg.bm = get_i(args, "--bm", 64);
    cfg.bn = get_i(args, "--bn", 64);
    cfg.bk = get_i(args, "--bk", 64);
    cfg.pack_a = get_b(args, "--pack_a", false);
    cfg.pack_b = get_b(args, "--pack_b", false);
    cfg.simd = get_b(args, "--simd", false);
    cfg.threads = get_i(args, "--threads", 1);
    cfg.unroll_k = std::max(1, get_i(args, "--unroll_k", 1));
    const int warmup = std::max(0, get_i(args, "--warmup", 2));
    const int iters = std::max(1, get_i(args, "--iters", 6));
    const bool verify = get_b(args, "--verify", true);
    const std::string variant = get_s(args, "--kernel_variant", "naive");
    const std::string input_mode = get_s(args, "--input_mode", "random");

    const size_t a_size = static_cast<size_t>(cfg.m) * cfg.k;
    const size_t b_size = static_cast<size_t>(cfg.k) * cfg.n;
    const size_t c_size = static_cast<size_t>(cfg.m) * cfg.n;
    std::vector<float> a(a_size), b(b_size), c(c_size), cref(c_size);
    if (input_mode == "deterministic") {
      fill_deterministic(a, 123);
      fill_deterministic(b, 321);
    } else {
      fill_rand(a, 123);
      fill_rand(b, 321);
    }

    if (verify) {
      kc::gemm_reference_f32(a.data(), b.data(), cref.data(), cfg);
    }

    for (int i = 0; i < warmup; ++i) {
      run_variant(variant, a.data(), b.data(), c.data(), cfg);
    }

    std::vector<double> lats;
    lats.reserve(static_cast<size_t>(iters));
    kc::Timer timer;
    for (int i = 0; i < iters; ++i) {
      timer.tic();
      run_variant(variant, a.data(), b.data(), c.data(), cfg);
      lats.push_back(timer.toc_ms());
    }

    const double p50 = percentile(lats, 0.50);
    const double p95 = percentile(lats, 0.95);
    const double flops = 2.0 * static_cast<double>(cfg.m) * cfg.n * cfg.k;
    const double gflops = flops / (std::max(p50, 1e-9) * 1e6);

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    bool valid = true;
    if (verify) {
      for (size_t i = 0; i < c_size; ++i) {
        const double diff = std::abs(static_cast<double>(c[i]) - cref[i]);
        max_abs_err = std::max(max_abs_err, diff);
        const double denom = std::max(std::abs(static_cast<double>(cref[i])), 1e-9);
        max_rel_err = std::max(max_rel_err, diff / denom);
      }
      valid = max_abs_err <= 2e-3 || max_rel_err <= 2e-3;
    }

    double output_sum = 0.0;
    double output_l2 = 0.0;
    for (size_t i = 0; i < c_size; ++i) {
      const double v = static_cast<double>(c[i]);
      output_sum += v;
      output_l2 += v * v;
    }

    std::ostringstream oss;
    oss << "{";
    oss << "\"valid\":" << (valid ? "true" : "false") << ",";
    oss << "\"kernel_variant\":\"" << kc::escape_json_string(variant) << "\",";
    oss << "\"m\":" << cfg.m << ",";
    oss << "\"n\":" << cfg.n << ",";
    oss << "\"k\":" << cfg.k << ",";
    oss << "\"bm\":" << cfg.bm << ",";
    oss << "\"bn\":" << cfg.bn << ",";
    oss << "\"bk\":" << cfg.bk << ",";
    oss << "\"pack_a\":" << (cfg.pack_a ? "true" : "false") << ",";
    oss << "\"pack_b\":" << (cfg.pack_b ? "true" : "false") << ",";
    oss << "\"simd\":" << (cfg.simd ? "true" : "false") << ",";
    oss << "\"threads\":" << cfg.threads << ",";
    oss << "\"unroll_k\":" << cfg.unroll_k << ",";
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
    std::cout << "{\"valid\":false,\"error\":\"" << kc::escape_json_string(ex.what()) << "\"}\n";
    return 1;
  }
}

