#pragma once

#include <chrono>

namespace kc {

class Timer {
 public:
  void tic() { t0_ = std::chrono::high_resolution_clock::now(); }
  double toc_ms() const {
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0_).count();
  }

 private:
  std::chrono::high_resolution_clock::time_point t0_;
};

}  // namespace kc

