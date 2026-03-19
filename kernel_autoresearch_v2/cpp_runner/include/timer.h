#pragma once

#include <chrono>

namespace kc {

class Timer {
 public:
  void tic() { start_ = std::chrono::high_resolution_clock::now(); }

  double toc_ms() const {
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> ms = end - start_;
    return ms.count();
  }

 private:
  std::chrono::high_resolution_clock::time_point start_{};
};

}  // namespace kc

