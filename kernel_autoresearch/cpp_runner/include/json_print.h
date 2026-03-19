#pragma once

#include <sstream>
#include <string>

namespace kc {

inline std::string escape_json_string(const std::string& s) {
  std::ostringstream oss;
  for (const char c : s) {
    switch (c) {
      case '"':
        oss << "\\\"";
        break;
      case '\\':
        oss << "\\\\";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        oss << c;
        break;
    }
  }
  return oss.str();
}

}  // namespace kc

