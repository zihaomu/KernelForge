#pragma once

#include <string>

namespace kc {

inline std::string escape_json_string(const std::string& src) {
  std::string out;
  out.reserve(src.size() + 16);
  for (char ch : src) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(ch);
        break;
    }
  }
  return out;
}

}  // namespace kc

