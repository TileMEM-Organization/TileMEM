#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemem::parsing {

inline std::string trim(std::string value) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(),
              value.end());
  return value;
}

inline std::string unquote(std::string value) {
  value = trim(value);
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

inline std::size_t find_key(const std::string &line, const std::string &key) {
  auto pos = line.find("\"" + key + "\"");
  if (pos == std::string::npos) {
    throw std::runtime_error("missing JSON key: " + key);
  }
  pos = line.find(':', pos);
  if (pos == std::string::npos) {
    throw std::runtime_error("missing JSON colon for key: " + key);
  }
  return pos + 1;
}

inline std::string json_string(const std::string &line, const std::string &key) {
  auto pos = find_key(line, key);
  pos = line.find('"', pos);
  if (pos == std::string::npos) {
    throw std::runtime_error("missing JSON string value for key: " + key);
  }
  auto end = line.find('"', pos + 1);
  if (end == std::string::npos) {
    throw std::runtime_error("unterminated JSON string for key: " + key);
  }
  return line.substr(pos + 1, end - pos - 1);
}

template <typename T>
inline T json_integer(const std::string &line, const std::string &key) {
  auto pos = find_key(line, key);
  while (pos < line.size() &&
         (std::isspace(static_cast<unsigned char>(line[pos])) ||
          line[pos] == '"')) {
    ++pos;
  }
  auto end = pos;
  while (end < line.size() &&
         (std::isdigit(static_cast<unsigned char>(line[end])) ||
          line[end] == '-')) {
    ++end;
  }
  if (end == pos) {
    throw std::runtime_error("missing JSON integer for key: " + key);
  }
  std::istringstream input(line.substr(pos, end - pos));
  long long value = 0;
  input >> value;
  return static_cast<T>(value);
}

inline std::vector<int> json_int_array(const std::string &line,
                                       const std::string &key) {
  auto pos = find_key(line, key);
  pos = line.find('[', pos);
  auto end = line.find(']', pos);
  if (pos == std::string::npos || end == std::string::npos || end < pos) {
    throw std::runtime_error("missing JSON array for key: " + key);
  }
  std::vector<int> result;
  std::string body = line.substr(pos + 1, end - pos - 1);
  std::stringstream ss(body);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = trim(item);
    if (!item.empty()) {
      result.push_back(std::stoi(item));
    }
  }
  return result;
}

} // namespace tilemem::parsing
