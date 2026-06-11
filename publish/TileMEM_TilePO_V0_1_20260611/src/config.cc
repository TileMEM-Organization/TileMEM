#include "tilemem/config.h"

#include <fstream>
#include <stdexcept>
#include <unordered_map>

#include "parsing.h"

namespace tilemem {
namespace {

using SectionMap = std::unordered_map<std::string, std::string>;

std::string get_required(const std::unordered_map<std::string, SectionMap> &data,
                         const std::string &section,
                         const std::string &key) {
  auto section_it = data.find(section);
  if (section_it == data.end()) {
    throw std::runtime_error("missing TOML section [" + section + "]");
  }
  auto key_it = section_it->second.find(key);
  if (key_it == section_it->second.end()) {
    throw std::runtime_error("missing TOML key: " + section + "." + key);
  }
  return key_it->second;
}

std::string get_optional(const std::unordered_map<std::string, SectionMap> &data,
                         const std::string &section, const std::string &key,
                         const std::string &fallback) {
  auto section_it = data.find(section);
  if (section_it == data.end()) {
    return fallback;
  }
  auto key_it = section_it->second.find(key);
  if (key_it == section_it->second.end()) {
    return fallback;
  }
  return key_it->second;
}

std::size_t parse_size(const std::string &value) {
  return static_cast<std::size_t>(std::stoull(value));
}

int parse_int(const std::string &value) { return std::stoi(value); }

} // namespace

ReplayConfig ReplayConfig::from_file(const std::string &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open config: " + path);
  }

  std::unordered_map<std::string, SectionMap> data;
  std::string section;
  std::string line;
  while (std::getline(input, line)) {
    auto comment = line.find('#');
    if (comment != std::string::npos) {
      line = line.substr(0, comment);
    }
    line = parsing::trim(line);
    if (line.empty()) {
      continue;
    }
    if (line.front() == '[' && line.back() == ']') {
      section = line.substr(1, line.size() - 2);
      continue;
    }
    auto eq = line.find('=');
    if (eq == std::string::npos || section.empty()) {
      throw std::runtime_error("invalid TOML line: " + line);
    }
    auto key = parsing::trim(line.substr(0, eq));
    auto value = parsing::unquote(line.substr(eq + 1));
    data[section][key] = value;
  }

  ReplayConfig config;
  config.hbm_budget_bytes =
      parse_size(get_required(data, "hardware", "hbm_budget_bytes"));
  config.dram_budget_bytes =
      parse_size(get_required(data, "hardware", "dram_budget_bytes"));
  config.tile_bytes =
      parse_size(get_optional(data, "tiles", "tile_bytes", "1024"));
  config.lookahead =
      parse_int(get_optional(data, "scheduler", "lookahead", "1"));
  config.prefetch_depth =
      parse_int(get_optional(data, "scheduler", "prefetch_depth", "4"));
  config.hbm_high_watermark =
      std::stod(get_optional(data, "scheduler", "hbm_high_watermark", "0.93"));
  config.dram_high_watermark =
      std::stod(get_optional(data, "scheduler", "dram_high_watermark", "0.90"));
  config.io_queue_depth =
      parse_int(get_optional(data, "io", "queue_depth", "8"));
  config.default_m = parse_int(get_optional(data, "cuda", "m", "16"));
  config.default_k = parse_int(get_optional(data, "cuda", "k", "64"));
  config.default_n = parse_int(get_optional(data, "cuda", "n", "32"));
  config.policy =
      parse_policy(get_optional(data, "scheduler", "policy", "tilemem"));
  config.expert_file = get_required(data, "paths", "expert_file");
  config.manifest_path = get_required(data, "paths", "manifest");
  config.trace_path = get_required(data, "paths", "trace");
  config.metrics_dir = get_required(data, "paths", "metrics_dir");
  return config;
}

} // namespace tilemem
