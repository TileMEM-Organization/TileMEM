#pragma once

#include <cstddef>
#include <string>

#include "tilemem/types.h"

namespace tilemem {

struct ReplayConfig {
  std::size_t hbm_budget_bytes = 0;
  std::size_t dram_budget_bytes = 0;
  std::size_t tile_bytes = 0;
  int lookahead = 0;
  int prefetch_depth = 4;
  int io_queue_depth = 8;
  int default_m = 16;
  int default_k = 64;
  int default_n = 32;
  double hbm_high_watermark = 0.93;
  double dram_high_watermark = 0.90;
  Policy policy = Policy::Tilemem;
  std::string expert_file;
  std::string manifest_path;
  std::string trace_path;
  std::string metrics_dir;

  static ReplayConfig from_file(const std::string &path);
};

} // namespace tilemem
