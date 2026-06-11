#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace tilemem {

struct Metrics {
  std::uint64_t tokens = 0;
  std::uint64_t io_bytes = 0;
  std::uint64_t h2d_bytes = 0;
  std::uint64_t gemm_calls = 0;
  double gpu_idle_us = 0.0;
  std::uint64_t reloads = 0;
  std::uint64_t hbm_hits = 0;
  std::uint64_t hbm_misses = 0;
  std::uint64_t dram_hits = 0;
  std::uint64_t dram_misses = 0;
  double io_wait_us = 0.0;
  double h2d_wait_us = 0.0;
  double compute_us = 0.0;
  double overlap_us = 0.0;
  std::uint64_t deadline_misses = 0;
  std::uint64_t eviction_count = 0;
  std::uint64_t prefetch_hits = 0;
  std::uint64_t prefetch_waste = 0;
  std::uint64_t dram_occupancy_bytes = 0;
  std::uint64_t hbm_occupancy_bytes = 0;
  double modeled_elapsed_us = 0.0;
  double elapsed_sec = 0.0;
  double wall_elapsed_sec = 0.0;
  double tok_per_sec = 0.0;
};

void write_metrics_json(const Metrics &metrics, const std::string &path);

} // namespace tilemem
