#include "tilemem/metrics.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace tilemem {

void write_metrics_json(const Metrics &m, const std::string &path) {
  auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open metrics output: " + path);
  }
  out << "{\n";
  out << "  \"tokens\": " << m.tokens << ",\n";
  out << "  \"io_bytes\": " << m.io_bytes << ",\n";
  out << "  \"h2d_bytes\": " << m.h2d_bytes << ",\n";
  out << "  \"gemm_calls\": " << m.gemm_calls << ",\n";
  out << "  \"gpu_idle_us\": " << m.gpu_idle_us << ",\n";
  out << "  \"reloads\": " << m.reloads << ",\n";
  out << "  \"hbm_hits\": " << m.hbm_hits << ",\n";
  out << "  \"hbm_misses\": " << m.hbm_misses << ",\n";
  out << "  \"dram_hits\": " << m.dram_hits << ",\n";
  out << "  \"dram_misses\": " << m.dram_misses << ",\n";
  out << "  \"io_wait_us\": " << m.io_wait_us << ",\n";
  out << "  \"h2d_wait_us\": " << m.h2d_wait_us << ",\n";
  out << "  \"compute_us\": " << m.compute_us << ",\n";
  out << "  \"overlap_us\": " << m.overlap_us << ",\n";
  out << "  \"deadline_misses\": " << m.deadline_misses << ",\n";
  out << "  \"eviction_count\": " << m.eviction_count << ",\n";
  out << "  \"prefetch_hits\": " << m.prefetch_hits << ",\n";
  out << "  \"prefetch_waste\": " << m.prefetch_waste << ",\n";
  out << "  \"dram_occupancy_bytes\": " << m.dram_occupancy_bytes << ",\n";
  out << "  \"hbm_occupancy_bytes\": " << m.hbm_occupancy_bytes << ",\n";
  out << "  \"modeled_elapsed_us\": " << m.modeled_elapsed_us << ",\n";
  out << "  \"elapsed_sec\": " << m.elapsed_sec << ",\n";
  out << "  \"wall_elapsed_sec\": " << m.wall_elapsed_sec << ",\n";
  out << "  \"tok_per_sec\": " << m.tok_per_sec << "\n";
  out << "}\n";
}

} // namespace tilemem
