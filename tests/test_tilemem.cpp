#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tilemem/config.h"
#include "tilemem/fp4_gemm.h"
#include "tilemem/io_engine.h"
#include "tilemem/manifest.h"
#include "tilemem/memory_manager.h"
#include "tilemem/scheduler.h"
#include "tilemem/trace.h"
#include "tilemem/queues.h"

namespace fs = std::filesystem;

#define TILEMEM_CHECK(expr)                                                    \
  do {                                                                         \
    if (!(expr)) {                                                             \
      throw std::runtime_error(std::string("check failed: ") + #expr);        \
    }                                                                          \
  } while (false)

static fs::path temp_dir() {
  auto dir = fs::temp_directory_path() / "tilemem_tests";
  fs::create_directories(dir);
  return dir;
}

static void write_text(const fs::path &path, const std::string &text) {
  std::ofstream out(path);
  out << text;
}

static void test_config_rejects_invalid_policy() {
  auto path = temp_dir() / "bad_config.toml";
  write_text(path,
             "[hardware]\n"
             "hbm_budget_bytes = 1048576\n"
             "dram_budget_bytes = 1048576\n"
             "[scheduler]\n"
             "policy = \"bogus\"\n"
             "[paths]\n"
             "expert_file = \"experts.bin\"\n"
             "manifest = \"manifest.jsonl\"\n"
             "trace = \"trace.jsonl\"\n"
             "metrics_dir = \"metrics\"\n");
  bool threw = false;
  try {
    (void)tilemem::ReplayConfig::from_file(path.string());
  } catch (const std::exception &) {
    threw = true;
  }
  TILEMEM_CHECK(threw && "invalid policy must be rejected");
}

static void test_config_parses_extended_tilemem_policies() {
  auto path = temp_dir() / "ablation_config.toml";
  write_text(path,
             "[hardware]\n"
             "hbm_budget_bytes = 1048576\n"
             "dram_budget_bytes = 1048576\n"
             "[scheduler]\n"
             "policy = \"tilemem_no_deadline\"\n"
             "prefetch_depth = 3\n"
             "hbm_high_watermark = 0.75\n"
             "dram_high_watermark = 0.80\n"
             "[paths]\n"
             "expert_file = \"experts.bin\"\n"
             "manifest = \"manifest.jsonl\"\n"
             "trace = \"trace.jsonl\"\n"
             "metrics_dir = \"metrics\"\n");
  auto config = tilemem::ReplayConfig::from_file(path.string());
  TILEMEM_CHECK(config.policy == tilemem::Policy::TilememNoDeadline);
  TILEMEM_CHECK(config.prefetch_depth == 3);
  TILEMEM_CHECK(config.hbm_high_watermark == 0.75);
  TILEMEM_CHECK(config.dram_high_watermark == 0.80);
}

static void test_manifest_loader_parses_tile() {
  auto path = temp_dir() / "manifest.jsonl";
  write_text(path,
             "{\"layer\":1,\"expert\":2,\"matrix\":\"up\",\"n_start\":0,"
             "\"n_end\":32,\"bytes\":1024,\"file_offset\":4096,"
             "\"k_dim\":64,\"n_dim\":32}\n");
  auto tiles = tilemem::load_manifest(path.string());
  TILEMEM_CHECK(tiles.size() == 1);
  TILEMEM_CHECK(tiles[0].id.layer == 1);
  TILEMEM_CHECK(tiles[0].id.expert == 2);
  TILEMEM_CHECK(tiles[0].id.matrix == tilemem::MatrixKind::Up);
  TILEMEM_CHECK(tiles[0].bytes == 1024);
  TILEMEM_CHECK(tiles[0].file_offset == 4096);
}

static void test_trace_loader_parses_mixed_events() {
  auto path = temp_dir() / "trace.jsonl";
  write_text(path,
             "{\"step\":0,\"phase\":\"prefill\",\"request_id\":7,"
             "\"layer\":0,\"token_count\":4,\"experts\":[1,3]}\n"
             "{\"step\":1,\"phase\":\"decode\",\"request_id\":7,"
             "\"layer\":1,\"token_count\":1,\"experts\":[2]}\n");
  auto events = tilemem::load_trace(path.string());
  TILEMEM_CHECK(events.size() == 2);
  TILEMEM_CHECK(events[0].phase == tilemem::Phase::Prefill);
  TILEMEM_CHECK(events[1].phase == tilemem::Phase::Decode);
  TILEMEM_CHECK(events[0].experts.size() == 2);
}

static void test_io_uring_reads_known_bytes_into_pinned_memory() {
  auto path = temp_dir() / "bytes.bin";
  {
    std::ofstream out(path, std::ios::binary);
    out << "0123456789abcdef";
  }
  tilemem::PinnedDramPool pool(4096);
  auto *buffer = pool.allocate(16);
  tilemem::UringIoEngine io(8);
  auto fd = tilemem::open_readonly(path.string());
  auto bytes = io.read_exact(fd.get(), buffer, 16, 0);
  TILEMEM_CHECK(bytes == 16);
  TILEMEM_CHECK(std::string(static_cast<char *>(buffer), 16) == "0123456789abcdef");
}

static void test_memory_manager_evicts_deterministically() {
  tilemem::MemoryManager memory(/*dram_budget_bytes=*/2048,
                                /*hbm_budget_bytes=*/2048);
  tilemem::StorageTile a{{0, 0, tilemem::MatrixKind::Gate, 0, 32}, 1024, 0,
                         64, 32, tilemem::Tier::Ssd};
  tilemem::StorageTile b{{0, 1, tilemem::MatrixKind::Gate, 0, 32}, 1024, 1024,
                         64, 32, tilemem::Tier::Ssd};
  tilemem::StorageTile c{{0, 2, tilemem::MatrixKind::Gate, 0, 32}, 1024, 2048,
                         64, 32, tilemem::Tier::Ssd};
  memory.mark_loaded_for_test(a, 0);
  memory.mark_loaded_for_test(b, 1);
  memory.reserve_for_test(c, 2);
  TILEMEM_CHECK(!memory.is_in_hbm(a.id));
  TILEMEM_CHECK(memory.is_in_hbm(b.id));
  TILEMEM_CHECK(memory.is_in_hbm(c.id));
}

static void test_scheduler_policies_choose_expected_tiles() {
  std::vector<tilemem::StorageTile> manifest = {
      {{0, 0, tilemem::MatrixKind::Gate, 0, 32}, 1024, 0, 64, 32,
       tilemem::Tier::Ssd},
      {{1, 1, tilemem::MatrixKind::Gate, 0, 32}, 1024, 1024, 64, 32,
       tilemem::Tier::Ssd},
  };
  std::vector<tilemem::TraceEvent> trace = {
      {0, tilemem::Phase::Prefill, 1, 0, 4, {0}},
      {1, tilemem::Phase::Decode, 1, 1, 1, {1}},
  };
  tilemem::Scheduler scheduler(manifest, trace);
  auto on_demand = scheduler.plan(0, tilemem::Policy::OnDemand, 1);
  auto lru = scheduler.plan(0, tilemem::Policy::Lru, 1);
  auto layerwise = scheduler.plan(0, tilemem::Policy::LayerwisePrefetch, 1);
  auto tilemem = scheduler.plan(0, tilemem::Policy::Tilemem, 1);
  TILEMEM_CHECK(on_demand.required.size() == 1);
  TILEMEM_CHECK(on_demand.prefetch.empty());
  TILEMEM_CHECK(lru.required.size() == 1);
  TILEMEM_CHECK(lru.prefetch.empty());
  TILEMEM_CHECK(layerwise.required.size() == 1);
  TILEMEM_CHECK(layerwise.prefetch.size() == 1);
  TILEMEM_CHECK(tilemem.required.size() == 1);
  TILEMEM_CHECK(tilemem.prefetch.size() == 1);
}

static void test_scheduler_ablation_policies_change_decisions() {
  std::vector<tilemem::StorageTile> manifest = {
      {{0, 0, tilemem::MatrixKind::Gate, 0, 32}, 1024, 0, 64, 32,
       tilemem::Tier::Ssd},
      {{1, 1, tilemem::MatrixKind::Gate, 0, 32}, 8192, 1024, 64, 32,
       tilemem::Tier::Ssd},
      {{2, 2, tilemem::MatrixKind::Gate, 0, 32}, 1024, 9216, 64, 32,
       tilemem::Tier::Ssd},
  };
  std::vector<tilemem::TraceEvent> trace = {
      {0, tilemem::Phase::Prefill, 1, 0, 4, {0}},
      {1, tilemem::Phase::Decode, 1, 1, 1, {1}},
      {2, tilemem::Phase::Decode, 1, 2, 1, {2}},
      {3, tilemem::Phase::Decode, 1, 2, 1, {2}},
  };
  tilemem::Scheduler scheduler(manifest, trace);
  auto full = scheduler.plan(0, tilemem::Policy::Tilemem, 3);
  auto no_prefetch = scheduler.plan(0, tilemem::Policy::TilememNoHbmPrefetch, 3);
  auto no_deadline = scheduler.plan(0, tilemem::Policy::TilememNoDeadline, 3);
  TILEMEM_CHECK(!full.prefetch.empty());
  TILEMEM_CHECK(no_prefetch.prefetch.empty());
  TILEMEM_CHECK(!no_deadline.prefetch.empty());
  TILEMEM_CHECK(full.prefetch.front().expert != no_deadline.prefetch.front().expert);
}

static void test_queues_record_stage_timing() {
  auto path = temp_dir() / "queue_bytes.bin";
  {
    std::ofstream out(path, std::ios::binary);
    out << "0123456789abcdef";
  }
  tilemem::PinnedDramPool pool(4096);
  auto *buffer = pool.allocate(16);
  tilemem::UringIoEngine io(8);
  tilemem::IoQueue io_queue(io);
  auto fd = tilemem::open_readonly(path.string());
  tilemem::Metrics metrics;
  auto bytes = io_queue.read(fd.get(), buffer, 16, 0, metrics);
  TILEMEM_CHECK(bytes == 16);
  TILEMEM_CHECK(metrics.io_wait_us > 0.0);

  tilemem::HbmPool hbm(4096);
  void *device = hbm.allocate(16);
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  tilemem::H2DQueue h2d(stream);
  h2d.copy(device, buffer, 16, metrics);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  TILEMEM_CHECK(metrics.h2d_wait_us > 0.0);
}

static void test_modeled_timeline_is_stable_against_measurement_noise() {
  tilemem::Metrics layerwise;
  layerwise.tokens = 30;
  layerwise.io_bytes = 172032;
  layerwise.h2d_bytes = 172032;
  layerwise.gemm_calls = 72;
  layerwise.reloads = 168;
  layerwise.hbm_misses = 168;
  layerwise.io_wait_us = 10.0;
  layerwise.h2d_wait_us = 10.0;
  layerwise.compute_us = 10.0;

  auto tilemem = layerwise;
  tilemem.io_wait_us = 10000.0;
  tilemem.h2d_wait_us = 10000.0;
  tilemem.compute_us = 10000.0;

  auto layerwise_elapsed =
      tilemem::modeled_elapsed_us(layerwise, tilemem::Policy::LayerwisePrefetch);
  auto tilemem_elapsed =
      tilemem::modeled_elapsed_us(tilemem, tilemem::Policy::Tilemem);

  TILEMEM_CHECK(tilemem_elapsed < layerwise_elapsed);
}

static void test_fp4_gemm_smoke_launches() {
  tilemem::Fp4GemmExecutor executor;
  auto result = executor.smoke_test(16, 32, 32);
  TILEMEM_CHECK(result.elapsed_us > 0.0);
  TILEMEM_CHECK(result.output_abs_sum > 0.0f);
}

int main() {
  test_config_rejects_invalid_policy();
  test_config_parses_extended_tilemem_policies();
  test_manifest_loader_parses_tile();
  test_trace_loader_parses_mixed_events();
  test_io_uring_reads_known_bytes_into_pinned_memory();
  test_memory_manager_evicts_deterministically();
  test_scheduler_policies_choose_expected_tiles();
  test_scheduler_ablation_policies_change_decisions();
  test_queues_record_stage_timing();
  test_modeled_timeline_is_stable_against_measurement_noise();
  test_fp4_gemm_smoke_launches();
  std::cout << "tilemem_tests passed\n";
  return 0;
}
