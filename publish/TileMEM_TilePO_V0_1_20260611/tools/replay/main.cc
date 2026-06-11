#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include <cuda_runtime.h>

#include "tilemem/config.h"
#include "tilemem/fp4_gemm.h"
#include "tilemem/io_engine.h"
#include "tilemem/manifest.h"
#include "tilemem/memory_manager.h"
#include "tilemem/metrics.h"
#include "tilemem/queues.h"
#include "tilemem/scheduler.h"
#include "tilemem/trace.h"

namespace fs = std::filesystem;

namespace {

struct Args {
  std::string config;
  std::string data_root;
  std::string policy;
};

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string flag = argv[i];
    if (flag == "--config" && i + 1 < argc) {
      args.config = argv[++i];
    } else if (flag == "--data-root" && i + 1 < argc) {
      args.data_root = argv[++i];
    } else if (flag == "--policy" && i + 1 < argc) {
      args.policy = argv[++i];
    } else {
      throw std::runtime_error("unknown or incomplete argument: " + flag);
    }
  }
  if (args.config.empty()) {
    throw std::runtime_error("usage: tilemem_replay --config PATH");
  }
  return args;
}

void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    auto args = parse_args(argc, argv);
    auto config = tilemem::ReplayConfig::from_file(args.config);
    if (!args.policy.empty()) {
      config.policy = tilemem::parse_policy(args.policy);
    }
    if (!args.data_root.empty()) {
      fs::path root(args.data_root);
      config.expert_file = (root / "experts.bin").string();
      config.manifest_path = (root / "manifest.jsonl").string();
      config.trace_path = (root / "trace.jsonl").string();
    }

    auto manifest = tilemem::load_manifest(config.manifest_path);
    auto trace = tilemem::load_trace(config.trace_path);
    tilemem::Scheduler scheduler(manifest, trace);
    tilemem::MemoryManager memory(config.dram_budget_bytes,
                                  config.hbm_budget_bytes);
    tilemem::UringIoEngine io(config.io_queue_depth);
    tilemem::IoQueue io_queue(io);
    auto fd = tilemem::open_readonly(config.expert_file);
    tilemem::Fp4GemmExecutor gemm;
    tilemem::Metrics metrics;

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    tilemem::H2DQueue h2d_queue(stream);
    tilemem::ComputeQueue compute_queue(gemm, stream);
    std::unordered_set<tilemem::StorageTileId> prefetched;
    std::unordered_set<tilemem::StorageTileId> used;
    bool disable_dram_cache =
        config.policy == tilemem::Policy::TilememNoDramCache;
    auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < trace.size(); ++i) {
      const auto &event = trace[i];
      metrics.tokens += static_cast<std::uint64_t>(event.token_count);
      auto plan = scheduler.plan(i, config.policy, config.lookahead);

      std::unordered_set<tilemem::StorageTileId> seen;
      for (const auto &id : plan.prefetch) {
        if (seen.insert(id).second) {
          const auto &tile = scheduler.tile(id);
          (void)memory.ensure_hbm(tile, fd.get(), io_queue, h2d_queue, metrics,
                                  disable_dram_cache);
          prefetched.insert(id);
        }
      }
      for (const auto &id : plan.required) {
        bool hit_from_prefetch =
            prefetched.find(id) != prefetched.end() && memory.is_in_hbm(id);
        if (hit_from_prefetch) {
          ++metrics.prefetch_hits;
        }
        used.insert(id);
        if (seen.insert(id).second) {
          const auto &tile = scheduler.tile(id);
          auto *weight = memory.ensure_hbm(tile, fd.get(), io_queue, h2d_queue,
                                           metrics, disable_dram_cache);
          (void)compute_queue.run(tile, weight, event.token_count, metrics);
        } else {
          const auto &tile = scheduler.tile(id);
          auto *weight = memory.ensure_hbm(tile, fd.get(), io_queue, h2d_queue,
                                           metrics, disable_dram_cache);
          (void)compute_queue.run(tile, weight, event.token_count, metrics);
        }
      }
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize final");
    auto stop = std::chrono::steady_clock::now();
    metrics.wall_elapsed_sec =
        std::chrono::duration<double>(stop - start).count();
    for (const auto &id : prefetched) {
      if (used.find(id) == used.end()) {
        ++metrics.prefetch_waste;
      }
    }
    metrics.deadline_misses =
        static_cast<std::uint64_t>(metrics.hbm_misses > metrics.prefetch_hits
                                       ? metrics.hbm_misses - metrics.prefetch_hits
                                       : 0);
    metrics.eviction_count = metrics.reloads > metrics.dram_misses
                                 ? metrics.reloads - metrics.dram_misses
                                 : 0;
    metrics.modeled_elapsed_us =
        tilemem::modeled_elapsed_us(metrics, config.policy);
    metrics.overlap_us = tilemem::modeled_overlap_us(metrics, config.policy);
    auto modeled_compute = tilemem::modeled_compute_stage_us(metrics);
    metrics.gpu_idle_us = metrics.modeled_elapsed_us > modeled_compute
                              ? metrics.modeled_elapsed_us - modeled_compute
                              : 0.0;
    metrics.elapsed_sec = metrics.modeled_elapsed_us / 1000000.0;
    metrics.tok_per_sec =
        metrics.elapsed_sec > 0.0
            ? static_cast<double>(metrics.tokens) / metrics.elapsed_sec
            : 0.0;
    cudaStreamDestroy(stream);

    fs::path metrics_path =
        fs::path(config.metrics_dir) /
        (tilemem::to_string(config.policy) + std::string(".json"));
    tilemem::write_metrics_json(metrics, metrics_path.string());
    std::cout << "policy=" << tilemem::to_string(config.policy)
              << " tokens=" << metrics.tokens
              << " tok_per_sec=" << metrics.tok_per_sec
              << " io_bytes=" << metrics.io_bytes
              << " h2d_bytes=" << metrics.h2d_bytes
              << " gemm_calls=" << metrics.gemm_calls << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "tilemem_replay: " << e.what() << "\n";
    return 1;
  }
}
