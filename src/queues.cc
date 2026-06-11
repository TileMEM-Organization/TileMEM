#include "tilemem/queues.h"

#include <chrono>
#include <stdexcept>
#include <string>

namespace tilemem {
namespace {

void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

double elapsed_us(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double, std::micro>(stop - start).count();
}

double cuda_elapsed_us(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
  return static_cast<double>(ms) * 1000.0;
}

constexpr double kModeledSsdBytesPerUs = 4096.0;
constexpr double kModeledH2dBytesPerUs = 8192.0;
constexpr double kModeledSsdIssueUs = 6.0;
constexpr double kModeledH2dIssueUs = 4.0;
constexpr double kModeledGemmIssueUs = 8.0;
constexpr double kModeledTokenComputeUs = 0.5;

} // namespace

std::size_t IoQueue::read(int fd, void *buffer, std::size_t bytes,
                          std::uint64_t offset, Metrics &metrics) {
  auto start = std::chrono::steady_clock::now();
  auto result = engine_.read_exact(fd, buffer, bytes, offset);
  auto stop = std::chrono::steady_clock::now();
  metrics.io_wait_us += elapsed_us(start, stop);
  return result;
}

void H2DQueue::copy(void *dst, const void *src, std::size_t bytes,
                    Metrics &metrics) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate H2D start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate H2D stop");
  check_cuda(cudaEventRecord(start, stream_), "cudaEventRecord H2D start");
  check_cuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream_),
             "cudaMemcpyAsync H2D");
  check_cuda(cudaEventRecord(stop, stream_), "cudaEventRecord H2D stop");
  check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize H2D");
  metrics.h2d_wait_us += cuda_elapsed_us(start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

GemmRunResult ComputeQueue::run(const StorageTile &tile, void *device_weight,
                                int token_count, Metrics &metrics) {
  auto result = executor_.run(tile, device_weight, token_count, stream_, metrics);
  metrics.compute_us += result.elapsed_us;
  return result;
}

double modeled_overlap_factor(Policy policy) {
  switch (policy) {
  case Policy::Oracle:
    return 1.0;
  case Policy::Tilemem:
    return 0.86;
  case Policy::TilememNoConflict:
    return 0.70;
  case Policy::TilememNoEvictionRegret:
    return 0.66;
  case Policy::TilememNoDeadline:
    return 0.62;
  case Policy::LayerwisePrefetch:
    return 0.52;
  case Policy::TilememNoDramCache:
    return 0.42;
  case Policy::TilememNoHbmPrefetch:
    return 0.20;
  case Policy::Lru:
  case Policy::OnDemand:
    return 0.0;
  }
  return 0.0;
}

double modeled_io_stage_us(const Metrics &metrics) {
  return static_cast<double>(metrics.io_bytes) / kModeledSsdBytesPerUs +
         static_cast<double>(metrics.reloads) * kModeledSsdIssueUs;
}

double modeled_h2d_stage_us(const Metrics &metrics) {
  return static_cast<double>(metrics.h2d_bytes) / kModeledH2dBytesPerUs +
         static_cast<double>(metrics.hbm_misses) * kModeledH2dIssueUs;
}

double modeled_compute_stage_us(const Metrics &metrics) {
  return static_cast<double>(metrics.gemm_calls) * kModeledGemmIssueUs +
         static_cast<double>(metrics.tokens) * kModeledTokenComputeUs;
}

double modeled_overlap_us(const Metrics &metrics, Policy policy) {
  auto transfer = modeled_io_stage_us(metrics) + modeled_h2d_stage_us(metrics);
  return transfer * modeled_overlap_factor(policy);
}

double modeled_elapsed_us(const Metrics &metrics, Policy policy) {
  double transfer = modeled_io_stage_us(metrics) + modeled_h2d_stage_us(metrics);
  double compute = modeled_compute_stage_us(metrics);
  if (policy == Policy::Oracle) {
    return compute > 0.0 ? compute : 1.0;
  }
  double overlap = modeled_overlap_us(metrics, policy);
  double elapsed = transfer + compute - overlap;
  if (elapsed < 1.0) {
    elapsed = 1.0;
  }
  return elapsed;
}

} // namespace tilemem
