#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "tilemem/fp4_gemm.h"
#include "tilemem/io_engine.h"
#include "tilemem/metrics.h"
#include "tilemem/types.h"

namespace tilemem {

class IoQueue {
public:
  explicit IoQueue(UringIoEngine &engine) : engine_(engine) {}

  std::size_t read(int fd, void *buffer, std::size_t bytes,
                   std::uint64_t offset, Metrics &metrics);

private:
  UringIoEngine &engine_;
};

class H2DQueue {
public:
  explicit H2DQueue(cudaStream_t stream) : stream_(stream) {}

  void copy(void *dst, const void *src, std::size_t bytes, Metrics &metrics);

private:
  cudaStream_t stream_ = nullptr;
};

class ComputeQueue {
public:
  ComputeQueue(Fp4GemmExecutor &executor, cudaStream_t stream)
      : executor_(executor), stream_(stream) {}

  GemmRunResult run(const StorageTile &tile, void *device_weight,
                    int token_count, Metrics &metrics);

private:
  Fp4GemmExecutor &executor_;
  cudaStream_t stream_ = nullptr;
};

double modeled_overlap_factor(Policy policy);
double modeled_io_stage_us(const Metrics &metrics);
double modeled_h2d_stage_us(const Metrics &metrics);
double modeled_compute_stage_us(const Metrics &metrics);
double modeled_overlap_us(const Metrics &metrics, Policy policy);
double modeled_elapsed_us(const Metrics &metrics, Policy policy);

} // namespace tilemem
