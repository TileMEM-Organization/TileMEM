#pragma once

#include <cstddef>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "tilemem/metrics.h"
#include "tilemem/types.h"

namespace tilemem {

struct GemmRunResult {
  double elapsed_us = 0.0;
  float output_abs_sum = 0.0f;
};

void launch_fp4_gemm_sm120(const half *a, const unsigned char *b_packed,
                           float *c, int m, int n, int k,
                           cudaStream_t stream);

void launch_fill_half(half *a, int count, cudaStream_t stream);

class Fp4GemmExecutor {
public:
  Fp4GemmExecutor();

  GemmRunResult smoke_test(int m, int n, int k);
  GemmRunResult run(const StorageTile &tile, void *device_weight,
                    int token_count, cudaStream_t stream, Metrics &metrics);
};

} // namespace tilemem
