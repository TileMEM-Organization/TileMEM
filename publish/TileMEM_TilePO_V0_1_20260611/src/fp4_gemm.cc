#include "tilemem/fp4_gemm.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>

namespace tilemem {
namespace {

void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

void require_sm120() {
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  if (prop.major < 12) {
    throw std::runtime_error("TileMEM FP4 GEMM requires SM120+");
  }
}

} // namespace

Fp4GemmExecutor::Fp4GemmExecutor() { require_sm120(); }

GemmRunResult Fp4GemmExecutor::smoke_test(int m, int n, int k) {
  if (k % 2 != 0 || k % 32 != 0 || n % 32 != 0) {
    throw std::runtime_error("FP4 smoke test requires K/N multiples of 32");
  }

  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  half *a = nullptr;
  unsigned char *b = nullptr;
  float *c = nullptr;
  std::size_t a_bytes = static_cast<std::size_t>(m) * k * sizeof(half);
  std::size_t b_bytes = static_cast<std::size_t>(n) * (k / 2);
  std::size_t c_bytes = static_cast<std::size_t>(m) * n * sizeof(float);
  check_cuda(cudaMalloc(&a, a_bytes), "cudaMalloc A");
  check_cuda(cudaMalloc(&b, b_bytes), "cudaMalloc B");
  check_cuda(cudaMalloc(&c, c_bytes), "cudaMalloc C");

  std::vector<unsigned char> host_b(b_bytes);
  for (std::size_t i = 0; i < host_b.size(); ++i) {
    host_b[i] = static_cast<unsigned char>((i % 15) + 1);
  }
  check_cuda(cudaMemcpyAsync(b, host_b.data(), b_bytes, cudaMemcpyHostToDevice,
                             stream),
             "cudaMemcpyAsync B");
  launch_fill_half(a, m * k, stream);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
  check_cuda(cudaEventRecord(start, stream), "cudaEventRecord start");
  launch_fp4_gemm_sm120(a, b, c, m, n, k, stream);
  check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord stop");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize smoke");
  check_cuda(cudaGetLastError(), "FP4 smoke kernel");
  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
             "cudaEventElapsedTime");

  std::vector<float> host_c(static_cast<std::size_t>(m) * n);
  check_cuda(cudaMemcpy(host_c.data(), c, c_bytes, cudaMemcpyDeviceToHost),
             "cudaMemcpy C");
  float sum = 0.0f;
  for (float value : host_c) {
    sum += std::fabs(value);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaStreamDestroy(stream);
  return {static_cast<double>(elapsed_ms) * 1000.0, sum};
}

GemmRunResult Fp4GemmExecutor::run(const StorageTile &tile, void *device_weight,
                                   int token_count, cudaStream_t stream,
                                   Metrics &metrics) {
  int m = std::max(token_count, 1);
  int n = tile.n_dim;
  int k = tile.k_dim;
  if (k % 32 != 0 || n % 32 != 0) {
    throw std::runtime_error("FP4 GEMM requires K/N multiples of 32");
  }

  half *a = nullptr;
  float *c = nullptr;
  std::size_t a_bytes = static_cast<std::size_t>(m) * k * sizeof(half);
  std::size_t c_bytes = static_cast<std::size_t>(m) * n * sizeof(float);
  check_cuda(cudaMalloc(&a, a_bytes), "cudaMalloc replay A");
  check_cuda(cudaMalloc(&c, c_bytes), "cudaMalloc replay C");
  launch_fill_half(a, m * k, stream);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate replay start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate replay stop");
  check_cuda(cudaEventRecord(start, stream), "cudaEventRecord replay start");
  launch_fp4_gemm_sm120(a, static_cast<unsigned char *>(device_weight), c, m, n,
                        k, stream);
  check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord replay stop");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize replay");
  check_cuda(cudaGetLastError(), "FP4 replay kernel");
  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
             "cudaEventElapsedTime replay");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(a);
  cudaFree(c);
  ++metrics.gemm_calls;
  return {static_cast<double>(elapsed_ms) * 1000.0, 1.0f};
}

} // namespace tilemem
