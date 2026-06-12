#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace {

constexpr int kThreadsX = 16;
constexpr int kThreadsY = 16;
constexpr int kScaleBlockN = 32;

__device__ __forceinline__ float decode_fp8_sample(unsigned char value, const float *scale, int col) {
  int signed_value = static_cast<int>(static_cast<int8_t>(value));
  return static_cast<float>(signed_value) * scale[col / kScaleBlockN];
}

extern "C" __global__ void tilemem_gemm_fp8_kernel(
    const float *a,
    const unsigned char *b_packed,
    const float *scale,
    float *c,
    int m,
    int n,
    int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    float av = a[row * k + kk];
    float bv = decode_fp8_sample(b_packed[kk * n + col], scale, col);
    acc += av * bv;
  }
  c[row * n + col] = acc;
}

}  // namespace

#ifdef TILEMEM_STANDALONE_BENCHMARK

void check_cuda(cudaError_t status, const char *expr) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at %s: %s\n", expr, cudaGetErrorString(status));
    std::exit(1);
  }
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr)

int positive_arg(char **argv, int index, int fallback) {
  if (!argv[index]) {
    return fallback;
  }
  int value = std::atoi(argv[index]);
  return value > 0 ? value : fallback;
}

std::vector<float> make_a(int m, int k) {
  std::vector<float> values(static_cast<size_t>(m) * static_cast<size_t>(k));
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = 0.25f + static_cast<float>((i * 13) % 17) * 0.015625f;
  }
  return values;
}

std::vector<unsigned char> make_b(int k, int n) {
  std::vector<unsigned char> values(static_cast<size_t>(k) * static_cast<size_t>(n));
  for (size_t i = 0; i < values.size(); ++i) {
    int q = static_cast<int>((i * 17 + 3) % 127) - 63;
    values[i] = static_cast<unsigned char>(static_cast<int8_t>(q));
  }
  return values;
}

std::vector<float> make_scales(int n) {
  int blocks = (n + kScaleBlockN - 1) / kScaleBlockN;
  std::vector<float> values(static_cast<size_t>(blocks));
  for (int i = 0; i < blocks; ++i) {
    values[static_cast<size_t>(i)] = 0.0078125f * (1.0f + static_cast<float>(i % 5) * 0.125f);
  }
  return values;
}

float checksum_prefix(const std::vector<float> &values) {
  size_t count = std::min<size_t>(values.size(), 256);
  return std::accumulate(values.begin(), values.begin() + count, 0.0f);
}

int main(int argc, char **argv) {
  int m = argc > 1 ? positive_arg(argv, 1, 64) : 64;
  int n = argc > 2 ? positive_arg(argv, 2, 512) : 512;
  int k = argc > 3 ? positive_arg(argv, 3, 512) : 512;
  int iterations = argc > 4 ? positive_arg(argv, 4, 20) : 20;

  std::vector<float> h_a = make_a(m, k);
  std::vector<unsigned char> h_b = make_b(k, n);
  std::vector<float> h_scale = make_scales(n);
  std::vector<float> h_c(static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);

  float *d_a = nullptr;
  unsigned char *d_b = nullptr;
  float *d_scale = nullptr;
  float *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(unsigned char)));
  CUDA_CHECK(cudaMalloc(&d_scale, h_scale.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, h_c.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scale, h_scale.data(), h_scale.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(kThreadsX, kThreadsY);
  dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);

  for (int i = 0; i < 3; ++i) {
    tilemem_gemm_fp8_kernel<<<blocks, threads>>>(d_a, d_b, d_scale, d_c, m, n, k);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    tilemem_gemm_fp8_kernel<<<blocks, threads>>>(d_a, d_b, d_scale, d_c, m, n, k);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float), cudaMemcpyDeviceToHost));
  float checksum = checksum_prefix(h_c);

  size_t packed_weight_bytes = h_b.size() * sizeof(unsigned char);
  size_t scale_bytes = h_scale.size() * sizeof(float);
  size_t bf16_weight_bytes = static_cast<size_t>(k) * static_cast<size_t>(n) * 2;
  double payload_reduction = 1.0 - static_cast<double>(packed_weight_bytes + scale_bytes) /
                                     static_cast<double>(bf16_weight_bytes);
  double avg_ms = static_cast<double>(elapsed_ms) / static_cast<double>(iterations);
  double ops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) *
               static_cast<double>(iterations);
  double gflops = ops / (static_cast<double>(elapsed_ms) * 1.0e6);

  std::printf(
      "{\"precision\":\"fp8\",\"status\":\"success\",\"m\":%d,\"n\":%d,\"k\":%d,"
      "\"iterations\":%d,\"time_ms\":%.6f,\"avg_ms\":%.6f,\"gflops\":%.6f,"
      "\"packed_weight_bytes\":%zu,\"scale_bytes\":%zu,\"bf16_weight_bytes\":%zu,"
      "\"payload_reduction_vs_bf16\":%.6f,\"checksum\":%.6f}\n",
      m,
      n,
      k,
      iterations,
      elapsed_ms,
      avg_ms,
      gflops,
      packed_weight_bytes,
      scale_bytes,
      bf16_weight_bytes,
      payload_reduction,
      checksum);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_scale));
  CUDA_CHECK(cudaFree(d_c));
  return 0;
}

#endif  // TILEMEM_STANDALONE_BENCHMARK

extern "C" cudaError_t tilemem_launch_gemm_fp8(
    const float *a,
    const unsigned char *b_packed,
    const float *scale,
    float *c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  dim3 threads(kThreadsX, kThreadsY);
  dim3 blocks((n + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
  tilemem_gemm_fp8_kernel<<<blocks, threads, 0, stream>>>(a, b_packed, scale, c, m, n, k);
  return cudaGetLastError();
}
