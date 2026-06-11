#include "tilemem/fp4_gemm.h"

#include <cuda_fp16.h>

namespace tilemem {
namespace {

__device__ float fp4_e2m1_to_float(unsigned char nibble) {
  switch (nibble & 0x0F) {
  case 0x0:
  case 0x8:
    return 0.0f;
  case 0x1:
    return 0.5f;
  case 0x2:
    return 1.0f;
  case 0x3:
    return 1.5f;
  case 0x4:
    return 2.0f;
  case 0x5:
    return 3.0f;
  case 0x6:
    return 4.0f;
  case 0x7:
    return 6.0f;
  case 0x9:
    return -0.5f;
  case 0xA:
    return -1.0f;
  case 0xB:
    return -1.5f;
  case 0xC:
    return -2.0f;
  case 0xD:
    return -3.0f;
  case 0xE:
    return -4.0f;
  case 0xF:
    return -6.0f;
  }
  return 0.0f;
}

__global__ void fill_half_kernel(half *a, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    float value = 1.0f + static_cast<float>(idx % 7) * 0.125f;
    a[idx] = __float2half(value);
  }
}

__global__ void fp4_gemm_kernel(const half *a, const unsigned char *b_packed,
                                float *c, int m, int n, int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) {
    return;
  }
  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    unsigned char packed = b_packed[col * (k / 2) + kk / 2];
    unsigned char nibble = (kk & 1) ? (packed >> 4) : (packed & 0x0F);
    acc += __half2float(a[row * k + kk]) * fp4_e2m1_to_float(nibble);
  }
  c[row * n + col] = acc;
}

} // namespace

void launch_fill_half(half *a, int count, cudaStream_t stream) {
  int threads = 256;
  int blocks = (count + threads - 1) / threads;
  fill_half_kernel<<<blocks, threads, 0, stream>>>(a, count);
}

void launch_fp4_gemm_sm120(const half *a, const unsigned char *b_packed,
                           float *c, int m, int n, int k,
                           cudaStream_t stream) {
  dim3 threads(16, 16);
  dim3 blocks((n + threads.x - 1) / threads.x,
              (m + threads.y - 1) / threads.y);
  fp4_gemm_kernel<<<blocks, threads, 0, stream>>>(a, b_packed, c, m, n, k);
}

} // namespace tilemem
