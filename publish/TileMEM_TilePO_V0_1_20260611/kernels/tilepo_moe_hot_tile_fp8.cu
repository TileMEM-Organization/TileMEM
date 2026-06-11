#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void tilepo_moe_hot_tile_fp8(
    const uint8_t* a, const uint8_t* b, float* c, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < elements) {
    c[idx] = static_cast<float>(a[idx]) + static_cast<float>(b[idx]);
  }
}

