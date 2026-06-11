#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void tilepo_moe_hot_tile_mxfp4(
    const uint8_t* packed_a, const uint8_t* packed_b, float* c, int elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < elements) {
    uint8_t a = packed_a[idx >> 1];
    uint8_t b = packed_b[idx >> 1];
    uint8_t an = (idx & 1) ? (a >> 4) : (a & 0x0f);
    uint8_t bn = (idx & 1) ? (b >> 4) : (b & 0x0f);
    c[idx] = static_cast<float>(an) + static_cast<float>(bn);
  }
}

