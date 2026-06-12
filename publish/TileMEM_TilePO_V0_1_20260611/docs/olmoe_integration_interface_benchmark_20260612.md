# OLMoE External CUDA Integration Benchmark

This document records the real CUDA integration sample for TileMEM's external
kernel interface. TileMEM provides tile IDs, dtype/format tags, scale metadata,
tile handles, dispatch metadata, and BF16 fallback descriptors. The FP8, FP6,
and FP4 GEMM implementations are customer-owned CUDA samples under `kernels/`.

The benchmark uses actual `__global__` CUDA kernels compiled from:

- `kernels/gemm_fp8.cu`
- `kernels/gemm_fp6.cu`
- `kernels/gemm_fp4.cu`

These kernels use software dequantization to demonstrate the integration
contract. They do not claim model quality, calibration quality, or production
Tensor Core throughput for OLMoE.

## Running

```bash
tools/benchmark_olmoe_integration_interface \
  --out-dir build/olmoe_cuda_integration_real \
  --iterations 20 \
  --m 64 \
  --n 512 \
  --k 512
```

Outputs:

- `build/olmoe_cuda_integration_real/olmoe_integration_summary.json`
- `build/olmoe_cuda_integration_real/olmoe_integration_report.md`
- `build/olmoe_cuda_integration_real/cuda_build/gemm_fp8`
- `build/olmoe_cuda_integration_real/cuda_build/gemm_fp6`
- `build/olmoe_cuda_integration_real/cuda_build/gemm_fp4`

## Environment

Measured on 2026-06-12:

- GPU: NVIDIA GeForce RTX 5090
- Driver: 591.74
- NVCC: `/usr/local/cuda-13.2/bin/nvcc`
- Compile flag: `-arch=native`
- CUDA GEMM micro-shape: `M=64, N=512, K=512`
- Iterations: 20

## Results

The payload numbers are computed over the OLMoE-like fixture:

- layers: 16
- experts per layer: 64
- active experts per layer in fixture: 8
- projection groups: `gate_up`, `down`
- shards per projection: 4
- total tiles: 1024

| Case | Backend | Payload bytes | Reduction vs BF16 | CUDA avg ms | CUDA GFLOP/s | Dispatchable tiles |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `bf16_fp32_baseline` | `kt_fallback` | 2147483648 | 0.00% | n/a | n/a | 1024 |
| `fp8_external_cuda_packed` | `olmoe_external_cuda` | 1073807360 | 50.00% | 0.024390 | 1375.723 | 1024 |
| `fp6_external_cuda_packed` | `olmoe_external_cuda` | 805339136 | 62.50% | 0.025274 | 1327.647 | 1024 |
| `fp4_external_cuda_packed` | `olmoe_external_cuda` | 536903680 | 75.00% | 0.016643 | 2016.105 | 1024 |

## Interpretation

This benchmark validates the integration interface contract:

- external backend capability registration
- tile handle construction from manifest metadata
- FP8/FP6/FP4 dtype and format tags
- scale metadata address/size/layout reporting
- C ABI launch function naming
- dispatchable versus fallback tile accounting
- real CUDA compile and execution for customer-owned kernels

The runtime numbers are for the standalone reference kernels only. This
benchmark does not claim GPU runtime speed for a production OLMoE deployment,
model accuracy, or universal low-precision support.
