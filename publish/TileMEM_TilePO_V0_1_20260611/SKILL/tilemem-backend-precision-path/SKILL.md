---
name: tilemem-backend-precision-path
description: Use when adding or modifying TileMEM low-precision backend integration, FP8/F6/F4 metadata, scale layouts, backend capabilities, TileHandle dispatch, external CUDA/TileLang/Triton/ROCm kernels, or fallback chains.
---

# TileMEM Backend Precision Path

## Overview

Keep TileMEM responsible for tile planning and dispatch metadata, not model-specific quantization or quality. Low-precision support is an integration contract unless real kernels, calibration, quality gates, and performance evidence are added.

## Ownership Boundary

TileMEM owns:

- tile splitting and stable tile IDs;
- tile dtype tags and format metadata;
- scale metadata address, size, and layout descriptions;
- backend capability registration;
- manifest generation;
- runtime `TileHandle` construction;
- fallback chain descriptors.

External backend owners provide:

- FP8/F6/F4 kernels and backend-specific layouts;
- model architecture adaptation;
- weight quantization and packing;
- calibration method and scale generation;
- quality evaluation and admission gates;
- CUDA, TileLang, Triton, ROCm, or serving-runtime implementation.

## Likely Files

Inspect before editing:

- `tilepo/mir/schema.py` for public dtype/MIR contracts;
- `tilepo/integration.py` for `TileFormat`, `ScaleLayout`, `BackendCapability`, `BackendRegistry`, and `TileHandle`;
- `tilemem/sdk.py` and `tilemem/__init__.py` for public `import tilemem as TM` exports;
- `examples/olmoe_external_cuda_backend.py` and customer examples for integration flows;
- `kernels/gemm_fp8.cu`, `kernels/gemm_fp6.cu`, `kernels/gemm_fp4.cu` for sample C ABI launchers;
- `tools/benchmark_olmoe_integration_interface` for smoke and benchmark evidence.

## Implementation Pattern

1. Add or extend metadata first: dtype, format, scale layout, backend capability, fallback descriptor.
2. Generate or update manifest/handles so external kernels can answer: where is this tile, what format is it, where are scales, which backend owns it, and what fallback exists?
3. Register backend capability with explicit formats, layouts, projection groups, runtime entrypoint, hardware targets, and ownership flags.
4. Add or update sample kernel entrypoints behind stable names such as `tilemem_launch_gemm_fp8`.
5. Preserve BF16/KT fallback for unsupported tiles, unsupported hardware, and failed quality gates.

## Required Tests

Run focused tests before broad verification:

```bash
python3 tools/tests/assert_integration_interface.py
python3 tools/tests/assert_olmoe_integration_benchmark.py
python3 tools/tests/assert_public_mir_interface.py
python3 tools/tests/assert_tilemem_industrial_quickstart.py
python3 tools/benchmark_olmoe_integration_interface \
  --out-dir /tmp/tilemem_olmoe_backend \
  --iterations 7
```

If CUDA is available, also verify compile/runtime paths for the touched kernel. Finish with:

```bash
bash scripts/verify_artifact.sh
```

## Red Lines

- Do not bake CUDA-specific assumptions into generic MIR validation unless expressed as backend capability.
- Do not claim FP8/F6/F4 model-quality safety without calibration and quality evidence.
- Do not remove or weaken BF16/KT fallback.
- Do not dispatch a tile to an external backend unless format, layout, projection group, and hardware capability match.
- Do not report performance wins without naming baseline, GPU, shapes, warmups/repeats, and statistic.
