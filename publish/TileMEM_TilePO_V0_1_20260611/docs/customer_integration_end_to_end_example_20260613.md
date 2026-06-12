# Customer End-to-End Integration Example

This document explains the customer-facing integration demo for TileMEM's
external kernel interface. The demo uses an OLMoE-like offline fixture and a
real CUDA sample backend to show how TileMEM hands tile placement and dispatch
metadata to customer-owned FP8, FP6, and FP4 kernels.

The sample is a contract demo, not a production performance or model-quality
claim. The CUDA kernels under `kernels/` are real `__global__` kernels with C
ABI launch wrappers, but they use simple software dequantization so the
interface can be inspected and run without hiding the boundary behind a vendor
kernel library.

## What The Demo Shows

The end-to-end path is:

1. A model adapter describes an OLMoE-like set of layers, experts, projection
   groups, and shards.
2. TileMEM creates stable tile IDs for those shards and records where each
   packed tile and its scale metadata live in the manifest.
3. An external backend registers the formats and scale layouts it can consume.
4. TileMEM builds runtime tile handles from the manifest and the backend
   registry.
5. Dispatch chooses either the registered external backend or the BF16 fallback
   chain for each tile.
6. The standalone benchmark compiles and runs the sample CUDA kernels, then
   reports payload size, dispatchability, and small-kernel runtime measurements.

The reference entry points are:

- `examples/olmoe_external_cuda_backend.py`
- `tilepo/integration.py`
- `kernels/gemm_fp8.cu`
- `kernels/gemm_fp6.cu`
- `kernels/gemm_fp4.cu`
- `docs/olmoe_integration_interface_benchmark_20260612.md`

## Ownership Boundary

TileMEM owns the integration contract and runtime metadata:

- tile splitting and stable tile ID management
- tile dtype tags and tile format metadata
- scale metadata offset, byte size, granularity, and layout tags
- backend capability registration
- manifest generation
- runtime dispatch tile handles
- fallback chain metadata
- enough tile metadata for an external kernel to know where the tile is, what
  format it uses, where scales are, and which backend should dispatch it

The external developer owns the numerical implementation and model-specific
policy:

- concrete FP8, FP6, and FP4 kernels
- model structure adapter
- weight quantization
- calibration
- quality evaluation
- backend-specific packed layout
- CUDA, TileLang, Triton, ROCm, or other backend implementation

In the sample, TileMEM does not decide whether FP8, FP6, or FP4 is acceptable
for a customer model. It records the customer's format choice, verifies whether
a registered backend says it can consume that format and scale layout, and
keeps BF16 fallback metadata available.

## Backend Registration

The sample backend registers this capability:

```python
BackendCapability(
    name="olmoe_external_cuda",
    formats=["fp8_e4m3_sample", "fp6_s6_sample", "fp4_s4_sample"],
    layouts=["block_n32_fp32", "block_n64_fp32"],
    scale_granularities=["block"],
    projection_groups=["gate_up", "down"],
    runtime_entrypoint="kernels/gemm_fp{4,6,8}.cu:tilemem_launch_gemm_fp{4,6,8}",
    owns_quantization=True,
    owns_calibration=True,
    owns_quality=True,
    hardware_targets=["cuda_native_arch"],
    fallback_dtype="bf16",
)
```

Those `owns_*` fields are part of the contract: TileMEM knows that this backend
is customer-owned for quantization, calibration, and quality. TileMEM can route
tiles to it when the format and scale layout match, but the developer remains
responsible for the correctness of the low-precision representation.

## Manifest Metadata

For each tile, TileMEM emits manifest entries that identify the tile and its
payload:

- `tile_ids`: layer, expert, projection group, shard ID, and `n_start:n_end`
  range
- `tile_offsets` and `tile_bytes`: where the packed weight tile starts and how
  many bytes it occupies
- `tile_dtype_map`: logical dtype tag such as `bf16`, `fp8`, `fp6`, or `fp4`
- `tile_format_map`: external format name, storage bits, accumulation dtype,
  and layout owner
- `scale_offsets` and `scale_bytes`: where per-tile scale metadata starts and
  how many bytes it occupies
- `scale_layout_map`: scale layout tag such as `block_n32_fp32` or
  `block_n64_fp32`
- `backend_owner_map`: preferred backend for the tile
- `tile_fallback_map`: fallback dtype and backend
- `gpu_hot_tiles`: residency hint used by the runtime handle builder
- `fallback_chain`: ordered fallback descriptor, currently `["bf16",
  "kt_fallback"]` in the sample fixture

This is the minimum information an external runtime needs to locate a packed
tile, interpret its format tag, find its scale metadata, and determine whether
to dispatch through the external backend or the fallback path.

## Runtime Tile Handle

`tilepo/integration.py` converts the manifest into `TileHandle` objects. A
handle carries the stable tile key, model coordinates, byte offsets, scale
metadata, residency, backend name, fallback backend, and a `dispatchable` flag.

The dispatchability check is deliberately narrow. A tile is dispatchable when:

- the backend is the built-in `kt_fallback`, or
- the registered external backend supports the tile format, scale layout, and
  projection group.

If the backend does not support the tile, the handle still carries the BF16
fallback metadata. This lets a runtime keep routing deterministic even when a
low-precision path is unavailable for a specific tile, projection, or platform.

## Real CUDA Sample

The three sample kernels demonstrate the external side of the interface:

- `kernels/gemm_fp8.cu` exports `tilemem_launch_gemm_fp8`
- `kernels/gemm_fp6.cu` exports `tilemem_launch_gemm_fp6`
- `kernels/gemm_fp4.cu` exports `tilemem_launch_gemm_fp4`

Each wrapper accepts input activation, packed weight, scale, output, GEMM
shape, and CUDA stream arguments. The kernels decode the packed sample format
and accumulate into FP32. The FP8 sample uses one byte per stored value with
`block_n32_fp32` scales. The FP6 and FP4 samples use packed bit layouts with
`block_n64_fp32` scales.

These kernels are intentionally simple. They are useful for validating ABI
shape, metadata handoff, scale lookup, and dispatch wiring. They are not a
claim of production Tensor Core throughput.

## Running The Demo

Run the benchmark wrapper:

```bash
tools/benchmark_olmoe_integration_interface \
  --out-dir build/olmoe_cuda_integration_real \
  --iterations 20 \
  --m 64 \
  --n 512 \
  --k 512
```

When `nvcc` is available, the wrapper compiles:

- `build/olmoe_cuda_integration_real/cuda_build/gemm_fp8`
- `build/olmoe_cuda_integration_real/cuda_build/gemm_fp6`
- `build/olmoe_cuda_integration_real/cuda_build/gemm_fp4`

It also writes:

- `build/olmoe_cuda_integration_real/olmoe_integration_summary.json`
- `build/olmoe_cuda_integration_real/olmoe_integration_report.md`

If CUDA is unavailable, the integration metadata can still be inspected, but
the standalone CUDA runtime portion is skipped or reported as not compiled.

## What To Look For In The Output

The output should show:

- the external backend capability registered under `olmoe_external_cuda`
- 1024 fixture tiles for the OLMoE-like example
- FP8, FP6, and FP4 tiles carrying customer-owned format tags
- scale metadata reported separately from packed weight bytes
- all supported sample tiles marked dispatchable when the capability matches
- BF16 fallback descriptors present in the manifest and handles
- CUDA compile/runtime status for each real sample kernel when CUDA is present

Payload reduction is reported from packed weight bytes plus scale metadata
relative to the BF16 fixture payload. Treat those values as interface and
storage-accounting evidence only. They do not establish model accuracy,
calibration quality, or universal low-precision suitability.

## Customer Integration Checklist

To replace the sample kernels with a customer backend:

1. Implement the model adapter that maps model layers, experts, projections,
   and shards into TileMEM tile IDs.
2. Quantize weights into the customer's packed FP8, FP6, FP4, or other format.
3. Produce the matching scale metadata and choose explicit scale layout tags.
4. Register a `BackendCapability` with the supported formats, layouts,
   projection groups, hardware targets, and runtime entrypoint.
5. Provide launchable kernels for the registered backend.
6. Use TileMEM's manifest and `TileHandle` fields to locate packed weights and
   scales at runtime.
7. Keep BF16 or another validated fallback path for unsupported tiles or
   quality-sensitive regions.
8. Run customer-owned calibration and quality evaluation before relying on the
   low-precision path in a model deployment.

The important separation is that TileMEM carries the integration metadata and
fallback routing, while the external developer supplies and validates the
numerical kernel path.

## Non-Claims

This demo does not claim:

- production optimized Tensor Core performance
- production OLMoE serving throughput
- universal FP8, FP6, or FP4 model quality
- calibration correctness for a real model
- a required CUDA-only implementation path

It does claim a working integration contract: TileMEM can describe externally
owned low-precision tiles, register a backend capability, build dispatchable
tile handles, preserve fallback metadata, and call real customer-style CUDA
kernels through the documented boundary.
