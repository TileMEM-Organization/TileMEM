#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from tilepo.integration import (
    BackendCapability,
    TileHandle,
    backend_registry,
    benchmark_dispatch_plan,
    build_tile_handles,
    register_backend,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "tilepo_olmoe_integration_benchmark_v1"
BACKEND_NAME = "olmoe_external_cuda"
RUNTIME_ENTRYPOINT = "kernels/gemm_fp{4,6,8}.cu:tilemem_launch_gemm_fp{4,6,8}"
DISCLAIMER = (
    "This benchmark uses actual __global__ CUDA kernels compiled from "
    "kernels/gemm_fp8.cu, kernels/gemm_fp6.cu, and kernels/gemm_fp4.cu. "
    "The kernels are customer-owned integration samples with software "
    "dequantization; they do not provide model quality, calibration, or "
    "production Tensor Core optimization claims."
)

OLMOE_LIKE_CONFIG = {
    "family": "OLMoE-like offline fixture",
    "layers": 16,
    "experts_per_layer": 64,
    "active_experts_per_layer": 8,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "projection_groups": ["gate_up", "down"],
    "shards_per_projection": 4,
    "tile_n": 512,
}


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    description: str
    dtype: str
    format_name: str
    storage_bits: int
    scale_layout: str
    scale_block_size: int
    backend: str
    residency_policy: str
    cuda_kernel_source: str | None
    c_abi_launch_function: str | None


BENCHMARK_CASES = [
    BenchmarkCase(
        name="bf16_fp32_baseline",
        description="BF16 payload baseline through the KT fallback placement path.",
        dtype="bf16",
        format_name="bf16_accum_fp32",
        storage_bits=16,
        scale_layout="none",
        scale_block_size=0,
        backend="kt_fallback",
        residency_policy="hot",
        cuda_kernel_source=None,
        c_abi_launch_function=None,
    ),
    BenchmarkCase(
        name="fp8_external_cuda_packed",
        description="FP8 packed weight tiles dispatched to a customer-owned CUDA sample.",
        dtype="fp8",
        format_name="fp8_e4m3_sample",
        storage_bits=8,
        scale_layout="block_n32_fp32",
        scale_block_size=32,
        backend=BACKEND_NAME,
        residency_policy="hot",
        cuda_kernel_source="kernels/gemm_fp8.cu",
        c_abi_launch_function="tilemem_launch_gemm_fp8",
    ),
    BenchmarkCase(
        name="fp6_external_cuda_packed",
        description="FP6 packed weight tiles dispatched to a customer-owned CUDA sample.",
        dtype="fp6",
        format_name="fp6_s6_sample",
        storage_bits=6,
        scale_layout="block_n64_fp32",
        scale_block_size=64,
        backend=BACKEND_NAME,
        residency_policy="hot",
        cuda_kernel_source="kernels/gemm_fp6.cu",
        c_abi_launch_function="tilemem_launch_gemm_fp6",
    ),
    BenchmarkCase(
        name="fp4_external_cuda_packed",
        description="FP4 packed weight tiles dispatched to a customer-owned CUDA sample.",
        dtype="fp4",
        format_name="fp4_s4_sample",
        storage_bits=4,
        scale_layout="block_n64_fp32",
        scale_block_size=64,
        backend=BACKEND_NAME,
        residency_policy="cold",
        cuda_kernel_source="kernels/gemm_fp4.cu",
        c_abi_launch_function="tilemem_launch_gemm_fp4",
    ),
]


def external_cuda_capability() -> BackendCapability:
    return BackendCapability(
        name=BACKEND_NAME,
        formats=["fp8_e4m3_sample", "fp6_s6_sample", "fp4_s4_sample"],
        layouts=["block_n32_fp32", "block_n64_fp32"],
        scale_granularities=["block"],
        projection_groups=["gate_up", "down"],
        runtime_entrypoint=RUNTIME_ENTRYPOINT,
        owns_quantization=True,
        owns_calibration=True,
        owns_quality=True,
        hardware_targets=["cuda_native_arch"],
        fallback_dtype="bf16",
    )


def register_external_cuda_backend(registry: Any | None = None) -> BackendCapability:
    return register_backend(external_cuda_capability(), registry=registry)


def run_benchmark(
    *,
    iterations: int = 25,
    m: int = 16,
    n: int = 256,
    k: int = 256,
    build_dir: Path | None = None,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")

    registry = backend_registry()
    capability = register_external_cuda_backend(registry)
    build_dir = build_dir or ROOT / "build" / "olmoe_cuda_gemm"
    cuda_environment = detect_cuda_environment()
    kernel_compile = compile_cuda_kernels(build_dir, cuda_environment)
    kernel_runtime = run_cuda_kernels(kernel_compile, m=m, n=n, k=k, iterations=iterations)

    cases = []
    baseline_payload = 0
    for case in BENCHMARK_CASES:
        manifest = build_case_manifest(case)
        handles = build_tile_handles(manifest, registry=registry)
        dispatch_summary = benchmark_dispatch_plan(handles, iterations=iterations)
        precision = case.dtype if case.backend == BACKEND_NAME else None
        case_summary = summarize_case(
            case,
            handles,
            dispatch_summary,
            compile_result=kernel_compile.get(precision) if precision else None,
            runtime_result=kernel_runtime.get(precision) if precision else None,
        )
        if case.name == "bf16_fp32_baseline":
            baseline_payload = case_summary["estimated_payload_bytes"]
        cases.append(case_summary)

    for case_summary in cases:
        if baseline_payload:
            reduction = 1.0 - (case_summary["estimated_payload_bytes"] / baseline_payload)
        else:
            reduction = 0.0
        case_summary["payload_reduction_vs_bf16"] = round(max(0.0, reduction), 6)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_on": "2026-06-12",
        "offline": True,
        "real_cuda_sources": True,
        "simulated_external_cuda": False,
        "disclaimer": DISCLAIMER,
        "iterations": iterations,
        "cuda_gemm_shape": {"m": m, "n": n, "k": k},
        "model": dict(OLMOE_LIKE_CONFIG),
        "registered_external_backend": capability.to_dict(),
        "cuda_environment": cuda_environment,
        "kernel_compile": kernel_compile,
        "cases": cases,
    }


def detect_cuda_environment() -> dict[str, Any]:
    nvcc = shutil.which("nvcc")
    gpu_name = ""
    driver_version = ""
    try:
        gpu_query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        if gpu_query.returncode == 0 and gpu_query.stdout.strip():
            first = gpu_query.stdout.strip().splitlines()[0]
            pieces = [piece.strip() for piece in first.split(",")]
            gpu_name = pieces[0]
            driver_version = pieces[1] if len(pieces) > 1 else ""
    except (OSError, subprocess.TimeoutExpired):
        pass

    arch = "native"
    compile_skipped_reason = "" if nvcc else "nvcc not found"
    return {
        "nvcc_available": nvcc is not None,
        "nvcc_path": nvcc or "",
        "gpu_available": bool(gpu_name),
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "cuda_compile_arch": arch,
        "compile_skipped_reason": compile_skipped_reason,
    }


def compile_cuda_kernels(build_dir: Path, cuda_environment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    nvcc = cuda_environment.get("nvcc_path", "")
    results: dict[str, dict[str, Any]] = {}
    for precision in ("fp8", "fp6", "fp4"):
        source = ROOT / "kernels" / f"gemm_{precision}.cu"
        binary = build_dir / f"gemm_{precision}"
        if not nvcc:
            results[precision] = {
                "status": "skipped_no_nvcc",
                "source": _display_path(source),
                "binary": _display_path(binary),
                "command": [],
                "stderr": "",
            }
            continue
        command = [
            nvcc,
            "-O3",
            "-std=c++17",
            "-DTILEMEM_STANDALONE_BENCHMARK",
            "-arch=native",
            str(source),
            "-o",
            str(binary),
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            fallback_command = [
                nvcc,
                "-O3",
                "-std=c++17",
                "-DTILEMEM_STANDALONE_BENCHMARK",
                str(source),
                "-o",
                str(binary),
            ]
            fallback = subprocess.run(fallback_command, cwd=ROOT, capture_output=True, text=True, check=False)
            command = fallback_command
            completed = fallback
        results[precision] = {
            "status": "compiled" if completed.returncode == 0 else "compile_failed",
            "source": _display_path(source),
            "binary": _display_path(binary),
            "command": command,
            "stderr": completed.stderr[-4000:],
        }
    return results


def run_cuda_kernels(
    kernel_compile: dict[str, dict[str, Any]],
    *,
    m: int,
    n: int,
    k: int,
    iterations: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for precision, compile_result in kernel_compile.items():
        if compile_result["status"] != "compiled":
            results[precision] = {
                "status": "skipped_not_compiled",
                "precision": precision,
                "stderr": compile_result.get("stderr", ""),
            }
            continue
        binary = Path(compile_result["binary"])
        if not binary.is_absolute():
            binary = ROOT / binary
        command = [str(binary), str(m), str(n), str(k), str(iterations)]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            results[precision] = {
                "status": "runtime_failed",
                "precision": precision,
                "command": command,
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-4000:],
            }
            continue
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            results[precision] = {
                "status": "runtime_json_failed",
                "precision": precision,
                "command": command,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
            continue
        payload["command"] = command
        results[precision] = payload
    return results


def build_case_manifest(case: BenchmarkCase) -> dict[str, Any]:
    tile_offsets: dict[str, int] = {}
    tile_bytes: dict[str, int] = {}
    tile_dtype_map: dict[str, str] = {}
    tile_ids: dict[str, dict[str, Any]] = {}
    tile_format_map: dict[str, dict[str, Any]] = {}
    scale_offsets: dict[str, int] = {}
    scale_bytes: dict[str, int] = {}
    scale_layout_map: dict[str, str] = {}
    backend_owner_map: dict[str, str] = {}
    tile_fallback_map: dict[str, dict[str, str]] = {}
    gpu_hot_tiles: list[str] = []

    offset = 0
    scale_offset = 0
    for tile_id in _iter_tile_ids():
        key = _stable_key(tile_id)
        elements = (int(tile_id["n_end"]) - int(tile_id["n_start"])) * OLMOE_LIKE_CONFIG["hidden_size"]
        weight_bytes = _packed_weight_bytes(elements, case.storage_bits)
        tile_n = int(tile_id["n_end"]) - int(tile_id["n_start"])
        per_tile_scale_bytes = _scale_bytes(tile_n, case.scale_block_size)

        tile_ids[key] = tile_id
        tile_offsets[key] = offset
        tile_bytes[key] = weight_bytes
        tile_dtype_map[key] = case.dtype
        backend_owner_map[key] = case.backend
        if case.residency_policy == "hot":
            gpu_hot_tiles.append(key)

        if case.backend == BACKEND_NAME:
            tile_format_map[key] = {
                "format": case.format_name,
                "storage_bits": case.storage_bits,
                "compute_dtype": case.format_name,
                "accum_dtype": "fp32",
                "layout_owner": BACKEND_NAME,
            }
            scale_layout_map[key] = case.scale_layout
            tile_fallback_map[key] = {
                "dtype": "bf16",
                "backend": "kt_fallback",
            }

        offset += weight_bytes
        if per_tile_scale_bytes:
            scale_offsets[key] = scale_offset
            scale_bytes[key] = per_tile_scale_bytes
            scale_offset += per_tile_scale_bytes

    manifest = {
        "schema_version": "tilepo_manifest_v1",
        "model": "olmoe_external_cuda_integration_fixture",
        "tile_ids": tile_ids,
        "tile_offsets": tile_offsets,
        "tile_bytes": tile_bytes,
        "tile_dtype_map": tile_dtype_map,
        "tile_format_map": tile_format_map,
        "scale_offsets": scale_offsets,
        "scale_bytes": scale_bytes,
        "scale_layout_map": scale_layout_map,
        "backend_owner_map": backend_owner_map,
        "tile_fallback_map": tile_fallback_map,
        "gpu_hot_tiles": gpu_hot_tiles,
        "fallback_chain": ["bf16", "kt_fallback"],
        "backend_priority": ["cuda", "kt_fallback"],
        "runtime_gates": ["capability", "offline_deterministic"],
        "mode": "verify",
        "deployment_mode": "balanced",
        "benchmark_case": case.name,
    }
    manifest["checksum"] = _checksum(manifest)
    return manifest


def summarize_case(
    case: BenchmarkCase,
    handles: list[TileHandle],
    dispatch_summary: dict[str, Any],
    *,
    compile_result: dict[str, Any] | None,
    runtime_result: dict[str, Any] | None,
) -> dict[str, Any]:
    weight_bytes = sum(handle.weight_bytes for handle in handles)
    scale_bytes = sum(max(0, handle.scale_bytes) for handle in handles)
    external_tiles = sum(1 for handle in handles if handle.backend == BACKEND_NAME)
    kt_fallback_tiles = sum(1 for handle in handles if handle.backend == "kt_fallback")
    dispatch_success_tiles = sum(1 for handle in handles if handle.dispatchable)
    unsupported_fallback_tiles = len(handles) - dispatch_success_tiles
    hot_tiles = sum(1 for handle in handles if handle.residency == "vram")
    cold_tiles = sum(1 for handle in handles if handle.residency == "dram")

    return {
        "name": case.name,
        "description": case.description,
        "dtype": case.dtype,
        "format": case.format_name,
        "scale_layout": case.scale_layout,
        "backend": case.backend,
        "residency_policy": case.residency_policy,
        "total_tiles": len(handles),
        "hot_tiles": hot_tiles,
        "cold_tiles": cold_tiles,
        "external_backend_tiles": external_tiles,
        "kt_fallback_tiles": kt_fallback_tiles,
        "dispatch_success_tiles": dispatch_success_tiles,
        "unsupported_fallback_tiles": unsupported_fallback_tiles,
        "estimated_weight_bytes": weight_bytes,
        "estimated_scale_bytes": scale_bytes,
        "estimated_payload_bytes": dispatch_summary["estimated_payload_bytes"],
        "dispatch_overhead_ms": dispatch_summary["dispatch_overhead_ms"],
        "dispatch_overhead_per_tile_us": dispatch_summary["dispatch_overhead_per_tile_us"],
        "interface_summary": dispatch_summary,
        "cuda_kernel_source": case.cuda_kernel_source,
        "c_abi_launch_function": case.c_abi_launch_function,
        "compile_status": compile_result["status"] if compile_result else "not_applicable",
        "cuda_runtime": runtime_result,
        "quality_claim": "external_developer_owned" if case.backend == BACKEND_NAME else "not_applicable",
    }


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# OLMoE External CUDA Integration Benchmark",
        "",
        summary["disclaimer"],
        "",
        "The runtime numbers below are for the small standalone reference kernels only. "
        "This benchmark does not claim GPU runtime speed for a production OLMoE deployment, model "
        "accuracy, or universal low-precision support.",
        "",
        "TileMEM provides tile IDs, format tags, scale metadata, tile handles, and fallback metadata. "
        "The FP8/FP6/FP4 CUDA code is intentionally shown as customer-owned kernels.",
        "",
        "## Fixture",
        "",
        f"- Layers: {summary['model']['layers']}",
        f"- Experts per layer: {summary['model']['experts_per_layer']}",
        f"- Active experts per layer in fixture: {summary['model']['active_experts_per_layer']}",
        f"- Hidden size: {summary['model']['hidden_size']}",
        f"- Intermediate size: {summary['model']['intermediate_size']}",
        f"- CUDA GEMM shape: M={summary['cuda_gemm_shape']['m']}, N={summary['cuda_gemm_shape']['n']}, K={summary['cuda_gemm_shape']['k']}",
        f"- Iterations: {summary['iterations']}",
        f"- GPU: {summary['cuda_environment'].get('gpu_name') or 'not detected'}",
        "",
        "## Results",
        "",
        "| Case | Backend | Tiles | Payload bytes | Reduction vs BF16 | CUDA avg ms | CUDA GFLOP/s | Dispatchable | Per tile dispatch us |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        runtime = case.get("cuda_runtime") or {}
        avg_ms = runtime.get("avg_ms")
        gflops = runtime.get("gflops")
        lines.append(
            "| {name} | {backend} | {tiles} | {payload} | {reduction:.2%} | {avg_ms} | {gflops} | {dispatchable} | {overhead:.6f} |".format(
                name=case["name"],
                backend=case["backend"],
                tiles=case["total_tiles"],
                payload=case["estimated_payload_bytes"],
                reduction=case["payload_reduction_vs_bf16"],
                avg_ms=f"{avg_ms:.6f}" if isinstance(avg_ms, (float, int)) else "n/a",
                gflops=f"{gflops:.3f}" if isinstance(gflops, (float, int)) else "n/a",
                dispatchable=case["dispatch_success_tiles"],
                overhead=case["dispatch_overhead_per_tile_us"],
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `bf16_fp32_baseline` is the uncompressed payload baseline through the KT fallback path.",
            "- FP8/FP6/FP4 cases register `olmoe_external_cuda` as a developer-owned backend.",
            "- Payload reductions are computed from tile weights plus scale metadata.",
            "- CUDA kernels use software dequantization to demonstrate real integration, not maximum tensor-core throughput.",
            "",
            "```json",
            json.dumps(summary, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _iter_tile_ids() -> list[dict[str, int | str]]:
    ids: list[dict[str, int | str]] = []
    for layer in range(OLMOE_LIKE_CONFIG["layers"]):
        for expert in range(OLMOE_LIKE_CONFIG["active_experts_per_layer"]):
            for projection_group in OLMOE_LIKE_CONFIG["projection_groups"]:
                for shard_id in range(OLMOE_LIKE_CONFIG["shards_per_projection"]):
                    n_start = shard_id * OLMOE_LIKE_CONFIG["tile_n"]
                    n_end = n_start + OLMOE_LIKE_CONFIG["tile_n"]
                    ids.append(
                        {
                            "layer": layer,
                            "expert": expert,
                            "projection_group": projection_group,
                            "shard_id": shard_id,
                            "n_start": n_start,
                            "n_end": n_end,
                        }
                    )
    return ids


def _stable_key(tile_id: dict[str, int | str]) -> str:
    return (
        f"L{tile_id['layer']}:E{tile_id['expert']}:{tile_id['projection_group']}:"
        f"S{tile_id['shard_id']}:N{tile_id['n_start']}-{tile_id['n_end']}"
    )


def _packed_weight_bytes(elements: int, storage_bits: int) -> int:
    return max(1, (elements * storage_bits + 7) // 8)


def _scale_bytes(tile_n: int, block_size: int) -> int:
    if block_size <= 0:
        return 0
    fp32_scale_bytes = 4
    return max(fp32_scale_bytes, ((tile_n + block_size - 1) // block_size) * fp32_scale_bytes)


def _checksum(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
