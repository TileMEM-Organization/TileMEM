#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.olmoe_external_cuda_backend import (  # noqa: E402
    BACKEND_NAME,
    BENCHMARK_CASES,
    build_case_manifest,
    render_markdown_report,
    run_benchmark,
    register_external_cuda_backend,
)
from tilepo.integration import BackendRegistry, build_tile_handles  # noqa: E402


SCHEMA_VERSION = "tilemem_customer_integration_demo_v1"


def build_demo(*, out_dir: Path, iterations: int, m: int, n: int, k: int) -> dict[str, Any]:
    registry = BackendRegistry()
    capability = register_external_cuda_backend(registry)
    contracts: dict[str, dict[str, Any]] = {}
    manifest_evidence: dict[str, Any] | None = None

    for case in BENCHMARK_CASES:
        if case.backend != BACKEND_NAME:
            continue
        manifest = build_case_manifest(case)
        handles = build_tile_handles(manifest, registry=registry)
        handle = next(item for item in handles if item.backend == BACKEND_NAME)
        contracts[case.dtype] = {
            "stable_key": handle.stable_key,
            "tile_id": {
                "layer": handle.layer,
                "expert": handle.expert,
                "projection_group": handle.projection_group,
                "shard_id": handle.shard_id,
                "n_start": handle.n_start,
                "n_end": handle.n_end,
            },
            "dtype_tag": handle.dtype,
            "format": handle.format,
            "weight_offset": handle.weight_offset,
            "weight_bytes": handle.weight_bytes,
            "scale_offset": handle.scale_offset,
            "scale_bytes": handle.scale_bytes,
            "scale_layout": handle.scale_layout,
            "residency": handle.residency,
            "backend": handle.backend,
            "dispatchable": handle.dispatchable,
            "fallback": {
                "dtype": handle.fallback_dtype,
                "backend": handle.fallback_backend,
            },
            "kernel_source": case.cuda_kernel_source,
            "c_abi_launch_function": case.c_abi_launch_function,
        }
        if manifest_evidence is None:
            manifest_evidence = {
                "schema_version": manifest["schema_version"],
                "model": manifest["model"],
                "total_tiles": len(manifest["tile_ids"]),
                "checksum": manifest["checksum"],
                "required_maps": sorted(
                    [
                        "tile_ids",
                        "tile_offsets",
                        "tile_bytes",
                        "tile_dtype_map",
                        "tile_format_map",
                        "scale_offsets",
                        "scale_bytes",
                        "scale_layout_map",
                        "backend_owner_map",
                        "tile_fallback_map",
                    ]
                ),
            }

    benchmark_summary = run_benchmark(
        iterations=iterations,
        m=m,
        n=n,
        k=k,
        build_dir=out_dir / "cuda_build",
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "End-to-end customer integration demo: TileMEM emits tile metadata, "
            "dispatch handles, and fallback descriptors; customer-owned kernels "
            "consume those handles through FP8/FP6/FP4 CUDA samples."
        ),
        "tilemem_responsibilities": _tilemem_responsibilities(capability.to_dict(), contracts, manifest_evidence or {}),
        "external_developer_responsibilities": _external_developer_responsibilities(contracts),
        "kernel_contract_by_precision": contracts,
        "benchmark_summary": benchmark_summary,
        "artifacts": {
            "json": str(out_dir / "customer_integration_end_to_end.json"),
            "markdown": str(out_dir / "customer_integration_end_to_end.md"),
            "cuda_build_dir": str(out_dir / "cuda_build"),
        },
    }
    return payload


def render_demo_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# TileMEM Customer Integration End-to-End Example",
        "",
        payload["description"],
        "",
        "This demo uses real CUDA sample kernels, but it does not claim model quality, calibration quality, "
        "or production Tensor Core performance.",
        "",
        "## TileMEM owns",
        "",
    ]
    for name, item in payload["tilemem_responsibilities"].items():
        lines.append(f"- `{name}`: {item['summary']}")
    lines.extend(["", "## External developer owns", ""])
    for name, item in payload["external_developer_responsibilities"].items():
        lines.append(f"- `{name}`: {item['summary']}")
    lines.extend(
        [
            "",
            "## Kernel contract by precision",
            "",
            "| Precision | Tile | Format | Weight bytes | Scale bytes | Scale layout | Launcher |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for precision, contract in sorted(payload["kernel_contract_by_precision"].items()):
        lines.append(
            "| {precision} | `{tile}` | `{fmt}` | {weight} | {scale_bytes} | `{scale_layout}` | `{launcher}` |".format(
                precision=precision,
                tile=contract["stable_key"],
                fmt=contract["format"],
                weight=contract["weight_bytes"],
                scale_bytes=contract["scale_bytes"],
                scale_layout=contract["scale_layout"],
                launcher=contract["c_abi_launch_function"],
            )
        )
    lines.extend(
        [
            "",
            "## CUDA benchmark summary",
            "",
            render_markdown_report(payload["benchmark_summary"]),
        ]
    )
    return "\n".join(lines)


def _tilemem_responsibilities(
    capability: dict[str, Any],
    contracts: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    first_precision = sorted(contracts)[0]
    first = contracts[first_precision]
    return {
        "tile_splitting_and_id_management": {
            "owner": "TileMEM",
            "status": "provided",
            "summary": "Creates stable tile IDs from layer, expert, projection group, shard, and N range.",
            "evidence": {
                "sample_stable_key": first["stable_key"],
                "sample_tile_id": first["tile_id"],
                "total_tiles": manifest.get("total_tiles", 0),
            },
        },
        "tile_dtype_tag_format_metadata": {
            "owner": "TileMEM",
            "status": "provided",
            "summary": "Carries dtype tags and external format metadata for each tile.",
            "evidence": {
                precision: {
                    "dtype_tag": contract["dtype_tag"],
                    "format": contract["format"],
                }
                for precision, contract in sorted(contracts.items())
            },
        },
        "scale_metadata_address_size_layout": {
            "owner": "TileMEM",
            "status": "provided",
            "summary": "Reports scale metadata address, byte size, and layout string for external kernels.",
            "evidence": {
                precision: {
                    "scale_offset": contract["scale_offset"],
                    "scale_bytes": contract["scale_bytes"],
                    "scale_layout": contract["scale_layout"],
                }
                for precision, contract in sorted(contracts.items())
            },
        },
        "backend_capability_registration": {
            "owner": "TileMEM",
            "status": "provided",
            "summary": "Registers backend capabilities before tile handles are marked dispatchable.",
            "evidence": capability,
        },
        "manifest_generation": {
            "owner": "TileMEM",
            "status": "provided",
            "summary": "Generates the manifest maps consumed by runtime and external dispatch paths.",
            "evidence": manifest,
        },
        "runtime_dispatch_tile_handle": {
            "owner": "TileMEM",
            "status": "provided",
            "summary": "Builds runtime tile handles that contain offsets, format tags, scale metadata, and dispatch state.",
            "evidence": first,
        },
        "fallback_chain": {
            "owner": "TileMEM",
            "status": "provided",
            "summary": "Describes BF16/KT fallback when external backend dispatch is unsupported.",
            "evidence": {
                "manifest_fallback_chain": ["bf16", "kt_fallback"],
                "sample_tile_fallback": first["fallback"],
            },
        },
        "external_kernel_contract": {
            "owner": "TileMEM",
            "status": "provided",
            "summary": "Gives external kernels enough metadata to locate the tile, decode its format and scales, and dispatch.",
            "evidence": contracts,
        },
    }


def _external_developer_responsibilities(contracts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    kernel_sources = {
        precision: {
            "source": contract["kernel_source"],
            "launcher": contract["c_abi_launch_function"],
        }
        for precision, contract in sorted(contracts.items())
    }
    return {
        "concrete_fp8_fp6_fp4_kernels": {
            "owner": "External developer",
            "status": "customer_owned_sample_provided",
            "summary": "Provides the concrete FP8/FP6/FP4 GEMM kernels and C ABI launchers.",
            "evidence": kernel_sources,
        },
        "model_structure_adapter": {
            "owner": "External developer",
            "status": "customer_owned_contract_declared",
            "summary": "Maps model-specific MoE projections and expert layout into TileMEM tile groups.",
            "evidence": {
                "sample_adapter": "examples/olmoe_external_cuda_backend.py",
                "fixture": "OLMoE-like gate_up/down expert projection groups",
            },
        },
        "weight_quantization": {
            "owner": "External developer",
            "status": "customer_owned_contract_declared",
            "summary": "Chooses and implements how BF16/FP32 weights are quantized into FP8/FP6/FP4 packed storage.",
            "evidence": {
                "sample_location": "kernels/gemm_fp{4,6,8}.cu standalone benchmark packers",
                "tilemem_boundary": "TileMEM records dtype/format metadata but does not decide quantization numerics.",
            },
        },
        "calibration_method": {
            "owner": "External developer",
            "status": "customer_owned_contract_declared",
            "summary": "Owns calibration data and scale selection policy for each low-precision format.",
            "evidence": {
                "scale_metadata_consumed_by_tilemem": True,
                "calibration_algorithm_in_tilemem": False,
            },
        },
        "quality_evaluation": {
            "owner": "External developer",
            "status": "customer_owned_contract_declared",
            "summary": "Runs model quality gates and decides whether a low-precision backend can serve.",
            "evidence": {
                "tilemem_claims_quality": False,
                "fallback_available": {"dtype": "bf16", "backend": "kt_fallback"},
            },
        },
        "backend_specific_layout": {
            "owner": "External developer",
            "status": "customer_owned_contract_declared",
            "summary": "Defines the backend-specific physical layout and keeps it compatible with registered capabilities.",
            "evidence": {
                "registered_layouts": ["block_n32_fp32", "block_n64_fp32"],
                "layout_owner": BACKEND_NAME,
            },
        },
        "cuda_tilelang_triton_rocm_implementation": {
            "owner": "External developer",
            "status": "customer_owned_sample_provided",
            "summary": "Implements the concrete backend. This demo ships CUDA samples; TileLang/Triton/ROCm can implement the same contract.",
            "evidence": {
                "cuda_samples": kernel_sources,
                "other_backends": ["TileLang", "Triton", "ROCm"],
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_demo(out_dir=out_dir, iterations=args.iterations, m=args.m, n=args.n, k=args.k)
    json_path = out_dir / "customer_integration_end_to_end.json"
    markdown_path = out_dir / "customer_integration_end_to_end.md"
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    json_path.write_text(serialized)
    markdown_path.write_text(render_demo_markdown(payload))
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
