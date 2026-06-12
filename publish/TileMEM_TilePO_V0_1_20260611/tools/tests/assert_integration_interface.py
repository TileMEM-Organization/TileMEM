#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilepo.integration import (  # noqa: E402
    BackendCapability,
    ScaleLayout,
    TileFormat,
    backend_registry,
    benchmark_dispatch_plan,
    build_tile_handles,
    register_backend,
)
from tilepo.mir import (  # noqa: E402
    Backend,
    DeploymentMode,
    ModelIR,
    PrecisionIR,
    ResidencyIR,
    RouteIR,
    RuntimeMode,
    ScheduleIR,
    TileDType,
    TileIR,
    TileId,
    build_manifest,
)


def make_model() -> ModelIR:
    fp8_format = TileFormat(
        name="fp8_e4m3",
        storage_bits=8,
        compute_dtype="fp8_e4m3",
        accum_dtype="bf16",
        layout_owner="external_backend",
    )
    scale = ScaleLayout(
        required=True,
        granularity="block",
        block_size=32,
        scale_dtype="e8m0",
        axis="n",
        layout="block_n32_e8m0",
    )
    tiles = [
        TileIR(
            tile_id=TileId(0, 0, "gate_up", 0, 0, 64),
            dtype=TileDType.BF16,
            bytes=8192,
        ),
        TileIR(
            tile_id=TileId(0, 1, "gate_up", 0, 0, 64),
            dtype=TileDType.FP8,
            bytes=4096,
            scale_bytes=256,
            format=fp8_format,
            scale=scale,
            backend="olmoe_external_cuda",
            fallback_dtype="bf16",
            fallback_backend="kt_fallback",
        ),
    ]
    return ModelIR(
        name="olmoe_integration_toy",
        layers=1,
        experts_per_layer=2,
        hidden_size=64,
        intermediate_size=128,
        routes=[RouteIR("mixed", {"0": [0, 1]})],
        tiles=tiles,
        residency=ResidencyIR(
            gpu_cache_budget_gib=1.0,
            cpu_cache_budget_gib=8.0,
            gpu_hot_tiles=[tile.tile_id for tile in tiles],
            fallback_chain=[TileDType.BF16, "kt_fallback"],
        ),
        precision=PrecisionIR("backend_owned", [TileDType.BF16, TileDType.FP8], True),
        schedule=ScheduleIR(
            RuntimeMode.VERIFY,
            DeploymentMode.BALANCED,
            [Backend.CUDA, Backend.KT_FALLBACK],
            ["capability"],
            "hotset",
            "fallback",
        ),
    )


def test_backend_capability_registration_and_lookup() -> None:
    registry = backend_registry()
    registry.clear()
    capability = BackendCapability(
        name="olmoe_external_cuda",
        formats=["fp8_e4m3"],
        layouts=["block_n32_e8m0"],
        scale_granularities=["block"],
        projection_groups=["gate_up", "down"],
        runtime_entrypoint="examples.olmoe_external_cuda_backend:run_tile",
        owns_quantization=True,
        owns_calibration=True,
        owns_quality=True,
        hardware_targets=["cuda_sm90"],
        fallback_dtype="bf16",
    )
    register_backend(capability, registry=registry)
    assert registry.get("olmoe_external_cuda") == capability
    assert capability.supports_format("fp8_e4m3")
    assert capability.supports_layout("block_n32_e8m0")


def test_manifest_carries_external_backend_metadata() -> None:
    model = make_model()
    manifest = build_manifest(model)
    fp8_key = model.tiles[1].tile_id.stable_key()
    assert manifest["tile_dtype_map"][fp8_key] == "fp8"
    assert manifest["tile_format_map"][fp8_key]["format"] == "fp8_e4m3"
    assert manifest["scale_offsets"][fp8_key] == 0
    assert manifest["scale_bytes"][fp8_key] == 256
    assert manifest["scale_layout_map"][fp8_key] == "block_n32_e8m0"
    assert manifest["backend_owner_map"][fp8_key] == "olmoe_external_cuda"
    assert manifest["tile_fallback_map"][fp8_key] == {
        "dtype": "bf16",
        "backend": "kt_fallback",
    }


def test_external_low_precision_dtype_tags_are_public_mir_values() -> None:
    assert TileDType.FP8.value == "fp8"
    assert TileDType.FP6.value == "fp6"
    assert TileDType.FP4.value == "fp4"
    assert TileDType.MXFP4.value == "mxfp4"


def test_tile_handles_are_consumable_by_external_kernel() -> None:
    registry = backend_registry()
    registry.clear()
    register_backend(
        BackendCapability(
            name="olmoe_external_cuda",
            formats=["fp8_e4m3"],
            layouts=["block_n32_e8m0"],
            scale_granularities=["block"],
            projection_groups=["gate_up"],
            runtime_entrypoint="examples.olmoe_external_cuda_backend:run_tile",
            owns_quantization=True,
            owns_calibration=True,
            owns_quality=True,
            hardware_targets=["cuda_sm90"],
            fallback_dtype="bf16",
        ),
        registry=registry,
    )
    model = make_model()
    handles = build_tile_handles(build_manifest(model), registry=registry)
    assert len(handles) == 2
    fp8 = [handle for handle in handles if handle.backend == "olmoe_external_cuda"][0]
    assert fp8.stable_key == model.tiles[1].tile_id.stable_key()
    assert fp8.dtype == "fp8"
    assert fp8.format == "fp8_e4m3"
    assert fp8.scale_offset == 0
    assert fp8.scale_bytes == 256
    assert fp8.scale_layout == "block_n32_e8m0"
    assert fp8.fallback_dtype == "bf16"
    assert fp8.fallback_backend == "kt_fallback"
    assert fp8.dispatchable is True


def test_unsupported_backend_falls_back_in_dispatch_plan() -> None:
    registry = backend_registry()
    registry.clear()
    model = make_model()
    handles = build_tile_handles(build_manifest(model), registry=registry)
    fp8 = [handle for handle in handles if handle.backend == "olmoe_external_cuda"][0]
    assert fp8.dispatchable is False
    assert fp8.fallback_backend == "kt_fallback"
    summary = benchmark_dispatch_plan(handles, iterations=5)
    assert summary["tiles"] == 2
    assert summary["dispatchable_tiles"] == 1
    assert summary["fallback_tiles"] == 1
    assert summary["iterations"] == 5
    assert summary["estimated_payload_bytes"] == 12_544


def main() -> None:
    test_backend_capability_registration_and_lookup()
    test_manifest_carries_external_backend_metadata()
    test_external_low_precision_dtype_tags_are_public_mir_values()
    test_tile_handles_are_consumable_by_external_kernel()
    test_unsupported_backend_falls_back_in_dispatch_plan()
    print("Integration interface tests passed")


if __name__ == "__main__":
    main()
