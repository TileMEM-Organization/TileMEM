from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any


@dataclass(frozen=True)
class TileFormat:
    name: str
    storage_bits: int
    compute_dtype: str
    accum_dtype: str
    layout_owner: str

    def validate(self) -> None:
        if not self.name:
            raise ValueError("tile format name is required")
        if self.storage_bits <= 0:
            raise ValueError("tile format storage_bits must be positive")
        if not self.compute_dtype:
            raise ValueError("tile format compute_dtype is required")
        if not self.accum_dtype:
            raise ValueError("tile format accum_dtype is required")
        if not self.layout_owner:
            raise ValueError("tile format layout_owner is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "format": self.name,
            "storage_bits": self.storage_bits,
            "compute_dtype": self.compute_dtype,
            "accum_dtype": self.accum_dtype,
            "layout_owner": self.layout_owner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TileFormat":
        return cls(
            name=str(data.get("format", data.get("name", ""))),
            storage_bits=int(data["storage_bits"]),
            compute_dtype=str(data["compute_dtype"]),
            accum_dtype=str(data["accum_dtype"]),
            layout_owner=str(data["layout_owner"]),
        )


@dataclass(frozen=True)
class ScaleLayout:
    required: bool
    granularity: str
    block_size: int
    scale_dtype: str
    axis: str
    layout: str

    def validate(self) -> None:
        if not self.granularity:
            raise ValueError("scale granularity is required")
        if self.required and self.block_size <= 0:
            raise ValueError("required scale layout needs positive block_size")
        if self.required and not self.scale_dtype:
            raise ValueError("required scale layout needs scale_dtype")
        if not self.axis:
            raise ValueError("scale axis is required")
        if not self.layout:
            raise ValueError("scale layout is required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "required": self.required,
            "granularity": self.granularity,
            "block_size": self.block_size,
            "scale_dtype": self.scale_dtype,
            "axis": self.axis,
            "layout": self.layout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScaleLayout":
        return cls(
            required=bool(data["required"]),
            granularity=str(data["granularity"]),
            block_size=int(data.get("block_size", 0)),
            scale_dtype=str(data.get("scale_dtype", "")),
            axis=str(data["axis"]),
            layout=str(data["layout"]),
        )


@dataclass(frozen=True)
class BackendCapability:
    name: str
    formats: list[str]
    layouts: list[str]
    scale_granularities: list[str]
    projection_groups: list[str]
    runtime_entrypoint: str
    owns_quantization: bool
    owns_calibration: bool
    owns_quality: bool
    hardware_targets: list[str]
    fallback_dtype: str = "bf16"

    def validate(self) -> None:
        if not self.name:
            raise ValueError("backend name is required")
        if not self.formats:
            raise ValueError("backend formats must not be empty")
        if not self.layouts:
            raise ValueError("backend layouts must not be empty")
        if not self.projection_groups:
            raise ValueError("backend projection_groups must not be empty")
        if not self.runtime_entrypoint:
            raise ValueError("backend runtime_entrypoint is required")

    def supports_format(self, format_name: str) -> bool:
        return format_name in self.formats

    def supports_layout(self, layout: str) -> bool:
        return layout in self.layouts

    def supports_scale_granularity(self, granularity: str) -> bool:
        return granularity in self.scale_granularities

    def supports_projection_group(self, projection_group: str) -> bool:
        return projection_group in self.projection_groups

    def supports_handle(self, handle: "TileHandle") -> bool:
        return (
            self.supports_format(handle.format)
            and self.supports_layout(handle.scale_layout)
            and self.supports_projection_group(handle.projection_group)
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "name": self.name,
            "formats": list(self.formats),
            "layouts": list(self.layouts),
            "scale_granularities": list(self.scale_granularities),
            "projection_groups": list(self.projection_groups),
            "runtime_entrypoint": self.runtime_entrypoint,
            "owns_quantization": self.owns_quantization,
            "owns_calibration": self.owns_calibration,
            "owns_quality": self.owns_quality,
            "hardware_targets": list(self.hardware_targets),
            "fallback_dtype": self.fallback_dtype,
        }


class BackendRegistry:
    def __init__(self) -> None:
        self._capabilities: dict[str, BackendCapability] = {}

    def clear(self) -> None:
        self._capabilities.clear()

    def register(self, capability: BackendCapability) -> BackendCapability:
        capability.validate()
        self._capabilities[capability.name] = capability
        return capability

    def get(self, name: str) -> BackendCapability | None:
        return self._capabilities.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {name: capability.to_dict() for name, capability in sorted(self._capabilities.items())}


@dataclass(frozen=True)
class TileHandle:
    stable_key: str
    layer: int
    expert: int
    projection_group: str
    shard_id: int
    n_start: int
    n_end: int
    dtype: str
    format: str
    weight_offset: int
    weight_bytes: int
    scale_offset: int
    scale_bytes: int
    scale_layout: str
    residency: str
    backend: str
    fallback_dtype: str
    fallback_backend: str
    dispatchable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_key": self.stable_key,
            "layer": self.layer,
            "expert": self.expert,
            "projection_group": self.projection_group,
            "shard_id": self.shard_id,
            "n_start": self.n_start,
            "n_end": self.n_end,
            "dtype": self.dtype,
            "format": self.format,
            "weight_offset": self.weight_offset,
            "weight_bytes": self.weight_bytes,
            "scale_offset": self.scale_offset,
            "scale_bytes": self.scale_bytes,
            "scale_layout": self.scale_layout,
            "residency": self.residency,
            "backend": self.backend,
            "fallback_dtype": self.fallback_dtype,
            "fallback_backend": self.fallback_backend,
            "dispatchable": self.dispatchable,
        }


_GLOBAL_REGISTRY = BackendRegistry()


def backend_registry() -> BackendRegistry:
    return _GLOBAL_REGISTRY


def register_backend(capability: BackendCapability, *, registry: BackendRegistry | None = None) -> BackendCapability:
    return (registry or _GLOBAL_REGISTRY).register(capability)


def build_tile_handles(manifest: dict[str, Any], *, registry: BackendRegistry | None = None) -> list[TileHandle]:
    registry = registry or _GLOBAL_REGISTRY
    handles: list[TileHandle] = []
    hot_tiles = set(manifest.get("gpu_hot_tiles", []))
    tile_ids = manifest.get("tile_ids", {})
    tile_format_map = manifest.get("tile_format_map", {})
    scale_layout_map = manifest.get("scale_layout_map", {})
    backend_owner_map = manifest.get("backend_owner_map", {})
    fallback_map = manifest.get("tile_fallback_map", {})
    for key, weight_offset in sorted(manifest.get("tile_offsets", {}).items()):
        tile_id = tile_ids.get(key, {})
        tile_format = tile_format_map.get(key, {})
        fallback = fallback_map.get(key, {})
        dtype = str(manifest.get("tile_dtype_map", {}).get(key, "bf16"))
        backend = str(backend_owner_map.get(key, _default_backend(dtype)))
        scale_layout = str(scale_layout_map.get(key, "none"))
        handle = TileHandle(
            stable_key=key,
            layer=int(tile_id.get("layer", -1)),
            expert=int(tile_id.get("expert", -1)),
            projection_group=str(tile_id.get("projection_group", "")),
            shard_id=int(tile_id.get("shard_id", -1)),
            n_start=int(tile_id.get("n_start", 0)),
            n_end=int(tile_id.get("n_end", 0)),
            dtype=dtype,
            format=str(tile_format.get("format", dtype)),
            weight_offset=int(weight_offset),
            weight_bytes=int(manifest.get("tile_bytes", {}).get(key, 0)),
            scale_offset=int(manifest.get("scale_offsets", {}).get(key, -1)),
            scale_bytes=int(manifest.get("scale_bytes", {}).get(key, 0)),
            scale_layout=scale_layout,
            residency="vram" if key in hot_tiles else "dram",
            backend=backend,
            fallback_dtype=str(fallback.get("dtype", "bf16")),
            fallback_backend=str(fallback.get("backend", "kt_fallback")),
            dispatchable=False,
        )
        capability = registry.get(backend)
        dispatchable = backend == "kt_fallback" or (capability is not None and capability.supports_handle(handle))
        handles.append(_replace_dispatchable(handle, dispatchable))
    return handles


def benchmark_dispatch_plan(handles: list[TileHandle], *, iterations: int = 1) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    started = time.perf_counter()
    dispatchable = sum(1 for handle in handles if handle.dispatchable)
    fallback = len(handles) - dispatchable
    payload_bytes = sum(handle.weight_bytes + max(0, handle.scale_bytes) for handle in handles)
    for _ in range(iterations):
        for handle in handles:
            _ = handle.stable_key if handle.dispatchable else handle.fallback_backend
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "tiles": len(handles),
        "dispatchable_tiles": dispatchable,
        "fallback_tiles": fallback,
        "iterations": iterations,
        "estimated_payload_bytes": payload_bytes,
        "dispatch_overhead_ms": elapsed_ms,
        "dispatch_overhead_per_tile_us": (elapsed_ms * 1000.0) / max(1, len(handles) * iterations),
    }


def _replace_dispatchable(handle: TileHandle, dispatchable: bool) -> TileHandle:
    return TileHandle(
        stable_key=handle.stable_key,
        layer=handle.layer,
        expert=handle.expert,
        projection_group=handle.projection_group,
        shard_id=handle.shard_id,
        n_start=handle.n_start,
        n_end=handle.n_end,
        dtype=handle.dtype,
        format=handle.format,
        weight_offset=handle.weight_offset,
        weight_bytes=handle.weight_bytes,
        scale_offset=handle.scale_offset,
        scale_bytes=handle.scale_bytes,
        scale_layout=handle.scale_layout,
        residency=handle.residency,
        backend=handle.backend,
        fallback_dtype=handle.fallback_dtype,
        fallback_backend=handle.fallback_backend,
        dispatchable=dispatchable,
    )


def _default_backend(dtype: str) -> str:
    if dtype == "bf16":
        return "kt_fallback"
    return "external"
