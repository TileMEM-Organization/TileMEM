"""TilePO MIR/DSL/runtime package."""

from .model_interface import (
    MODEL_SPEC_SCHEMA_VERSION,
    ModelAdapter,
    ModelSpec,
    build_mir_from_model_spec,
    model_spec_from_dict,
    model_spec_to_dict,
)
from .integration import (
    BackendCapability,
    BackendRegistry,
    ScaleLayout,
    TileFormat,
    TileHandle,
    backend_registry,
    benchmark_dispatch_plan,
    build_tile_handles,
    register_backend,
)

__version__ = "0.4.0"

__all__ = [
    "MODEL_SPEC_SCHEMA_VERSION",
    "BackendCapability",
    "BackendRegistry",
    "ModelAdapter",
    "ModelSpec",
    "ScaleLayout",
    "TileFormat",
    "TileHandle",
    "__version__",
    "backend_registry",
    "benchmark_dispatch_plan",
    "build_mir_from_model_spec",
    "build_tile_handles",
    "model_spec_from_dict",
    "model_spec_to_dict",
    "register_backend",
]
