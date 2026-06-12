"""TilePO MIR/DSL/runtime package."""

from .model_interface import (
    MODEL_SPEC_SCHEMA_VERSION,
    ModelAdapter,
    ModelSpec,
    build_mir_from_model_spec,
    model_spec_from_dict,
    model_spec_to_dict,
)

__version__ = "0.4.0"

__all__ = [
    "MODEL_SPEC_SCHEMA_VERSION",
    "ModelAdapter",
    "ModelSpec",
    "__version__",
    "build_mir_from_model_spec",
    "model_spec_from_dict",
    "model_spec_to_dict",
]
