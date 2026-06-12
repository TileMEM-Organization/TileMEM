from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import MIR_SCHEMA_VERSION, PUBLIC_MIR_INTERFACE, ModelIR, model_from_dict


def validate_mir_dict(data: dict[str, Any]) -> None:
    schema_version = str(data.get("schema_version", ""))
    if schema_version != MIR_SCHEMA_VERSION:
        raise ValueError(f"unsupported MIR schema_version: {schema_version}")
    interface = str(data.get("public_interface", PUBLIC_MIR_INTERFACE))
    if interface != PUBLIC_MIR_INTERFACE:
        raise ValueError(f"unsupported MIR public_interface: {interface}")
    model = model_from_dict(data)
    model.validate()


def load_mir(path: Path | str) -> ModelIR:
    data = json.loads(Path(path).read_text())
    validate_mir_dict(data)
    return model_from_dict(data)


def save_mir(model: ModelIR, path: Path | str) -> Path:
    model.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), indent=2, sort_keys=True) + "\n")
    return path
