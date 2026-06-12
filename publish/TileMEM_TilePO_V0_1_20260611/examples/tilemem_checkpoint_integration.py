#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tilemem as TM  # noqa: E402


REQUIRED_TILEMEM_APIS = (
    "model_spec_from_hf_config",
    "plan_from_hf_config",
    "match_checkpoint_weights",
    "export_checkpoint_artifact",
    "build_serving_command",
    "run_serving_backend",
)


def write_synthetic_checkpoint(root: Path) -> tuple[Path, Path]:
    """Create a tiny local HF-style fixture with no downloads."""
    checkpoint_dir = root / "synthetic_hf_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    hf_config: dict[str, Any] = {
        "architectures": ["OLMoEForCausalLM"],
        "model_type": "olmoe",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "torch_dtype": "bfloat16",
        "vocab_size": 128,
    }
    config_path = checkpoint_dir / "config.json"
    config_path.write_text(json.dumps(hf_config, indent=2, sort_keys=True) + "\n")

    shard_name = "model-00001-of-00001.safetensors"
    (checkpoint_dir / shard_name).write_bytes(b"tilemem synthetic checkpoint shard\n")
    weight_map: dict[str, str] = {}
    for layer in range(hf_config["num_hidden_layers"]):
        for expert in range(hf_config["num_experts"]):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            weight_map[f"{prefix}.w1.weight"] = shard_name
            weight_map[f"{prefix}.w3.weight"] = shard_name
            weight_map[f"{prefix}.w2.weight"] = shard_name
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": len(weight_map)}, "weight_map": weight_map}, indent=2) + "\n"
    )

    return config_path, checkpoint_dir


def require_checkpoint_api() -> None:
    missing = [name for name in REQUIRED_TILEMEM_APIS if not hasattr(TM, name)]
    if missing:
        formatted = ", ".join(f"TM.{name}" for name in missing)
        raise RuntimeError(
            "This example requires the TileMEM checkpoint adapter API. "
            f"The current installed package is missing: {formatted}."
        )


def run_example(execute: bool) -> dict[str, Any]:
    require_checkpoint_api()

    with tempfile.TemporaryDirectory(prefix="tilemem_checkpoint_") as tmp:
        tmp_path = Path(tmp)
        config_path, checkpoint_dir = write_synthetic_checkpoint(tmp_path)
        artifact_dir = tmp_path / "tilemem_artifact"

        model_spec = TM.model_spec_from_hf_config(config_path)
        plan = TM.plan_from_hf_config(config_path)
        topology = TM.infer_moe_topology(config_path)
        matches = TM.match_checkpoint_weights(
            TM.checkpoint_weight_names(checkpoint_dir),
            spec=model_spec,
            family=topology.family,
            layers=[0],
            experts=[0],
        )
        artifact = TM.export_checkpoint_artifact(
            checkpoint_dir,
            out_dir=artifact_dir,
            layers=[0],
            experts=[0],
            materialize=False,
        )
        command = TM.build_serving_command(
            checkpoint_dir=checkpoint_dir,
            backend="sglang",
            plan_path=artifact.manifest_path,
            expert_budget=model_spec.expert_budget,
            port=8080,
        )
        serving_result = TM.run_serving_backend(
            checkpoint_dir=checkpoint_dir,
            backend="sglang",
            plan_path=artifact.manifest_path,
            expert_budget=model_spec.expert_budget,
            execute=execute,
        )

        return {
            "checkpoint_dir": str(checkpoint_dir),
            "config_path": str(config_path),
            "model_spec": _jsonable(model_spec),
            "plan": _jsonable(plan),
            "matched_weights": _jsonable(matches),
            "artifact": _jsonable(artifact),
            "serving_command": _jsonable(command),
            "serving_execute": execute,
            "serving_result": _jsonable(serving_result),
        }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demonstrate the TileMEM checkpoint adapter flow."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Launch the serving backend instead of the default dry run.",
    )
    args = parser.parse_args()

    try:
        payload = run_example(execute=args.execute)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
