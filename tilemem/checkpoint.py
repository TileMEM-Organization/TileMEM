from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable

from tilepo.mir import build_manifest, save_mir
from tilepo.model_interface import ModelSpec, model_spec_from_dict, model_spec_to_dict, build_mir_from_model_spec


SCHEMA_VERSION = "tilemem_checkpoint_artifact_v1"


@dataclass(frozen=True)
class MoETopology:
    family: str
    model_type: str
    layers: int
    experts_per_layer: int
    active_experts_per_token: int
    hidden_size: int
    intermediate_size: int
    source_fields: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "model_type": self.model_type,
            "layers": self.layers,
            "experts_per_layer": self.experts_per_layer,
            "active_experts_per_token": self.active_experts_per_token,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "source_fields": dict(self.source_fields),
        }


@dataclass(frozen=True)
class WeightMatchResult:
    family: str
    resolved: dict[str, dict[str, list[str]]]
    missing: list[str]
    unmatched: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "resolved": self.resolved,
            "missing": self.missing,
            "unmatched": self.unmatched,
        }


@dataclass(frozen=True)
class CheckpointArtifact:
    checkpoint_dir: Path
    out_dir: Path
    model_spec_path: Path
    mir_path: Path
    manifest_path: Path
    weight_map_path: Path
    tile_checkpoint_map_path: Path
    summary_path: Path
    materialized_weight_dir: Path | None
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "out_dir": str(self.out_dir),
            "model_spec_path": str(self.model_spec_path),
            "mir_path": str(self.mir_path),
            "manifest_path": str(self.manifest_path),
            "weight_map_path": str(self.weight_map_path),
            "tile_checkpoint_map_path": str(self.tile_checkpoint_map_path),
            "summary_path": str(self.summary_path),
            "materialized_weight_dir": None if self.materialized_weight_dir is None else str(self.materialized_weight_dir),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class ServingCommand:
    backend: str
    command: list[str]
    cwd: Path
    env: dict[str, str]
    execute_supported: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "command": list(self.command),
            "cwd": str(self.cwd),
            "env": dict(self.env),
            "execute_supported": self.execute_supported,
        }


@dataclass(frozen=True)
class ServingResult:
    backend: str
    status: str
    returncode: int
    command: list[str]
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "status": self.status,
            "returncode": self.returncode,
            "command": list(self.command),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def load_hf_config(path_or_config: Path | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_config, dict):
        return dict(path_or_config)
    path = Path(path_or_config)
    config_path = path / "config.json" if path.is_dir() else path
    if not config_path.exists():
        raise FileNotFoundError(f"HF config not found: {config_path}")
    data = json.loads(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"HF config must be a JSON object: {config_path}")
    return data


def infer_moe_topology(config_or_path: Path | str | dict[str, Any]) -> MoETopology:
    config = load_hf_config(config_or_path)
    model_type = str(config.get("model_type", config.get("architectures", ["unknown"])[0])).lower()
    family = _family_from_model_type(model_type)
    layers, layer_field = _required_int_from(config, ("num_hidden_layers", "n_layers", "num_layers"), "layers")
    experts, experts_field = _required_int_from(
        config,
        ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts", "num_experts_per_layer"),
        "experts_per_layer",
    )
    active, active_field = _optional_int_from(
        config,
        ("num_experts_per_tok", "num_experts_per_token", "moe_top_k", "top_k", "router_top_k"),
        default=min(2, experts),
    )
    hidden, hidden_field = _required_int_from(config, ("hidden_size", "d_model", "n_embd"), "hidden_size")
    intermediate, intermediate_field = _required_int_from(
        config,
        ("moe_intermediate_size", "intermediate_size", "ffn_hidden_size", "expert_intermediate_size"),
        "intermediate_size",
    )
    return MoETopology(
        family=family,
        model_type=model_type,
        layers=layers,
        experts_per_layer=experts,
        active_experts_per_token=active,
        hidden_size=hidden,
        intermediate_size=intermediate,
        source_fields={
            "layers": layer_field,
            "experts_per_layer": experts_field,
            "active_experts_per_token": active_field,
            "hidden_size": hidden_field,
            "intermediate_size": intermediate_field,
        },
    )


def model_spec_from_hf_config(
    config_or_path: Path | str | dict[str, Any],
    *,
    name: str | None = None,
    expert_budget: int | None = None,
    workload: str = "mixed",
    tile: dict[str, Any] | None = None,
    memory: dict[str, Any] | None = None,
    precision: dict[str, Any] | None = None,
    schedule: dict[str, Any] | None = None,
) -> ModelSpec:
    topology = infer_moe_topology(config_or_path)
    model_name = name or _safe_model_name(topology)
    return model_spec_from_dict(
        {
            "schema_version": "tilemem_model_spec_v1",
            "name": model_name,
            "layers": topology.layers,
            "experts_per_layer": topology.experts_per_layer,
            "hidden_size": topology.hidden_size,
            "intermediate_size": topology.intermediate_size,
            "expert_budget": expert_budget if expert_budget is not None else topology.active_experts_per_token,
            "workload": workload,
            "tile": {
                "hidden_tile": min(topology.hidden_size, 256),
                "intermediate_tile": min(topology.intermediate_size, 256),
                "shard_count": 4,
                "projection_groups": ["gate_up", "down"],
                **(tile or {}),
            },
            "memory": {
                "gpu_cache_budget_gib": 8.0,
                "cpu_cache_budget_gib": 64.0,
                **(memory or {}),
            },
            "precision": {
                "dtype_policy": "bf16",
                "allowed": ["bf16"],
                "calibration_required": False,
                **(precision or {}),
            },
            "schedule": {
                "mode": "verify",
                "deployment_mode": "balanced",
                "backend_priority": ["cuda", "tilelang", "kt_fallback"],
                "runtime_gates": ["locality", "correctness"],
                "prewarm_policy": "hotset",
                "miss_policy": "fallback",
                **(schedule or {}),
            },
        }
    )


def plan_from_hf_config(config_or_path: Path | str | dict[str, Any], **kwargs: Any) -> Any:
    from .sdk import plan

    return plan(model_spec_from_hf_config(config_or_path, **kwargs))


def checkpoint_weight_names(checkpoint_dir: Path | str) -> list[str]:
    weight_map = load_checkpoint_weight_map(checkpoint_dir)
    return sorted(weight_map)


def load_checkpoint_weight_map(checkpoint_dir: Path | str) -> dict[str, str]:
    checkpoint_dir = Path(checkpoint_dir)
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = checkpoint_dir / name
        if index_path.exists():
            data = json.loads(index_path.read_text())
            weight_map = data.get("weight_map", {})
            if not isinstance(weight_map, dict):
                raise ValueError(f"checkpoint index has no weight_map object: {index_path}")
            return {str(key): str(value) for key, value in weight_map.items()}
    direct = sorted(path.name for path in checkpoint_dir.glob("*.safetensors"))
    direct.extend(path.name for path in checkpoint_dir.glob("*.bin"))
    return {name: name for name in direct}


def match_checkpoint_weights(
    weight_names: Iterable[str],
    *,
    spec: ModelSpec,
    family: str = "auto",
    layers: Iterable[int] | None = None,
    experts: Iterable[int] | None = None,
) -> WeightMatchResult:
    selected_layers = sorted(set(int(item) for item in (layers if layers is not None else range(spec.layers))))
    selected_experts = sorted(
        set(int(item) for item in (experts if experts is not None else range(min(spec.expert_budget, spec.experts_per_layer))))
    )
    family = _normalize_family(family)
    resolved: dict[str, dict[str, list[str]]] = {}
    for layer in selected_layers:
        for expert in selected_experts:
            resolved[f"L{layer}:E{expert}"] = {"gate_up": [], "down": []}

    unmatched = []
    for name in sorted(str(item) for item in weight_names):
        parsed = _parse_weight_name(name, family)
        if parsed is None:
            unmatched.append(name)
            continue
        layer, expert, group = parsed
        key = f"L{layer}:E{expert}"
        if key not in resolved:
            unmatched.append(name)
            continue
        resolved[key][group].append(name)

    missing = []
    for key, groups in resolved.items():
        for group in ("gate_up", "down"):
            if not groups[group]:
                missing.append(f"{key}:{group}")
    return WeightMatchResult(family=family, resolved=resolved, missing=missing, unmatched=unmatched)


def export_checkpoint_artifact(
    checkpoint_dir: Path | str,
    *,
    out_dir: Path | str,
    name: str | None = None,
    layers: Iterable[int] | None = None,
    experts: Iterable[int] | None = None,
    materialize: bool = False,
) -> CheckpointArtifact:
    checkpoint_dir = Path(checkpoint_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_hf_config(checkpoint_dir)
    topology = infer_moe_topology(config)
    spec = model_spec_from_hf_config(config, name=name or checkpoint_dir.name or topology.family)
    mir = build_mir_from_model_spec(spec)
    manifest = build_manifest(mir)
    weight_map = load_checkpoint_weight_map(checkpoint_dir)
    match = match_checkpoint_weights(
        weight_map.keys(),
        spec=spec,
        family=topology.family,
        layers=layers,
        experts=experts,
    )

    model_spec_path = out_dir / "model_spec.json"
    mir_path = out_dir / "model.mir.json"
    manifest_path = out_dir / "model.manifest.json"
    weight_map_path = out_dir / "checkpoint_weight_map.json"
    tile_checkpoint_map_path = out_dir / "tile_checkpoint_map.json"
    summary_path = out_dir / "checkpoint_artifact_summary.json"

    model_spec_path.write_text(json.dumps(model_spec_to_dict(spec), indent=2, sort_keys=True) + "\n")
    save_mir(mir, mir_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    materialized_weight_dir = None
    materialized = {}
    if materialize:
        materialized_weight_dir = out_dir / "weights"
        materialized_weight_dir.mkdir(parents=True, exist_ok=True)
        for source_name in sorted(set(weight_map.values())):
            source = checkpoint_dir / source_name
            if not source.exists():
                raise FileNotFoundError(f"checkpoint shard not found for materialization: {source}")
            target = materialized_weight_dir / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            materialized[source_name] = str(target)

    mapping_payload = {
        "schema_version": "tilemem_checkpoint_weight_map_v1",
        "source_checkpoint": str(checkpoint_dir),
        "family": topology.family,
        "weight_map": weight_map,
        "resolved": match.resolved,
        "missing": match.missing,
        "unmatched": match.unmatched,
        "materialized": materialize,
        "materialized_shards": materialized,
    }
    weight_map_path.write_text(json.dumps(mapping_payload, indent=2, sort_keys=True) + "\n")
    tile_checkpoint_map = build_tile_checkpoint_map(
        manifest=manifest,
        match=match,
        weight_map=weight_map,
        materialized_shards=materialized,
        source_checkpoint=checkpoint_dir,
    )
    tile_checkpoint_map_path.write_text(json.dumps(tile_checkpoint_map, indent=2, sort_keys=True) + "\n")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {
            "path": str(checkpoint_dir),
            "family": topology.family,
            "model_type": topology.model_type,
            "topology": topology.to_dict(),
        },
        "materialized": materialize,
        "model_spec_path": str(model_spec_path),
        "mir_path": str(mir_path),
        "manifest_path": str(manifest_path),
        "weight_map_path": str(weight_map_path),
        "tile_checkpoint_map_path": str(tile_checkpoint_map_path),
        "weight_name_mapping": match.to_dict(),
        "runtime_weight_aliases": build_runtime_weight_aliases(
            family=topology.family,
            layers=layers if layers is not None else range(spec.layers),
            experts=experts if experts is not None else range(min(spec.expert_budget, spec.experts_per_layer)),
        ),
        "artifact_boundary": {
            "tilemem_owns": [
                "HF config topology inference",
                "TileMEM model spec, MIR, and manifest generation",
                "checkpoint weight-name mapping to tile projection groups",
                "serving backend command generation",
            ],
            "external_runtime_owns": [
                "exact tensor loading and repacking",
                "kernel-specific physical layout",
                "quality validation and calibration",
                "long-running server lifecycle",
            ],
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return CheckpointArtifact(
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        model_spec_path=model_spec_path,
        mir_path=mir_path,
        manifest_path=manifest_path,
        weight_map_path=weight_map_path,
        tile_checkpoint_map_path=tile_checkpoint_map_path,
        summary_path=summary_path,
        materialized_weight_dir=materialized_weight_dir,
        summary=summary,
    )


def build_tile_checkpoint_map(
    *,
    manifest: dict[str, Any],
    match: WeightMatchResult,
    weight_map: dict[str, str],
    materialized_shards: dict[str, str],
    source_checkpoint: Path,
) -> dict[str, Any]:
    tiles: dict[str, dict[str, Any]] = {}
    for stable_key, tile_id in sorted(manifest.get("tile_ids", {}).items()):
        layer = int(tile_id["layer"])
        expert = int(tile_id["expert"])
        projection_group = str(tile_id["projection_group"])
        owner_key = f"L{layer}:E{expert}"
        source_tensors = list(match.resolved.get(owner_key, {}).get(projection_group, []))
        shards = [weight_map[name] for name in source_tensors if name in weight_map]
        tiles[stable_key] = {
            "layer": layer,
            "expert": expert,
            "projection_group": projection_group,
            "shard_id": int(tile_id["shard_id"]),
            "n_start": int(tile_id["n_start"]),
            "n_end": int(tile_id["n_end"]),
            "tile_weight_offset": int(manifest.get("tile_offsets", {}).get(stable_key, 0)),
            "tile_weight_bytes": int(manifest.get("tile_bytes", {}).get(stable_key, 0)),
            "source_tensors": source_tensors,
            "source_shards": shards,
            "source_checkpoint": str(source_checkpoint),
            "materialized_shards": [materialized_shards[name] for name in shards if name in materialized_shards],
        }
    return {
        "schema_version": "tilemem_tile_checkpoint_map_v1",
        "model": manifest.get("model", ""),
        "source_checkpoint": str(source_checkpoint),
        "tiles": tiles,
    }


def build_runtime_weight_aliases(
    *,
    family: str,
    layers: Iterable[int],
    experts: Iterable[int],
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Return common KT/SGLang aliases for each TileMEM projection group.

    The aliases are metadata hints for integration code. They do not load or
    rewrite tensors; checkpoint provenance and exact runtime module ownership
    remain with the serving backend.
    """

    del family
    aliases: dict[str, dict[str, dict[str, list[str]]]] = {}
    for layer in sorted(set(int(item) for item in layers)):
        for expert in sorted(set(int(item) for item in experts)):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            mixtral_prefix = f"model.layers.{layer}.block_sparse_moe.experts.{expert}"
            aliases[f"L{layer}:E{expert}"] = {
                "kt": {
                    "gate_up": [f"{prefix}.w1.weight", f"{prefix}.w3.weight"],
                    "down": [f"{prefix}.w2.weight"],
                },
                "sglang": {
                    "gate_up": [
                        f"{prefix}.gate_proj.weight",
                        f"{prefix}.up_proj.weight",
                        f"{mixtral_prefix}.w1.weight",
                        f"{mixtral_prefix}.w3.weight",
                    ],
                    "down": [
                        f"{prefix}.down_proj.weight",
                        f"{mixtral_prefix}.w2.weight",
                    ],
                },
            }
    return aliases


def build_serving_command(
    *,
    checkpoint_dir: Path | str,
    backend: str,
    plan_path: Path | str,
    out_dir: Path | str | None = None,
    expert_budget: int | None = None,
    port: int = 34000,
    dry_run: bool = False,
    python_executable: str | None = None,
) -> ServingCommand:
    checkpoint_dir = Path(checkpoint_dir)
    plan_path = Path(plan_path)
    backend = _normalize_backend(backend)
    py = python_executable or sys.executable
    env = {"TILEMEM_MODEL_PATH": str(checkpoint_dir), "TILEMEM_PLAN": str(plan_path)}
    if backend == "sglang":
        command = [
            py,
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(checkpoint_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--kt-method",
            "BF16",
            "--kt-num-gpu-experts",
            str(expert_budget or 0),
            "--kt-expert-placement-strategy",
            "frequency",
        ]
        return ServingCommand(backend=backend, command=command, cwd=_repo_root(), env=env)
    if backend == "kt_native":
        if out_dir is None:
            out_dir = _repo_root() / "build" / "tilemem_checkpoint_serving"
        command = [
            py,
            str(_repo_root() / "tools" / "run_tilepo_sweep"),
            "--mode",
            "serve",
            "--c-mode",
            "kt_native",
            "--plan",
            str(plan_path),
            "--out-dir",
            str(out_dir),
            "--workloads",
            "mixed",
            "--experts",
            str(expert_budget or 0),
            "--systems",
            "B,C",
            "--repeats",
            "1",
            "--request-count",
            "1",
            "--warmup-request-count",
            "0",
            "--output-tokens",
            "4",
            "--model-dir",
            str(checkpoint_dir),
            "--min-linux-available-gib",
            "8",
        ]
        command.append("--dry-run-commands" if dry_run else "--execute")
        return ServingCommand(backend=backend, command=command, cwd=_repo_root(), env=env)
    raise ValueError(f"unsupported serving backend: {backend}")


def run_serving_backend(
    *,
    checkpoint_dir: Path | str,
    backend: str,
    plan_path: Path | str,
    out_dir: Path | str | None = None,
    expert_budget: int | None = None,
    execute: bool = False,
    timeout_sec: int = 60,
) -> ServingResult:
    command = build_serving_command(
        checkpoint_dir=checkpoint_dir,
        backend=backend,
        plan_path=plan_path,
        out_dir=out_dir,
        expert_budget=expert_budget,
        dry_run=not execute,
    )
    if not execute:
        return ServingResult(
            backend=command.backend,
            status="dry_run",
            returncode=0,
            command=command.command,
            stdout="",
            stderr="",
        )
    try:
        completed = subprocess.run(
            command.command,
            cwd=command.cwd,
            env={**os.environ, **command.env},
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return ServingResult(
            backend=command.backend,
            status="timeout",
            returncode=124,
            command=command.command,
            stdout=_decode_timeout_stream(exc.stdout),
            stderr=_decode_timeout_stream(exc.stderr),
        )
    return ServingResult(
        backend=command.backend,
        status="completed" if completed.returncode == 0 else "failed",
        returncode=completed.returncode,
        command=command.command,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _family_from_model_type(model_type: str) -> str:
    text = model_type.lower()
    if "olmoe" in text:
        return "olmoe"
    if "qwen" in text and "moe" in text:
        return "qwen_moe"
    if "mixtral" in text:
        return "mixtral"
    return "generic_moe"


def _decode_timeout_stream(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _normalize_family(family: str) -> str:
    family = family.lower()
    if family in {"auto", ""}:
        return "generic_moe"
    if family in {"qwen", "qwen2_moe", "qwen3_moe"}:
        return "qwen_moe"
    return family


def _normalize_backend(backend: str) -> str:
    backend = backend.lower().replace("-", "_")
    if backend in {"kt", "kt_native", "ktransformers"}:
        return "kt_native"
    if backend in {"sglang", "sglang_kt"}:
        return "sglang"
    return backend


def _required_int_from(config: dict[str, Any], keys: tuple[str, ...], label: str) -> tuple[int, str]:
    value, key = _lookup(config, keys)
    if value is None:
        raise ValueError(f"cannot infer MoE topology field: {label}; checked {', '.join(keys)}")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result, key


def _optional_int_from(config: dict[str, Any], keys: tuple[str, ...], *, default: int) -> tuple[int, str]:
    value, key = _lookup(config, keys)
    if value is None:
        return int(default), "default"
    result = int(value)
    if result <= 0:
        raise ValueError(f"{key} must be positive")
    return result, key


def _lookup(config: dict[str, Any], keys: tuple[str, ...]) -> tuple[Any | None, str]:
    for key in keys:
        if key in config:
            return config[key], key
    text_config = {str(key).lower(): value for key, value in config.items()}
    for key in keys:
        lowered = key.lower()
        if lowered in text_config:
            return text_config[lowered], lowered
    return None, ""


def _safe_model_name(topology: MoETopology) -> str:
    return f"{topology.family}_{topology.layers}l_{topology.experts_per_layer}e"


def _parse_weight_name(name: str, family: str) -> tuple[int, int, str] | None:
    parts = name.split(".")
    try:
        layer = _number_after(parts, "layers")
        expert = _number_after(parts, "experts")
    except ValueError:
        return None
    leaf = _projection_leaf(parts)
    if leaf is None:
        return None
    group = _projection_group(leaf, family)
    if group is None:
        return None
    return layer, expert, group


def _number_after(parts: list[str], marker: str) -> int:
    for index, part in enumerate(parts[:-1]):
        if part == marker:
            return int(parts[index + 1])
    raise ValueError(marker)


def _projection_leaf(parts: list[str]) -> str | None:
    for part in reversed(parts):
        if part not in {"weight", "bias"} and not part.endswith("weight"):
            return part
    return None


def _projection_group(leaf: str, family: str) -> str | None:
    del family
    if leaf in {"w1", "gate_proj", "gate", "gate_up"}:
        return "gate_up"
    if leaf in {"w3", "up_proj", "up"}:
        return "gate_up"
    if leaf in {"w2", "down_proj", "down"}:
        return "down"
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
