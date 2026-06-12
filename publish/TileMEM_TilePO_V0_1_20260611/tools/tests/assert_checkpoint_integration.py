#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


OLMOE_CONFIG = {
    "model_type": "olmoe",
    "num_hidden_layers": 16,
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "hidden_size": 2048,
    "intermediate_size": 8192,
}

QWEN_MOE_CONFIG = {
    "model_type": "qwen2_moe",
    "num_hidden_layers": 24,
    "num_experts": 60,
    "num_experts_per_tok": 4,
    "hidden_size": 2048,
    "moe_intermediate_size": 1408,
}

MIXTRAL_CONFIG = {
    "model_type": "mixtral",
    "num_hidden_layers": 32,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "hidden_size": 4096,
    "intermediate_size": 14336,
}


def test_hf_configs_infer_model_specs_for_olmoe_qwen_and_mixtral() -> None:
    import tilemem as TM

    cases = [
        (OLMOE_CONFIG, "olmoe_fixture", 16, 64, 8, 2048, 8192),
        (QWEN_MOE_CONFIG, "qwen_moe_fixture", 24, 60, 4, 2048, 1408),
        (MIXTRAL_CONFIG, "mixtral_fixture", 32, 8, 2, 4096, 14336),
    ]
    for config, name, layers, experts, active, hidden, intermediate in cases:
        spec = TM.model_spec_from_hf_config(config, name=name)
        assert spec.name == name
        assert spec.layers == layers
        assert spec.experts_per_layer == experts
        assert spec.expert_budget == active
        assert spec.hidden_size == hidden
        assert spec.intermediate_size == intermediate
        compiled = TM.plan_from_hf_config(config, name=name)
        assert compiled.mir.name == name
        assert compiled.handles


def test_hf_config_can_be_loaded_from_checkpoint_directory() -> None:
    import tilemem as TM

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp)
        (checkpoint_dir / "config.json").write_text(json.dumps(MIXTRAL_CONFIG) + "\n")
        loaded = TM.load_hf_config(checkpoint_dir)
        assert loaded["model_type"] == "mixtral"
        topology = TM.infer_moe_topology(loaded)
        assert topology.family == "mixtral"
        assert topology.layers == 32
        assert topology.experts_per_layer == 8


def test_checkpoint_weight_names_are_matched_to_tilemem_projection_groups() -> None:
    import tilemem as TM

    spec = TM.model_spec_from_hf_config(MIXTRAL_CONFIG, name="mixtral_fixture")
    weight_names = [
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "model.layers.0.block_sparse_moe.experts.0.w3.weight",
        "model.layers.0.block_sparse_moe.experts.0.w2.weight",
    ]
    result = TM.match_checkpoint_weights(
        weight_names,
        spec=spec,
        family="mixtral",
        layers=[0],
        experts=[0],
    )
    assert result.missing == []
    resolved = result.resolved["L0:E0"]
    assert resolved["gate_up"] == [
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "model.layers.0.block_sparse_moe.experts.0.w3.weight",
    ]
    assert resolved["down"] == ["model.layers.0.block_sparse_moe.experts.0.w2.weight"]

    direct = TM.match_checkpoint_weights(
        ["layers.0.experts.0.gate_up.weight", "layers.0.experts.0.down.weight"],
        spec=spec,
        family="generic_moe",
        layers=[0],
        experts=[0],
    )
    assert direct.missing == []
    assert direct.resolved["L0:E0"]["gate_up"] == ["layers.0.experts.0.gate_up.weight"]


def test_checkpoint_artifact_exports_manifest_weight_map_and_serving_commands() -> None:
    import tilemem as TM

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checkpoint_dir = root / "checkpoint"
        out_dir = root / "artifact"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text(json.dumps(OLMOE_CONFIG, indent=2) + "\n")
        weight_map = {
            "model.layers.0.mlp.experts.0.w1.weight": "model-00001-of-00001.safetensors",
            "model.layers.0.mlp.experts.0.w3.weight": "model-00001-of-00001.safetensors",
            "model.layers.0.mlp.experts.0.w2.weight": "model-00001-of-00001.safetensors",
        }
        (checkpoint_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 123}, "weight_map": weight_map}, indent=2) + "\n"
        )

        artifact = TM.export_checkpoint_artifact(
            checkpoint_dir,
            out_dir=out_dir,
            layers=[0],
            experts=[0],
            materialize=False,
        )
        assert artifact.summary_path.exists()
        assert artifact.manifest_path.exists()
        assert artifact.tile_checkpoint_map_path.exists()
        summary = json.loads(artifact.summary_path.read_text())
        tile_map = json.loads(artifact.tile_checkpoint_map_path.read_text())
        assert summary["schema_version"] == "tilemem_checkpoint_artifact_v1"
        assert summary["checkpoint"]["family"] == "olmoe"
        assert summary["materialized"] is False
        assert summary["tile_checkpoint_map_path"] == str(artifact.tile_checkpoint_map_path)
        assert summary["weight_name_mapping"]["missing"] == []
        assert "L0:E0" in summary["weight_name_mapping"]["resolved"]
        assert tile_map["schema_version"] == "tilemem_tile_checkpoint_map_v1"
        first_tile = next(iter(tile_map["tiles"].values()))
        assert first_tile["layer"] == 0
        assert first_tile["expert"] == 0
        assert first_tile["projection_group"] in {"gate_up", "down"}
        assert first_tile["source_tensors"]
        assert first_tile["n_end"] > first_tile["n_start"]
        aliases = summary["runtime_weight_aliases"]["L0:E0"]
        assert "model.layers.0.mlp.experts.0.w1.weight" in aliases["kt"]["gate_up"]
        assert "model.layers.0.mlp.experts.0.gate_proj.weight" in aliases["sglang"]["gate_up"]
        assert "model.layers.0.mlp.experts.0.down_proj.weight" in aliases["sglang"]["down"]
        direct_aliases = TM.build_runtime_weight_aliases(family="olmoe", layers=[0], experts=[0])
        assert direct_aliases["L0:E0"] == aliases

        sglang = TM.build_serving_command(
            checkpoint_dir=checkpoint_dir,
            backend="sglang",
            plan_path=artifact.manifest_path,
            expert_budget=8,
        )
        assert sglang.backend == "sglang"
        assert "--model-path" in sglang.command
        assert str(checkpoint_dir) in sglang.command
        assert "--kt-num-gpu-experts" in sglang.command
        assert "--tilemem-plan" not in sglang.command
        assert sglang.env["TILEMEM_PLAN"] == str(artifact.manifest_path)

        kt = TM.run_serving_backend(
            checkpoint_dir=checkpoint_dir,
            backend="kt_native",
            plan_path=artifact.manifest_path,
            out_dir=out_dir / "serve",
            execute=False,
        )
        assert kt.status == "dry_run"
        assert kt.returncode == 0
        assert "--dry-run-commands" in kt.command


def test_checkpoint_prepare_cli_runs_end_to_end_dry_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checkpoint_dir = root / "qwen"
        out_dir = root / "out"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text(json.dumps(QWEN_MOE_CONFIG, indent=2) + "\n")
        weight_map = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": "model.safetensors",
            "model.layers.0.mlp.experts.0.up_proj.weight": "model.safetensors",
            "model.layers.0.mlp.experts.0.down_proj.weight": "model.safetensors",
        }
        (checkpoint_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 321}, "weight_map": weight_map}, indent=2) + "\n"
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "tilemem_checkpoint_prepare"),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--out-dir",
                str(out_dir),
                "--backend",
                "sglang",
                "--layers",
                "0",
                "--experts",
                "0",
                "--dry-run",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        assert payload["schema_version"] == "tilemem_checkpoint_prepare_v1"
        assert payload["artifact"]["summary_path"].endswith("checkpoint_artifact_summary.json")
        assert payload["serving"]["status"] == "dry_run"
        assert payload["serving"]["backend"] == "sglang"


def test_run_serving_backend_reports_execute_timeout_as_result() -> None:
    import tilemem as TM

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checkpoint_dir = root / "checkpoint"
        checkpoint_dir.mkdir()
        manifest = root / "manifest.json"
        manifest.write_text("{}\n")
        result = TM.run_serving_backend(
            checkpoint_dir=checkpoint_dir,
            backend="sglang",
            plan_path=manifest,
            execute=True,
            timeout_sec=0,
        )
        assert result.status == "timeout"
        assert result.returncode == 124


def main() -> None:
    test_hf_configs_infer_model_specs_for_olmoe_qwen_and_mixtral()
    test_hf_config_can_be_loaded_from_checkpoint_directory()
    test_checkpoint_weight_names_are_matched_to_tilemem_projection_groups()
    test_checkpoint_artifact_exports_manifest_weight_map_and_serving_commands()
    test_checkpoint_prepare_cli_runs_end_to_end_dry_run()
    test_run_serving_backend_reports_execute_timeout_as_result()
    print("TileMEM checkpoint integration tests passed")


if __name__ == "__main__":
    main()
