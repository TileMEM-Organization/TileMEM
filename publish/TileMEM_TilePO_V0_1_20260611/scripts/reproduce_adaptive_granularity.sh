#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXECUTE=0
BASE_PORT=35100
MODEL_DIR="/mnt/d/tilemem_runtime/models/OLMoE-1B-7B-0924-Instruct"
INIT_PATH="/mnt/d/tilemem_runtime/results/kt_tilemem_hotset_20260523/tilemem_hotset_counts.pt"
KT_ENV="tilemem-tilepo-ktransformers"
BENCH_TOOL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      EXECUTE=1
      shift
      ;;
    --base-port)
      BASE_PORT="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --init-expert-location)
      INIT_PATH="$2"
      shift 2
      ;;
    --kt-env)
      KT_ENV="$2"
      shift 2
      ;;
    --bench-tool)
      BENCH_TOOL="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

export TILEMEM_ADAPTIVE_EXECUTE="$EXECUTE"
export TILEMEM_ADAPTIVE_BASE_PORT="$BASE_PORT"
export TILEMEM_ADAPTIVE_MODEL_DIR="$MODEL_DIR"
export TILEMEM_ADAPTIVE_INIT_PATH="$INIT_PATH"
export TILEMEM_ADAPTIVE_KT_ENV="$KT_ENV"
export TILEMEM_ADAPTIVE_BENCH_TOOL="$BENCH_TOOL"

python3 - <<'PY'
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from tilepo.ablation import write_merged_manifest, write_tilepo_plan
from tilepo.dsl import DSLBlock, parse_tmem
from tilepo.sweep import run_sweep


ROOT = Path.cwd()
OUT_DIR = ROOT / "evidence" / "adaptive_granularity"
PLANS_DIR = OUT_DIR / "plans"
RUNS_DIR = OUT_DIR / "runs"
MERGED = OUT_DIR / "tilepo_adaptive_granularity_manifest.json"
BASE_PLAN = ROOT / "configs" / "tilepo_olmoe_bf16_only.tmem"
WORKLOADS = ["mixed", "long_context"]
EXPERTS = [6, 8, 10]
TILEPO_POLICIES = ["tilepo_coarse", "tilepo_fine", "tilepo_hybrid", "tilepo_adaptive"]
EXPECTED_ROWS = len(WORKLOADS) * len(EXPERTS) * (1 + len(TILEPO_POLICIES))


def main() -> int:
    execute = os.environ["TILEMEM_ADAPTIVE_EXECUTE"] == "1"
    base_port = int(os.environ["TILEMEM_ADAPTIVE_BASE_PORT"])
    model_dir = os.environ["TILEMEM_ADAPTIVE_MODEL_DIR"]
    init_path = os.environ["TILEMEM_ADAPTIVE_INIT_PATH"]
    kt_env = os.environ["TILEMEM_ADAPTIVE_KT_ENV"]
    bench_tool_text = os.environ["TILEMEM_ADAPTIVE_BENCH_TOOL"]
    bench_tool = Path(bench_tool_text) if bench_tool_text else None

    shutil.rmtree(PLANS_DIR, ignore_errors=True)
    shutil.rmtree(RUNS_DIR, ignore_errors=True)
    for stale in (
        MERGED,
        OUT_DIR / "tilepo_adaptive_granularity_summary.json",
        OUT_DIR / "tilepo_adaptive_granularity_report.md",
    ):
        stale.unlink(missing_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    plan_paths = {}
    for expert in EXPERTS:
        baseline_plan = _write_kt_baseline_plan(expert)
        plan_paths[("kt_expert", expert)] = baseline_plan
        for policy in TILEPO_POLICIES:
            plan_path = PLANS_DIR / f"{policy}_experts{expert}_throughput_async_on.tmem"
            write_tilepo_plan(
                BASE_PLAN,
                plan_path,
                expert_budget=expert,
                policy=policy,
                async_planning=True,
                adaptive_mode="throughput",
            )
            plan_paths[(policy, expert)] = plan_path

    manifest_paths = []
    blockers = []
    runs = []
    for expert in EXPERTS:
        runs.append(
            (
                "kt_expert",
                expert,
                plan_paths[("kt_expert", expert)],
                ["B"],
                "off",
                RUNS_DIR / f"kt_expert_experts{expert}_async_off",
            )
        )
        for policy in TILEPO_POLICIES:
            runs.append(
                (
                    policy,
                    expert,
                    plan_paths[(policy, expert)],
                    ["C"],
                    "on",
                    RUNS_DIR / f"{policy}_experts{expert}_async_on",
                )
            )

    plan_metadata: dict[tuple[str, int], dict] = {}
    for index, (policy, expert, plan_path, systems, async_mode, out_dir) in enumerate(runs):
        result = run_sweep(
            "verify",
            plan_path,
            out_dir,
            workloads=WORKLOADS,
            experts=[expert],
            repeats=1,
            require_real=execute,
            dry_run_commands=not execute,
            execute=execute,
            base_port=base_port + index * 100,
            model_dir=model_dir,
            init_path=init_path,
            kt_env=kt_env,
            bench_tool=bench_tool,
            systems=systems,
            request_count=5,
            warmup_request_count=1,
            output_tokens=4,
            skip_existing_success=True,
            c_mode="hook",
            ablation_policy=policy,
            async_planning_mode=async_mode,
        )
        manifest_path = Path(result["manifest_path"])
        manifest_paths.append(manifest_path)
        metadata = _compiled_plan_metadata(manifest_path)
        if metadata:
            plan_metadata[(policy, expert)] = metadata
        if result.get("blocked"):
            blockers.extend(str(item) for item in result.get("blockers", []))
        blockers.extend(_manifest_environment_blockers(manifest_path))

    write_merged_manifest(manifest_paths, MERGED)
    merged = json.loads(MERGED.read_text())
    rows = merged.get("runs", [])
    _attach_plan_metadata(rows, plan_metadata)
    blocked = bool(blockers)
    merged.update(
        {
            "schema_version": "tilepo_adaptive_granularity_manifest_v1",
            "adaptive_mode": "throughput",
            "matrix": {
                "workloads": WORKLOADS,
                "experts": EXPERTS,
                "policies": ["kt_expert", *TILEPO_POLICIES],
                "kt_async": "off",
                "tilepo_async": "on",
                "repeats": 1,
                "request_count": 5,
                "warmup_request_count": 1,
                "precision": "bf16_kt_native",
            },
            "blocked": blocked,
            "blockers": sorted(set(blockers)),
            "expected_result_rows": EXPECTED_ROWS,
            "actual_result_rows": len(rows),
            "evidence_level": "real" if execute and not blocked else ("blocked" if blocked else "dry_run_commands"),
        }
    )
    MERGED.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")

    if execute and not blocked and len(rows) == EXPECTED_ROWS:
        from tilepo.reporting.adaptive_granularity import generate_adaptive_granularity_report

        generate_adaptive_granularity_report(MERGED, OUT_DIR, require_real=True)
    elif blocked:
        (OUT_DIR / "tilepo_adaptive_granularity_report.md").write_text(
            "# TilePO Adaptive Granularity Report\n\n"
            "Gate: **BLOCKED**\n\n"
            "Real KT/SGLang execution did not start because the environment is incomplete.\n\n"
            "## Blockers\n\n"
            + "\n".join(f"- {item}" for item in sorted(set(blockers)))
            + "\n"
        )
    elif execute:
        (OUT_DIR / "tilepo_adaptive_granularity_report.md").write_text(
            "# TilePO Adaptive Granularity Report\n\n"
            "Gate: **FAIL**\n\n"
            f"Expected {EXPECTED_ROWS} real rows but found {len(rows)}.\n"
        )
    print(MERGED)
    if execute and (blocked or len(rows) != EXPECTED_ROWS):
        return 1
    return 0


def _write_kt_baseline_plan(expert_budget: int) -> Path:
    output = PLANS_DIR / f"kt_expert_experts{expert_budget}_async_off.tmem"
    plan = parse_tmem(BASE_PLAN.read_text())
    blocks = []
    for block in plan.blocks:
        values = dict(block.values)
        if block.kind == "workload":
            values["label"] = f"kt_expert_experts{expert_budget}"
        elif block.kind == "memory":
            values["experts_per_layer"] = int(expert_budget)
        elif block.kind == "schedule":
            values["async_planning"] = False
            values["deployment_mode"] = "safe"
        blocks.append(DSLBlock(block.kind, block.name, values, block.line))
    output.write_text(type(plan)(blocks).compiled_text())
    return output


def _compiled_plan_metadata(manifest_path: Path) -> dict:
    data = json.loads(manifest_path.read_text())
    compiled_manifest = data.get("compiled_manifest")
    if not compiled_manifest:
        return {}
    compiled_path = Path(compiled_manifest)
    if not compiled_path.exists():
        return {}
    compiled = json.loads(compiled_path.read_text())
    plan = compiled.get("tilepo_plan", {})
    if not plan:
        return {}
    tile_count = int(plan.get("tile_count", 0))
    dispatch = int(plan.get("estimated_dispatch_units", tile_count))
    return {"tile_count": tile_count, "estimated_dispatch_units": dispatch, "tilepo_plan": plan}


def _attach_plan_metadata(rows: list[dict], plan_metadata: dict[tuple[str, int], dict]) -> None:
    for row in rows:
        policy = str(row.get("tilepo_policy") or row.get("ablation_policy") or "")
        expert = int(row.get("experts_per_layer", 0))
        metadata = plan_metadata.get((policy, expert))
        if not metadata:
            continue
        row.setdefault("tile_count", metadata["tile_count"])
        row.setdefault("estimated_dispatch_units", metadata["estimated_dispatch_units"])
        if policy == "tilepo_adaptive":
            row.setdefault("tilepo_plan", metadata["tilepo_plan"])


def _manifest_environment_blockers(manifest_path: Path) -> list[str]:
    data = json.loads(manifest_path.read_text())
    env = data.get("environment", {})
    if not isinstance(env, dict) or env.get("ready", True):
        return []
    return [str(item) for item in env.get("blockers", [])]


if __name__ == "__main__":
    raise SystemExit(main())
PY
