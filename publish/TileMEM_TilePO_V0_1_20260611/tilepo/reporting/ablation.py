from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
from typing import Any


METRICS = ("tok_per_sec", "p95_ms", "p99_ms", "gpu_peak_gib", "cpu_ram_peak_gib")


class AblationReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class AblationReportResult:
    summary_path: Path
    markdown_path: Path
    summary: dict[str, Any]


def generate_ablation_report(
    manifest_path: Path | str,
    out_dir: Path | str,
    *,
    workloads: list[str],
    experts: list[int],
    policies: list[str],
    async_modes: list[str],
    repeats: int,
    require_real: bool = False,
) -> AblationReportResult:
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    data = json.loads(manifest_path.read_text())
    rows = [row for row in data.get("runs", []) if isinstance(row, dict)]
    failures: list[str] = []
    grouped: dict[tuple[str, int, str, str, str], list[dict[str, Any]]] = defaultdict(list)

    workload_set = set(workloads)
    expert_set = set(experts)
    policy_set = set(policies)
    async_set = set(async_modes)
    for index, row in enumerate(rows):
        row_failures = _validate_row(index, row, require_real=require_real)
        failures.extend(row_failures)
        workload = str(row.get("workload", ""))
        expert = _as_int(row.get("experts_per_layer"))
        policy = _row_policy(row)
        async_mode = _row_async(row)
        system = str(row.get("system", ""))
        if workload in workload_set and expert in expert_set and policy in policy_set and async_mode in async_set:
            grouped[(workload, expert, policy, async_mode, system)].append(row)

    groups = []
    for key in sorted(grouped):
        workload, expert, policy, async_mode, system = key
        group_rows = grouped[key]
        groups.append(
            {
                "workload": workload,
                "experts_per_layer": expert,
                "policy": policy,
                "async_planning": async_mode,
                "system": system,
                "repeats": len(group_rows),
                "metrics": {metric: _stats([_to_float(row.get(metric)) for row in group_rows]) for metric in METRICS},
            }
        )

    coverage_failures = _coverage_failures(
        grouped,
        workloads=workloads,
        experts=experts,
        policies=policies,
        async_modes=async_modes,
        repeats=repeats,
    )
    failures.extend(coverage_failures)
    summary = {
        "schema_version": "tilepo_ablation_report_v1",
        "source_manifest": str(manifest_path),
        "requested": {
            "workloads": workloads,
            "experts": experts,
            "policies": policies,
            "async_modes": async_modes,
            "repeats": repeats,
            "require_real": require_real,
        },
        "gate": {"status": "PASS" if not failures else "FAIL", "failures": failures},
        "groups": groups,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "tilepo_ablation_summary.json"
    markdown_path = out_dir / "tilepo_ablation_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_markdown(summary))
    if failures:
        raise AblationReportError("; ".join(failures))
    return AblationReportResult(summary_path, markdown_path, summary)


def _validate_row(index: int, row: dict[str, Any], *, require_real: bool) -> list[str]:
    failures: list[str] = []
    for key in ("system", "workload", "experts_per_layer", "repeat", "tok_per_sec", "p95_ms", "p99_ms"):
        if key not in row:
            failures.append(f"row {index} missing {key}")
    for metric in ("gpu_peak_gib", "cpu_ram_peak_gib"):
        if metric not in row:
            failures.append(f"row {index} missing {metric}")
    if not _row_policy(row):
        failures.append(f"row {index} missing tilepo_policy/ablation_policy")
    if not _row_async(row):
        failures.append(f"row {index} missing tilepo_async_planning/async_planning_mode")
    if require_real:
        if row.get("simulated") is not False or row.get("evidence_level") != "real":
            failures.append(f"row {index} is not real evidence")
        if row.get("status") != "success":
            failures.append(f"row {index} is not success: {row.get('status')}")
    return failures


def _coverage_failures(
    grouped: dict[tuple[str, int, str, str, str], list[dict[str, Any]]],
    *,
    workloads: list[str],
    experts: list[int],
    policies: list[str],
    async_modes: list[str],
    repeats: int,
) -> list[str]:
    failures: list[str] = []
    for workload in workloads:
        for expert in experts:
            baseline_key = (workload, expert, "kt_expert", "off", "B")
            if "kt_expert" in policies and "off" in async_modes:
                count = len(grouped.get(baseline_key, []))
                if count < repeats:
                    failures.append(f"missing baseline rows for {baseline_key}: {count}/{repeats}")
            for policy in policies:
                if policy == "kt_expert":
                    continue
                for async_mode in async_modes:
                    key = (workload, expert, policy, async_mode, "C")
                    count = len(grouped.get(key, []))
                    if count < repeats:
                        failures.append(f"missing TilePO rows for {key}: {count}/{repeats}")
    return failures


def _stats(values: list[float]) -> dict[str, float | int | None]:
    clean = [value for value in values if value is not None]
    if not clean:
        return {"median": None, "min": None, "max": None, "count": 0}
    return {"median": float(median(clean)), "min": min(clean), "max": max(clean), "count": len(clean)}


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TilePO V0.1 Ablation Report",
        "",
        f"Gate: **{summary['gate']['status']}**",
        "",
        "| Workload | Experts | Policy | Async | System | Repeats | tok/s | p95 ms | p99 ms | GPU peak GiB | CPU/DRAM peak GiB |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in summary["groups"]:
        metrics = group["metrics"]
        lines.append(
            "| {workload} | {experts} | {policy} | {async_mode} | {system} | {repeats} | "
            "{tok:.3f} | {p95:.1f} | {p99:.1f} | {gpu:.3f} | {cpu:.3f} |".format(
                workload=group["workload"],
                experts=group["experts_per_layer"],
                policy=group["policy"],
                async_mode=group["async_planning"],
                system=group["system"],
                repeats=group["repeats"],
                tok=_fmt(metrics["tok_per_sec"]["median"]),
                p95=_fmt(metrics["p95_ms"]["median"]),
                p99=_fmt(metrics["p99_ms"]["median"]),
                gpu=_fmt(metrics["gpu_peak_gib"]["median"]),
                cpu=_fmt(metrics["cpu_ram_peak_gib"]["median"]),
            )
        )
    if summary["gate"]["failures"]:
        lines.extend(["", "## Gate Failures", ""])
        lines.extend(f"- {failure}" for failure in summary["gate"]["failures"])
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _row_policy(row: dict[str, Any]) -> str:
    return str(row.get("tilepo_policy") or row.get("ablation_policy") or "")


def _row_async(row: dict[str, Any]) -> str:
    return str(row.get("tilepo_async_planning") or row.get("async_planning_mode") or "")


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _to_float(value: Any) -> float:
    return float(value)
