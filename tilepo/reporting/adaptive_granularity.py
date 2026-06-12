from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


WORKLOADS = ("mixed", "long_context")
EXPERTS = (6, 8, 10)
FIXED_POLICIES = ("tilepo_coarse", "tilepo_fine", "tilepo_hybrid")
POLICIES = ("kt_expert", *FIXED_POLICIES, "tilepo_adaptive")
EXPECTED_ROWS = len(WORKLOADS) * len(EXPERTS) * len(POLICIES)
CORE_METRICS = ("tok_per_sec", "p95_ms", "p99_ms", "gpu_peak_gib", "cpu_ram_peak_gib")
OPTIONAL_METRICS = ("tile_count", "estimated_dispatch_units")
METRICS = (*CORE_METRICS, *OPTIONAL_METRICS)


class AdaptiveGranularityReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdaptiveGranularityReportResult:
    summary_path: Path
    markdown_path: Path
    summary: dict[str, Any]


def generate_adaptive_granularity_report(
    manifest_path: Path | str,
    out_dir: Path | str,
    *,
    require_real: bool = False,
) -> AdaptiveGranularityReportResult:
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    data = json.loads(manifest_path.read_text())
    rows = [row for row in data.get("runs", []) if isinstance(row, dict)]
    failures: list[str] = []
    warnings: list[str] = []
    grouped: dict[tuple[str, int, str, str, str], list[dict[str, Any]]] = defaultdict(list)

    if data.get("blocked") is True:
        failures.append("source manifest is blocked")
    if data.get("expected_result_rows") is not None and int(data.get("expected_result_rows", 0)) != EXPECTED_ROWS:
        failures.append(f"source manifest expected_result_rows is not {EXPECTED_ROWS}: {data.get('expected_result_rows')}")
    if data.get("actual_result_rows") is not None and int(data.get("actual_result_rows", 0)) != len(rows):
        failures.append(
            f"source manifest actual_result_rows does not match runs length: {data.get('actual_result_rows')} != {len(rows)}"
        )

    for index, row in enumerate(rows):
        failures.extend(_validate_row(index, row, manifest_path=manifest_path, require_real=require_real))
        workload = str(row.get("workload", ""))
        expert = _as_int(row.get("experts_per_layer"))
        policy = _row_policy(row)
        async_mode = _row_async(row)
        system = str(row.get("system", ""))
        if workload in WORKLOADS and expert in EXPERTS and policy in POLICIES:
            grouped[(workload, expert, policy, async_mode, system)].append(row)

    groups = [_group_record(key, value) for key, value in sorted(grouped.items())]
    failures.extend(_coverage_failures(grouped))
    comparisons = []
    for workload in WORKLOADS:
        for expert in EXPERTS:
            comparison = _comparison(grouped, workload, expert)
            if comparison:
                comparisons.append(comparison)
                _gate_comparison(comparison, failures, warnings)

    summary = {
        "schema_version": "tilepo_adaptive_granularity_report_v1",
        "source_manifest": str(manifest_path),
        "requested": {
            "workloads": list(WORKLOADS),
            "experts": list(EXPERTS),
            "policies": list(POLICIES),
            "expected_rows": EXPECTED_ROWS,
            "require_real": require_real,
        },
        "gate": {"status": "PASS" if not failures else "FAIL", "failures": failures, "warnings": warnings},
        "groups": groups,
        "comparisons": comparisons,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "tilepo_adaptive_granularity_summary.json"
    markdown_path = out_dir / "tilepo_adaptive_granularity_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_markdown(summary))
    if failures:
        raise AdaptiveGranularityReportError("; ".join(failures))
    return AdaptiveGranularityReportResult(summary_path, markdown_path, summary)


def _validate_row(index: int, row: dict[str, Any], *, manifest_path: Path, require_real: bool) -> list[str]:
    failures: list[str] = []
    for key in ("system", "workload", "experts_per_layer", "repeat", "tok_per_sec", "p95_ms", "p99_ms"):
        if key not in row:
            failures.append(f"row {index} missing {key}")
    for metric in ("gpu_peak_gib", "cpu_ram_peak_gib"):
        if metric not in row:
            failures.append(f"row {index} missing {metric}")
    for metric in METRICS:
        if row.get(metric) is None:
            continue
        try:
            value = _to_float(row.get(metric))
        except (TypeError, ValueError):
            failures.append(f"row {index} has non-numeric {metric}: {row.get(metric)!r}")
            continue
        if not math.isfinite(value):
            failures.append(f"row {index} has non-finite {metric}: {row.get(metric)!r}")
    if not _row_policy(row):
        failures.append(f"row {index} missing tilepo_policy/ablation_policy")
    if not _row_async(row):
        failures.append(f"row {index} missing tilepo_async_planning/async_planning_mode")
    if require_real:
        if row.get("simulated") is not False or row.get("evidence_level") != "real":
            failures.append(f"row {index} is not real evidence")
        if row.get("status") != "success":
            failures.append(f"row {index} is not success: {row.get('status')}")
        raw_path = row.get("raw_path")
        if not raw_path:
            failures.append(f"row {index} missing raw_path")
        else:
            candidate = Path(str(raw_path))
            if not candidate.is_absolute():
                candidate = manifest_path.parent / candidate
            if not candidate.exists():
                failures.append(f"row {index} raw_path does not exist: {raw_path}")
    return failures


def _coverage_failures(grouped: dict[tuple[str, int, str, str, str], list[dict[str, Any]]]) -> list[str]:
    failures: list[str] = []
    for workload in WORKLOADS:
        for expert in EXPERTS:
            expected = (workload, expert, "kt_expert", "off", "B")
            if len(grouped.get(expected, [])) < 1:
                failures.append(f"missing KT baseline row for {expected}")
            for policy in (*FIXED_POLICIES, "tilepo_adaptive"):
                key = (workload, expert, policy, "on", "C")
                if len(grouped.get(key, [])) < 1:
                    failures.append(f"missing TilePO row for {key}")
    return failures


def _group_record(key: tuple[str, int, str, str, str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    workload, expert, policy, async_mode, system = key
    return {
        "workload": workload,
        "experts_per_layer": expert,
        "policy": policy,
        "async_planning": async_mode,
        "system": system,
        "repeats": len(rows),
        "metrics": {metric: _stats(_metric_values(rows, metric)) for metric in METRICS},
    }


def _comparison(
    grouped: dict[tuple[str, int, str, str, str], list[dict[str, Any]]],
    workload: str,
    expert: int,
) -> dict[str, Any] | None:
    kt_rows = grouped.get((workload, expert, "kt_expert", "off", "B"), [])
    adaptive_rows = grouped.get((workload, expert, "tilepo_adaptive", "on", "C"), [])
    fixed = {
        policy: _aggregate(grouped.get((workload, expert, policy, "on", "C"), []))
        for policy in FIXED_POLICIES
    }
    if not kt_rows or not adaptive_rows or any(value is None for value in fixed.values()):
        return None
    kt = _aggregate(kt_rows)
    adaptive = _aggregate(adaptive_rows)
    if kt is None or adaptive is None:
        return None
    best_fixed_policy, best_fixed = max(fixed.items(), key=lambda item: item[1]["tok_per_sec"])
    coarse = fixed["tilepo_coarse"]
    fine = fixed["tilepo_fine"]
    hybrid = fixed["tilepo_hybrid"]
    return {
        "workload": workload,
        "experts_per_layer": expert,
        "best_fixed_policy": best_fixed_policy,
        "adaptive_vs_kt": {
            "tok_gain_pct": _gain_pct(adaptive["tok_per_sec"], kt["tok_per_sec"]),
            "p95_reduction_pct": _reduction_pct(adaptive["p95_ms"], kt["p95_ms"]),
            "p99_reduction_pct": _reduction_pct(adaptive["p99_ms"], kt["p99_ms"]),
            "gpu_peak_delta_pct": _gain_pct(adaptive["gpu_peak_gib"], kt["gpu_peak_gib"]),
            "cpu_ram_peak_delta_pct": _gain_pct(adaptive["cpu_ram_peak_gib"], kt["cpu_ram_peak_gib"]),
        },
        "adaptive_vs_best_fixed": {
            "tok_gap_pct": _gap_pct(adaptive["tok_per_sec"], best_fixed["tok_per_sec"]),
            "p95_delta_pct": _gain_pct(adaptive["p95_ms"], best_fixed["p95_ms"]),
            "p99_delta_pct": _gain_pct(adaptive["p99_ms"], best_fixed["p99_ms"]),
            "gpu_peak_delta_pct": _gain_pct(adaptive["gpu_peak_gib"], best_fixed["gpu_peak_gib"]),
            "cpu_ram_peak_delta_pct": _gain_pct(adaptive["cpu_ram_peak_gib"], best_fixed["cpu_ram_peak_gib"]),
        },
        "tile_count_comparison": {
            "coarse": _maybe_int(coarse.get("tile_count")),
            "fine": _maybe_int(fine.get("tile_count")),
            "hybrid": _maybe_int(hybrid.get("tile_count")),
            "adaptive": _maybe_int(adaptive.get("tile_count")),
            "adaptive_vs_fine_pct": _ratio_pct(adaptive.get("tile_count"), fine.get("tile_count")),
            "adaptive_between_coarse_and_fine": _between(
                coarse.get("tile_count"),
                adaptive.get("tile_count"),
                fine.get("tile_count"),
            ),
        },
        "dispatch_proxy": {
            "coarse": _maybe_int(coarse.get("estimated_dispatch_units")),
            "fine": _maybe_int(fine.get("estimated_dispatch_units")),
            "hybrid": _maybe_int(hybrid.get("estimated_dispatch_units")),
            "adaptive": _maybe_int(adaptive.get("estimated_dispatch_units")),
            "adaptive_vs_fine_pct": _ratio_pct(
                adaptive.get("estimated_dispatch_units"),
                fine.get("estimated_dispatch_units"),
            ),
            "adaptive_vs_hybrid_pct": _ratio_pct(
                adaptive.get("estimated_dispatch_units"),
                hybrid.get("estimated_dispatch_units"),
            ),
        },
        "memory_comparison": {
            "kt_gpu_peak_gib": kt["gpu_peak_gib"],
            "kt_cpu_ram_peak_gib": kt["cpu_ram_peak_gib"],
            "adaptive_gpu_peak_gib": adaptive["gpu_peak_gib"],
            "adaptive_cpu_ram_peak_gib": adaptive["cpu_ram_peak_gib"],
            "best_fixed_gpu_peak_gib": best_fixed["gpu_peak_gib"],
            "best_fixed_cpu_ram_peak_gib": best_fixed["cpu_ram_peak_gib"],
            "coarse_gpu_peak_gib": coarse["gpu_peak_gib"],
            "coarse_cpu_ram_peak_gib": coarse["cpu_ram_peak_gib"],
            "fine_gpu_peak_gib": fine["gpu_peak_gib"],
            "fine_cpu_ram_peak_gib": fine["cpu_ram_peak_gib"],
            "hybrid_gpu_peak_gib": hybrid["gpu_peak_gib"],
            "hybrid_cpu_ram_peak_gib": hybrid["cpu_ram_peak_gib"],
        },
        "metrics": {"kt": kt, "adaptive": adaptive, "best_fixed": best_fixed},
    }


def _gate_comparison(comparison: dict[str, Any], failures: list[str], warnings: list[str]) -> None:
    workload = comparison["workload"]
    expert = comparison["experts_per_layer"]
    adaptive_vs_kt = comparison["adaptive_vs_kt"]
    adaptive_vs_best = comparison["adaptive_vs_best_fixed"]
    if adaptive_vs_kt["tok_gain_pct"] <= 0.0:
        failures.append(f"{workload}/{expert} adaptive tok/s does not beat KT")
    if adaptive_vs_kt["p95_reduction_pct"] < 0.0:
        failures.append(f"{workload}/{expert} adaptive p95 regresses vs KT")
    if adaptive_vs_kt["p99_reduction_pct"] < 0.0:
        failures.append(f"{workload}/{expert} adaptive p99 regresses vs KT")
    if adaptive_vs_best["tok_gap_pct"] > 5.0:
        warnings.append(
            (
                f"{workload}/{expert} adaptive is {adaptive_vs_best['tok_gap_pct']:.2f}% "
                f"behind best fixed policy {comparison['best_fixed_policy']}"
            )
        )
    tile_counts = comparison["tile_count_comparison"]
    if tile_counts["adaptive"] is None or tile_counts["coarse"] is None or tile_counts["fine"] is None:
        failures.append(f"{workload}/{expert} missing tile_count comparison data")
    elif not tile_counts["adaptive_between_coarse_and_fine"]:
        failures.append(f"{workload}/{expert} adaptive tile count is not between coarse and fine")
    dispatch = comparison["dispatch_proxy"]
    if dispatch["adaptive_vs_fine_pct"] is None:
        failures.append(f"{workload}/{expert} missing dispatch proxy comparison data")
    elif dispatch["adaptive_vs_fine_pct"] > 70.0:
        failures.append(f"{workload}/{expert} adaptive dispatch proxy exceeds 70% of fine")


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    if not rows:
        return None
    record = {metric: float(median(_metric_values(rows, metric))) for metric in CORE_METRICS}
    for metric in OPTIONAL_METRICS:
        values = _metric_values(rows, metric)
        record[metric] = float(median(values)) if values else None
    return record


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"median": None, "min": None, "max": None, "count": 0}
    return {"median": float(median(values)), "min": min(values), "max": max(values), "count": len(values)}


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TilePO Adaptive Granularity Report",
        "",
        f"Gate: **{summary['gate']['status']}**",
        "",
        "## Adaptive vs KT",
        "",
        "| Workload | Experts | tok/s gain | p95 reduction | p99 reduction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for comparison in summary["comparisons"]:
        kt = comparison["adaptive_vs_kt"]
        lines.append(
            "| {workload} | {experts} | {tok:.2f}% | {p95:.2f}% | {p99:.2f}% |".format(
                workload=comparison["workload"],
                experts=comparison["experts_per_layer"],
                tok=kt["tok_gain_pct"],
                p95=kt["p95_reduction_pct"],
                p99=kt["p99_reduction_pct"],
            )
        )
    lines.extend(
        [
            "",
            "## Adaptive vs Best Fixed",
            "",
            "| Workload | Experts | Best fixed policy | tok/s gap | p95 delta | p99 delta |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for comparison in summary["comparisons"]:
        best = comparison["adaptive_vs_best_fixed"]
        lines.append(
            "| {workload} | {experts} | {policy} | {tok:.2f}% | {p95:.2f}% | {p99:.2f}% |".format(
                workload=comparison["workload"],
                experts=comparison["experts_per_layer"],
                policy=comparison["best_fixed_policy"],
                tok=best["tok_gap_pct"],
                p95=best["p95_delta_pct"],
                p99=best["p99_delta_pct"],
            )
        )
    lines.extend(
        [
            "",
            "## Tile Count and Dispatch Proxy",
            "",
            "| Workload | Experts | Tile count C/F/H/A | Dispatch A/F | GPU peak A/KT/Best | DRAM peak A/KT/Best |",
            "| --- | ---: | --- | ---: | --- | --- |",
        ]
    )
    for comparison in summary["comparisons"]:
        tile_counts = comparison["tile_count_comparison"]
        dispatch = comparison["dispatch_proxy"]
        memory = comparison["memory_comparison"]
        lines.append(
            "| {workload} | {experts} | {coarse}/{fine}/{hybrid}/{adaptive} | {dispatch:.2f}% | "
            "{adaptive_gpu:.3f}/{kt_gpu:.3f}/{best_gpu:.3f} | "
            "{adaptive_cpu:.3f}/{kt_cpu:.3f}/{best_cpu:.3f} |".format(
                workload=comparison["workload"],
                experts=comparison["experts_per_layer"],
                coarse=_fmt_nullable(tile_counts["coarse"]),
                fine=_fmt_nullable(tile_counts["fine"]),
                hybrid=_fmt_nullable(tile_counts["hybrid"]),
                adaptive=_fmt_nullable(tile_counts["adaptive"]),
                dispatch=dispatch["adaptive_vs_fine_pct"] if dispatch["adaptive_vs_fine_pct"] is not None else 0.0,
                adaptive_gpu=memory["adaptive_gpu_peak_gib"],
                kt_gpu=memory["kt_gpu_peak_gib"],
                best_gpu=memory["best_fixed_gpu_peak_gib"],
                adaptive_cpu=memory["adaptive_cpu_ram_peak_gib"],
                kt_cpu=memory["kt_cpu_ram_peak_gib"],
                best_cpu=memory["best_fixed_cpu_ram_peak_gib"],
            )
        )
    if summary["gate"]["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["gate"]["warnings"])
    if summary["gate"]["failures"]:
        lines.extend(["", "## Gate Failures", ""])
        lines.extend(f"- {failure}" for failure in summary["gate"]["failures"])
    return "\n".join(lines) + "\n"


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


def _metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values = []
    for row in rows:
        if row.get(metric) is not None:
            values.append(_to_float(row.get(metric)))
    return values


def _gain_pct(candidate: float, baseline: float) -> float:
    return ((candidate / baseline) - 1.0) * 100.0 if baseline else 0.0


def _reduction_pct(candidate: float, baseline: float) -> float:
    return (1.0 - (candidate / baseline)) * 100.0 if baseline else 0.0


def _gap_pct(candidate: float, best: float) -> float:
    return max(0.0, (1.0 - (candidate / best)) * 100.0) if best else 0.0


def _ratio_pct(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline in {None, 0.0}:
        return None
    return (candidate / baseline) * 100.0


def _maybe_int(value: float | None) -> int | None:
    return int(value) if value is not None else None


def _between(lower: float | None, candidate: float | None, upper: float | None) -> bool | None:
    if lower is None or candidate is None or upper is None:
        return None
    return lower < candidate < upper


def _fmt_nullable(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(int(value))
