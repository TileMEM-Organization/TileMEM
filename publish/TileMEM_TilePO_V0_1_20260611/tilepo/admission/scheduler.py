from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


LOWBIT_DTYPES = {"fp8", "float8", "mxfp4", "mx-fp4", "fp4", "int4", "uint4"}
SUPPORTED_POLICIES = {"throughput", "tail_latency", "memory_saving", "balanced"}
METRICS = (
    "tok_per_sec",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "gpu_peak_gib",
    "cpu_ram_peak_gib",
    "server_ready_s",
    "fallback_count",
)


class AdmissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdmissionResult:
    summary_path: Path
    markdown_path: Path
    html_path: Path
    selected_plan_path: Path
    fallback_plan_path: Path
    server_command_path: Path
    summary: dict[str, Any]


def admit_tilepo(
    *,
    manifest_paths: list[Path | str],
    out_dir: Path | str,
    workloads: list[str],
    experts: list[int],
    systems: list[str] | None = None,
    repeats: int = 3,
    policy: str = "balanced",
    tie_band: float = 0.01,
    tok_win_threshold: float = 0.03,
    require_bf16: bool = False,
) -> AdmissionResult:
    if policy not in SUPPORTED_POLICIES:
        raise AdmissionError(f"unsupported policy: {policy}")
    selected_systems = systems or ["B", "C"]
    if selected_systems != ["B", "C"]:
        raise AdmissionError(f"TilePO admission requires systems B,C; got {selected_systems}")
    if not manifest_paths:
        raise AdmissionError("at least one manifest is required")

    out_dir = Path(out_dir)
    manifests = [Path(path) for path in manifest_paths]
    rows = _load_rows(manifests)
    failures = _validate_rows(rows, require_bf16=require_bf16)
    grouped = _group_rows(
        rows,
        workloads=set(workloads),
        experts=set(experts),
        systems=set(selected_systems),
    )
    per_budget, coverage_failures = _per_budget_decisions(
        grouped,
        workloads=workloads,
        experts=experts,
        repeats=repeats,
        tie_band=tie_band,
        tok_win_threshold=tok_win_threshold,
    )
    failures.extend(coverage_failures)
    selections = _select_workload_plans(per_budget, workloads=workloads, policy=policy)
    if not selections:
        failures.append("no selectable KT/TilePO candidate")
    gate = {"status": "PASS" if not failures else "FAIL", "failures": failures}
    summary = {
        "schema_version": "tilepo_v0_1_admission_v1",
        "input_manifests": [str(path) for path in manifests],
        "policy": policy,
        "requested": {
            "workloads": workloads,
            "experts": experts,
            "systems": selected_systems,
            "repeats": repeats,
            "tie_band": tie_band,
            "tok_win_threshold": tok_win_threshold,
            "require_bf16": require_bf16,
        },
        "gate": gate,
        "selections": selections,
        "per_budget_decisions": per_budget,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "admission_summary.json"
    markdown_path = out_dir / "admission_report.md"
    html_path = out_dir / "admission_report.html"
    selected_plan_path = out_dir / "selected_plan.json"
    fallback_plan_path = out_dir / "fallback_plan.json"
    server_command_path = out_dir / "selected_server_command.sh"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_markdown(summary))
    html_path.write_text(_html(summary))
    selected_plan_path.write_text(json.dumps(_selected_plan(summary), indent=2, sort_keys=True) + "\n")
    fallback_plan_path.write_text(json.dumps(_fallback_plan(summary), indent=2, sort_keys=True) + "\n")
    server_command_path.write_text(_server_command_script(summary))
    if failures:
        raise AdmissionError("; ".join(failures))
    return AdmissionResult(
        summary_path=summary_path,
        markdown_path=markdown_path,
        html_path=html_path,
        selected_plan_path=selected_plan_path,
        fallback_plan_path=fallback_plan_path,
        server_command_path=server_command_path,
        summary=summary,
    )


def _load_rows(manifest_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in manifest_paths:
        data = json.loads(path.read_text())
        manifest_rows = data.get("runs", [])
        if not isinstance(manifest_rows, list):
            raise AdmissionError(f"manifest has invalid runs: {path}")
        for row in manifest_rows:
            if isinstance(row, dict):
                rows.append(row)
    if not rows:
        raise AdmissionError("manifests contain no rows")
    return rows


def _validate_rows(rows: list[dict[str, Any]], *, require_bf16: bool) -> list[str]:
    failures: list[str] = []
    for index, row in enumerate(rows):
        if row.get("simulated") is not False or row.get("evidence_level") != "real":
            failures.append(f"row {index} is not real evidence")
        if row.get("status") != "success":
            failures.append(f"row {index} is not success: {row.get('status')}")
        if require_bf16:
            command = " ".join(str(item) for item in row.get("command", []))
            if "--kt-method BF16" not in command:
                failures.append(f"row {index} missing BF16 KT method")
            if "--dtype bfloat16" not in command:
                failures.append(f"row {index} missing bfloat16 dtype")
            for field in ("dtype_counts", "serving_hook_backend_dtype_counts"):
                counts = row.get(field, {})
                if not isinstance(counts, dict):
                    continue
                for dtype, count in counts.items():
                    if str(dtype).lower() in LOWBIT_DTYPES and _as_int(count, default=0) > 0:
                        failures.append(f"row {index} has low-bit dtype in {field}: {dtype}:{count}")
    return failures


def _group_rows(
    rows: list[dict[str, Any]],
    *,
    workloads: set[str],
    experts: set[int],
    systems: set[str],
) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        workload = str(row.get("workload"))
        expert = _as_int(row.get("experts_per_layer"))
        system = str(row.get("system"))
        if workload in workloads and expert in experts and system in systems:
            grouped[(workload, expert, system)].append(row)
    return grouped


def _per_budget_decisions(
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]],
    *,
    workloads: list[str],
    experts: list[int],
    repeats: int,
    tie_band: float,
    tok_win_threshold: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    decisions: list[dict[str, Any]] = []
    failures: list[str] = []
    for workload in workloads:
        has_candidate = False
        for expert in experts:
            b_rows = grouped.get((workload, expert, "B"), [])
            c_rows = grouped.get((workload, expert, "C"), [])
            if len(b_rows) < repeats or len(c_rows) < repeats:
                continue
            has_candidate = True
            b = _aggregate(b_rows[:repeats])
            c = _aggregate(c_rows[:repeats])
            strict = _strict_win(b, c, tok_win_threshold=tok_win_threshold)
            admitted_system = "TilePO" if strict["passes"] else "KT"
            decisions.append(
                {
                    "workload": workload,
                    "experts_per_layer": expert,
                    "admitted_system": admitted_system,
                    "fallback_system": "KT",
                    "status": "ADMIT_TilePO" if admitted_system == "TilePO" else "FALLBACK_KT",
                    "systems": {"B": b, "C": c},
                    "comparisons": _comparisons(b, c, tie_band=tie_band),
                    "strict_win": strict,
                    "reason": _decision_reason(workload, expert, strict, admitted_system),
                }
            )
        if not has_candidate:
            failures.append(f"no complete B/C candidate for workload {workload}")
    return decisions, failures


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {"repeats": len(rows)}
    for metric in METRICS:
        values = [_to_float(row.get(metric)) for row in rows if row.get(metric) is not None]
        record[metric] = {"median": float(median(values)) if values else None}
    record["init_expert_location"] = _last_init_path(rows)
    record["example_command"] = rows[0].get("command", []) if rows else []
    return record


def _last_init_path(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        command = [str(item) for item in row.get("command", [])]
        if "--init-expert-location" in command:
            index = len(command) - 1 - list(reversed(command)).index("--init-expert-location")
            if index + 1 < len(command):
                return command[index + 1]
    return ""


def _comparisons(b: dict[str, Any], c: dict[str, Any], *, tie_band: float) -> dict[str, Any]:
    comparisons = {}
    for metric in METRICS:
        b_value = b.get(metric, {}).get("median")
        c_value = c.get(metric, {}).get("median")
        if b_value is None or c_value is None:
            continue
        lower = metric != "tok_per_sec"
        delta = _relative_delta(c_value, b_value)
        if abs(delta) <= tie_band:
            relation = "tie"
        elif lower:
            relation = "win" if delta < 0 else "loss"
        else:
            relation = "win" if delta > 0 else "loss"
        comparisons[metric] = {"c_vs_b_delta": delta, "relation": relation}
    return comparisons


def _strict_win(b: dict[str, Any], c: dict[str, Any], *, tok_win_threshold: float) -> dict[str, Any]:
    b_tok = b["tok_per_sec"]["median"]
    c_tok = c["tok_per_sec"]["median"]
    b_p95 = b["p95_ms"]["median"]
    c_p95 = c["p95_ms"]["median"]
    b_p99 = b["p99_ms"]["median"]
    c_p99 = c["p99_ms"]["median"]
    tok_delta = _relative_delta(c_tok, b_tok)
    tok_pass = c_tok is not None and b_tok is not None and c_tok >= b_tok * (1.0 + tok_win_threshold)
    p95_pass = c_p95 is not None and b_p95 is not None and c_p95 <= b_p95
    p99_pass = c_p99 is not None and b_p99 is not None and c_p99 <= b_p99
    return {
        "passes": bool(tok_pass and p95_pass and p99_pass),
        "tok_per_sec_delta": tok_delta,
        "tok_per_sec_pass": bool(tok_pass),
        "p95_ms_pass": bool(p95_pass),
        "p99_ms_pass": bool(p99_pass),
    }


def _select_workload_plans(
    per_budget: list[dict[str, Any]],
    *,
    workloads: list[str],
    policy: str,
) -> dict[str, dict[str, Any]]:
    selections: dict[str, dict[str, Any]] = {}
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in per_budget:
        by_workload[item["workload"]].append(item)
    for workload in workloads:
        items = by_workload.get(workload, [])
        if not items:
            continue
        tilepo_items = [item for item in items if item["admitted_system"] == "TilePO"]
        if tilepo_items:
            chosen = max(tilepo_items, key=lambda item: _score(item, "C", policy))
            selected_system = "TilePO"
            selected_metrics = chosen["systems"]["C"]
        else:
            chosen = max(items, key=lambda item: _score(item, "B", policy))
            selected_system = "KT"
            selected_metrics = chosen["systems"]["B"]
        selections[workload] = {
            "workload": workload,
            "selected_system": selected_system,
            "selected_expert_budget": chosen["experts_per_layer"],
            "policy": policy,
            "fallback_system": "KT",
            "fallback_expert_budget": chosen["experts_per_layer"],
            "serving_precision": "BF16",
            "init_expert_location": selected_metrics.get("init_expert_location", ""),
            "median_tok_per_sec": selected_metrics["tok_per_sec"]["median"],
            "median_p95_ms": selected_metrics["p95_ms"]["median"],
            "median_p99_ms": selected_metrics["p99_ms"]["median"],
            "reason": chosen["reason"],
        }
    return selections


def _score(item: dict[str, Any], system: str, policy: str) -> float:
    metrics = item["systems"][system]
    tok = _finite(metrics["tok_per_sec"]["median"], default=0.0)
    p95 = _finite(metrics["p95_ms"]["median"], default=1_000_000.0)
    gpu = _finite(metrics["gpu_peak_gib"]["median"], default=1_000_000.0)
    if policy == "tail_latency":
        return -p95
    if policy == "memory_saving":
        return -gpu
    if policy == "balanced":
        return tok - 0.01 * p95 - 0.1 * gpu
    return tok


def _decision_reason(workload: str, expert: int, strict: dict[str, Any], admitted_system: str) -> str:
    delta = strict["tok_per_sec_delta"] * 100.0
    if admitted_system == "TilePO":
        return f"{workload}/{expert}: TilePO admitted, tok/s {delta:+.2f}% with non-regressing p95/p99"
    return f"{workload}/{expert}: KT fallback, TilePO did not pass tok/s+p95+p99 gate"


def _selected_plan(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "tilepo_v0_1_selected_plan_v1",
        "policy": summary["policy"],
        "gate": summary["gate"],
        "selections": summary["selections"],
    }


def _fallback_plan(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "tilepo_v0_1_fallback_plan_v1",
        "policy": summary["policy"],
        "fallback_system": "KT",
        "selections": {
            workload: {
                "selected_system": "KT",
                "selected_expert_budget": selection["fallback_expert_budget"],
                "serving_precision": "BF16",
            }
            for workload, selection in summary["selections"].items()
        },
    }


def _server_command_script(summary: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by TilePO V0.1 admission. Fill MODEL_PATH/PORT for the deployment target.",
    ]
    for workload, selection in sorted(summary["selections"].items()):
        init_path = selection.get("init_expert_location") or "<init-expert-location>"
        system = selection["selected_system"]
        strategy = "frequency"
        lines.extend(
            [
                "",
                f"# {workload}: {system} / experts={selection['selected_expert_budget']}",
                "python -m sglang.launch_server \\",
                "  --model-path \"$MODEL_PATH\" \\",
                "  --dtype bfloat16 \\",
                "  --kt-method BF16 \\",
                f"  --kt-num-gpu-experts {selection['selected_expert_budget']} \\",
                f"  --kt-expert-placement-strategy {strategy} \\",
                f"  --init-expert-location {init_path}",
            ]
        )
    return "\n".join(lines) + "\n"


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TilePO V0.1 Admission Report",
        "",
        f"Gate: **{summary['gate']['status']}**",
        f"Policy: `{summary['policy']}`",
        "",
        "## Selected Plans",
        "",
        "| Workload | Selected | Experts | tok/s | p95 | p99 | Reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for workload, selection in sorted(summary["selections"].items()):
        lines.append(
            f"| `{workload}` | {selection['selected_system']} | {selection['selected_expert_budget']} | "
            f"{_fmt(selection['median_tok_per_sec'])} | {_fmt(selection['median_p95_ms'])} | "
            f"{_fmt(selection['median_p99_ms'])} | {selection['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Per-Budget Decisions",
            "",
            "| Workload | Experts | Decision | B tok/s | C tok/s | C vs B tok/s | p95 | p99 |",
            "| --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for item in summary["per_budget_decisions"]:
        b = item["systems"]["B"]
        c = item["systems"]["C"]
        lines.append(
            f"| `{item['workload']}` | {item['experts_per_layer']} | {item['admitted_system']} | "
            f"{_fmt(b['tok_per_sec']['median'])} | {_fmt(c['tok_per_sec']['median'])} | "
            f"{item['strict_win']['tok_per_sec_delta'] * 100:+.2f}% | "
            f"{item['comparisons'].get('p95_ms', {}).get('relation', 'n/a')} | "
            f"{item['comparisons'].get('p99_ms', {}).get('relation', 'n/a')} |"
        )
    if summary["gate"]["failures"]:
        lines.extend(["", "## Gate Failures", ""])
        lines.extend(f"- {failure}" for failure in summary["gate"]["failures"])
    return "\n".join(lines) + "\n"


def _html(summary: dict[str, Any]) -> str:
    selected_rows = []
    for workload, selection in sorted(summary["selections"].items()):
        selected_rows.append(
            "<tr>"
            f"<td>{html.escape(workload)}</td>"
            f"<td>{html.escape(str(selection['selected_system']))}</td>"
            f"<td>{selection['selected_expert_budget']}</td>"
            f"<td>{_fmt(selection['median_tok_per_sec'])}</td>"
            f"<td>{_fmt(selection['median_p95_ms'])}</td>"
            f"<td>{_fmt(selection['median_p99_ms'])}</td>"
            f"<td>{html.escape(selection['reason'])}</td>"
            "</tr>"
        )
    budget_rows = []
    for item in summary["per_budget_decisions"]:
        b = item["systems"]["B"]
        c = item["systems"]["C"]
        status_class = "tilepo" if item["admitted_system"] == "TilePO" else "kt"
        budget_rows.append(
            f"<tr class=\"{status_class}\">"
            f"<td>{html.escape(item['workload'])}</td>"
            f"<td>{item['experts_per_layer']}</td>"
            f"<td>{item['admitted_system']}</td>"
            f"<td>{_fmt(b['tok_per_sec']['median'])}</td>"
            f"<td>{_fmt(c['tok_per_sec']['median'])}</td>"
            f"<td>{item['strict_win']['tok_per_sec_delta'] * 100:+.2f}%</td>"
            f"<td>{item['comparisons'].get('p95_ms', {}).get('relation', 'n/a')}</td>"
            f"<td>{item['comparisons'].get('p99_ms', {}).get('relation', 'n/a')}</td>"
            "</tr>"
        )
    failures = "".join(f"<li>{html.escape(failure)}</li>" for failure in summary["gate"]["failures"])
    failure_block = f"<section><h2>Gate Failures</h2><ul>{failures}</ul></section>" if failures else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TilePO V0.1 Admission Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #5b6670;
      --line: #d8dee4;
      --tilepo: #0f766e;
      --kt: #9a3412;
      --bg: #f7f8f9;
    }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
      line-height: 1.15;
    }}
    h2 {{
      margin: 28px 0 12px;
      font-size: 18px;
    }}
    .meta {{
      color: var(--muted);
      margin-bottom: 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border: 1px solid var(--line);
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      background: #eef2f4;
      font-weight: 650;
    }}
    tr.tilepo td:nth-child(3) {{
      color: var(--tilepo);
      font-weight: 700;
    }}
    tr.kt td:nth-child(3) {{
      color: var(--kt);
      font-weight: 700;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
  </style>
</head>
<body>
  <main>
    <h1>TilePO V0.1 Admission Report</h1>
    <p class="meta">Gate: <strong>{html.escape(summary['gate']['status'])}</strong> · Policy: <code>{html.escape(summary['policy'])}</code></p>
    <section>
      <h2>Selected Plans</h2>
      <table>
        <thead><tr><th>Workload</th><th>Selected</th><th>Experts</th><th>tok/s</th><th>p95</th><th>p99</th><th>Reason</th></tr></thead>
        <tbody>{''.join(selected_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>Per-Budget Decisions</h2>
      <table>
        <thead><tr><th>Workload</th><th>Experts</th><th>Decision</th><th>B tok/s</th><th>C tok/s</th><th>C vs B tok/s</th><th>p95</th><th>p99</th></tr></thead>
        <tbody>{''.join(budget_rows)}</tbody>
      </table>
    </section>
    {failure_block}
  </main>
</body>
</html>
"""


def _relative_delta(new: float | None, old: float | None) -> float:
    if new is None or old is None or old == 0 or not math.isfinite(old):
        return 0.0
    return (new - old) / old


def _finite(value: float | None, *, default: float) -> float:
    if value is None or not math.isfinite(value):
        return default
    return float(value)


def _as_int(value: Any, *, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any) -> float:
    return float(value)


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
