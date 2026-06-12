from __future__ import annotations

import json
from pathlib import Path

from .schema import PredictionResult


def write_prediction_outputs(result: PredictionResult, out_dir: Path | str) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "tmap_prediction_summary.json"
    report_path = out_dir / "tmap_prediction_report.md"
    summary_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    report_path.write_text(render_markdown(result))
    return summary_path, report_path


def render_markdown(result: PredictionResult) -> str:
    lines = [
        "# TMAP V0.2 Prediction Report",
        "",
        "TMAP is a two-tier Tile Memory Allocation Predictor for TileMEM. This",
        "report uses V0.1 TilePO/KT measurements as calibration samples and",
        "applies a VRAM/DRAM hardware profile to predict relative policy",
        "preference and conservative fallback decisions.",
        "",
        "## Hardware Profile",
        "",
        f"- Name: `{result.hardware.name}`",
        f"- VRAM: {result.hardware.vram_capacity_gib:.2f} GiB, {result.hardware.vram_bandwidth_gbps:.2f} GB/s, {result.hardware.vram_latency_ns:.2f} ns",
        f"- DRAM: {result.hardware.dram_capacity_gib:.2f} GiB, {result.hardware.dram_bandwidth_gbps:.2f} GB/s, {result.hardware.dram_latency_ns:.2f} ns",
        f"- Transfer: {result.hardware.transfer_bandwidth_gbps:.2f} GB/s, {result.hardware.transfer_latency_us:.2f} us",
        "",
        "## Summary",
        "",
        f"- Groups: {result.summary['groups']}",
        f"- Measured groups: {result.summary.get('measured_groups', result.summary['groups'])}",
        f"- Extrapolated groups: {result.summary.get('extrapolated_groups', 0)}",
        f"- Admit TilePO: {result.summary['admit_tilepo']}",
        f"- Fallback KT: {result.summary['fallback_kt']}",
        f"- TilePO candidate-rank accuracy against V0.1 observed best TilePO tok/s: {result.summary['rank_accuracy']:.2f}",
        f"- Mean predicted tok/s gain: {result.summary['mean_predicted_tok_gain_pct']:.2f}%",
        f"- Mean predicted p95 reduction: {result.summary['mean_predicted_p95_reduction_pct']:.2f}%",
        "",
        "## Decisions",
        "",
        "| Workload | Experts | Evidence | Admit | Recommended policy | Pred. tok/s gain | Pred. p95 reduction | Confidence | Probe | Factor | Risk |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for decision in result.decisions:
        lines.append(
            "| "
            f"{decision.workload} | {decision.experts_per_layer} | {decision.evidence_mode} | "
            f"{decision.admitted_system} | "
            f"{decision.recommended_policy} | {decision.predicted_tok_gain_pct:.2f}% | "
            f"{decision.predicted_p95_reduction_pct:.2f}% | {decision.confidence:.2f} | "
            f"{'yes' if decision.probe_recommended else 'no'} | {decision.dominant_factor} | {decision.risk} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "TMAP V0.2 predicts relative policy preference, not exact serving",
            "throughput. It is calibrated from V0.1 BF16 samples and uses a",
            "two-tier VRAM/DRAM model only. Extrapolated expert budgets are",
            "quick-planning estimates and must be validated with a short probe.",
            "Mixed precision and multi-tier memory are out of scope for this version.",
            "",
        ]
    )
    return "\n".join(lines)
