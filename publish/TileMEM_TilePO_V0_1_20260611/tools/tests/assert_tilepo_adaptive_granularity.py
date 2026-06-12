#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "adaptive_manifest.json"
        _write_raw_files(root, _rows())
        manifest.write_text(json.dumps({"schema_version": "tilepo_merged_manifest_v1", "runs": _rows()}, indent=2))
        out_dir = root / "report"

        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "report_tilepo_adaptive_granularity"),
                "--manifest",
                str(manifest),
                "--out-dir",
                str(out_dir),
                "--require-real",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        summary = json.loads((out_dir / "tilepo_adaptive_granularity_summary.json").read_text())
        assert summary["schema_version"] == "tilepo_adaptive_granularity_report_v1"
        assert summary["gate"]["status"] == "PASS", summary["gate"]
        assert summary["requested"]["expected_rows"] == 30
        assert len(summary["groups"]) == 30
        assert len(summary["comparisons"]) == 6

        mixed8 = _comparison(summary, "mixed", 8)
        assert mixed8["adaptive_vs_kt"]["tok_gain_pct"] > 0.0
        assert mixed8["adaptive_vs_kt"]["p95_reduction_pct"] >= 0.0
        assert mixed8["adaptive_vs_kt"]["p99_reduction_pct"] >= 0.0
        assert "gpu_peak_delta_pct" in mixed8["adaptive_vs_kt"]
        assert "cpu_ram_peak_delta_pct" in mixed8["adaptive_vs_kt"]
        assert mixed8["best_fixed_policy"] == "tilepo_fine"
        assert mixed8["adaptive_vs_best_fixed"]["tok_gap_pct"] <= 5.0
        assert "gpu_peak_delta_pct" in mixed8["adaptive_vs_best_fixed"]
        assert "cpu_ram_peak_delta_pct" in mixed8["adaptive_vs_best_fixed"]
        assert mixed8["tile_count_comparison"]["coarse"] < mixed8["tile_count_comparison"]["adaptive"]
        assert mixed8["tile_count_comparison"]["adaptive"] < mixed8["tile_count_comparison"]["fine"]
        assert mixed8["dispatch_proxy"]["adaptive_vs_fine_pct"] <= 70.0
        assert mixed8["memory_comparison"]["kt_gpu_peak_gib"] == 8.0
        assert mixed8["memory_comparison"]["best_fixed_gpu_peak_gib"] == 6.4
        assert mixed8["memory_comparison"]["adaptive_gpu_peak_gib"] <= mixed8["memory_comparison"]["coarse_gpu_peak_gib"]

        long10 = _comparison(summary, "long_context", 10)
        assert long10["best_fixed_policy"] == "tilepo_coarse"
        assert long10["adaptive_vs_best_fixed"]["tok_gap_pct"] <= 5.0

        markdown = (out_dir / "tilepo_adaptive_granularity_report.md").read_text()
        assert "Adaptive vs KT" in markdown
        assert "Adaptive vs Best Fixed" in markdown
        assert "Tile Count and Dispatch Proxy" in markdown
        _assert_non_finite_metrics_are_rejected(root)

    return 0


def _assert_non_finite_metrics_are_rejected(root: Path) -> None:
    rows = _rows()
    for row in rows:
        if row["tilepo_policy"] == "tilepo_adaptive" and row["workload"] == "mixed" and row["experts_per_layer"] == 8:
            row["tok_per_sec"] = "nan"
            break
    _write_raw_files(root, rows)
    bad_manifest = root / "bad_adaptive_manifest.json"
    bad_manifest.write_text(json.dumps({"schema_version": "tilepo_merged_manifest_v1", "runs": rows}, indent=2))
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "report_tilepo_adaptive_granularity"),
            "--manifest",
            str(bad_manifest),
            "--out-dir",
            str(root / "bad_report"),
            "--require-real",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "non-finite tok_per_sec" in proc.stderr


def _write_raw_files(root: Path, rows: list[dict]) -> None:
    for row in rows:
        raw_path = root / row["raw_path"]
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(row) + "\n")


def _comparison(summary: dict, workload: str, experts: int) -> dict:
    for item in summary["comparisons"]:
        if item["workload"] == workload and item["experts_per_layer"] == experts:
            return item
    raise AssertionError(f"missing comparison for {workload}/{experts}")


def _rows() -> list[dict]:
    rows = []
    for workload in ("mixed", "long_context"):
        for experts in (6, 8, 10):
            rows.append(_row("kt_expert", "off", "B", workload, experts, 0, 100.0, 100.0, 120.0, 8.0))
            if workload == "mixed" and experts == 8:
                fixed = {
                    "tilepo_coarse": (124.0, 83.0, 101.0, 7.2, 256, 256),
                    "tilepo_fine": (132.0, 80.0, 98.0, 6.4, 16384, 16384),
                    "tilepo_hybrid": (128.0, 82.0, 100.0, 6.9, 8200, 8200),
                    "tilepo_adaptive": (127.0, 79.0, 97.0, 6.6, 4480, 4480),
                }
            elif workload == "long_context" and experts == 10:
                fixed = {
                    "tilepo_coarse": (130.0, 78.0, 96.0, 7.1, 320, 320),
                    "tilepo_fine": (124.0, 82.0, 101.0, 6.3, 20480, 20480),
                    "tilepo_hybrid": (127.0, 80.0, 98.0, 6.7, 12300, 12300),
                    "tilepo_adaptive": (126.0, 79.0, 97.0, 6.5, 4640, 4640),
                }
            else:
                fixed = {
                    "tilepo_coarse": (120.0, 86.0, 104.0, 7.3, experts * 32, experts * 32),
                    "tilepo_fine": (122.0, 84.0, 102.0, 6.4, experts * 2048, experts * 2048),
                    "tilepo_hybrid": (123.0, 83.0, 101.0, 6.8, experts * 1100, experts * 1100),
                    "tilepo_adaptive": (121.5, 82.0, 100.0, 6.6, experts * 760, experts * 760),
                }
            for policy, (tok, p95, p99, gpu, tile_count, dispatch_units) in fixed.items():
                rows.append(_row(policy, "on", "C", workload, experts, 0, tok, p95, p99, gpu, tile_count, dispatch_units))
    return rows


def _row(
    policy: str,
    async_mode: str,
    system: str,
    workload: str,
    experts: int,
    repeat: int,
    tok: float,
    p95: float,
    p99: float,
    gpu: float,
    tile_count: int | None = None,
    dispatch_units: int | None = None,
) -> dict:
    row = {
        "system": system,
        "workload": workload,
        "experts_per_layer": experts,
        "repeat": repeat,
        "request_count": 5,
        "warmup_request_count": 1,
        "tok_per_sec": tok,
        "p50_ms": p95 * 0.8,
        "p95_ms": p95,
        "p99_ms": p99,
        "gpu_peak_gib": gpu,
        "cpu_ram_peak_gib": 18.0,
        "server_ready_s": 4.0,
        "fallback_count": 0,
        "dtype_counts": {"bf16": 1},
        "command": ["python", "-m", "sglang.launch_server", "--kt-method", "BF16", "--dtype", "bfloat16"],
        "evidence_level": "real",
        "simulated": False,
        "status": "success",
        "tilepo_policy": policy,
        "tilepo_async_planning": async_mode,
        "raw_path": f"raw/{policy}_{async_mode}_{system}_{experts}_{repeat}.jsonl",
    }
    if tile_count is not None:
        row["tile_count"] = tile_count
    if dispatch_units is not None:
        row["estimated_dispatch_units"] = dispatch_units
    return row


if __name__ == "__main__":
    raise SystemExit(main())
