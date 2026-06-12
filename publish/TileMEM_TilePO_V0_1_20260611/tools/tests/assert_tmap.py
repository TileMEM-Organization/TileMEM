#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "TMAP"))

from tmap.predictor import predict_from_summary  # noqa: E402
from tmap.report import write_prediction_outputs  # noqa: E402
from tmap.schema import HardwareProfile  # noqa: E402


SUMMARY = ROOT / "evidence" / "ablation" / "tilepo_ablation_summary.json"
SUMMARY_REL = Path("evidence") / "ablation" / "tilepo_ablation_summary.json"
MANIFEST = ROOT / "evidence" / "ablation" / "tilepo_ablation_manifest.json"


RTX_5090_DDR = HardwareProfile(
    name="rtx5090_ddr_test",
    vram_capacity_gib=32.0,
    vram_bandwidth_gbps=1792.0,
    vram_latency_ns=350.0,
    dram_capacity_gib=128.0,
    dram_bandwidth_gbps=95.0,
    dram_latency_ns=90_000.0,
    transfer_bandwidth_gbps=64.0,
    transfer_latency_us=12.0,
)


ABUNDANT_VRAM = HardwareProfile(
    name="abundant_vram_test",
    vram_capacity_gib=96.0,
    vram_bandwidth_gbps=2200.0,
    vram_latency_ns=300.0,
    dram_capacity_gib=128.0,
    dram_bandwidth_gbps=95.0,
    dram_latency_ns=90_000.0,
    transfer_bandwidth_gbps=64.0,
    transfer_latency_us=12.0,
)


SLOW_TRANSFER = HardwareProfile(
    name="slow_transfer_test",
    vram_capacity_gib=32.0,
    vram_bandwidth_gbps=1792.0,
    vram_latency_ns=350.0,
    dram_capacity_gib=128.0,
    dram_bandwidth_gbps=95.0,
    dram_latency_ns=90_000.0,
    transfer_bandwidth_gbps=1.0,
    transfer_latency_us=12.0,
)


def assert_close_enough(value: float, lower: float, upper: float, label: str) -> None:
    if not lower <= value <= upper:
        raise AssertionError(f"{label}={value:.4f} outside [{lower:.4f}, {upper:.4f}]")


def test_predicts_v0_1_matrix() -> None:
    result = predict_from_summary(SUMMARY, RTX_5090_DDR, admit_threshold_pct=3.0)
    assert result.hardware.name == "rtx5090_ddr_test"
    assert len(result.decisions) == 10
    assert result.summary["groups"] == 10
    assert result.summary["admit_tilepo"] >= 8
    assert result.summary["fallback_kt"] <= 2
    assert_close_enough(result.summary["rank_accuracy"], 0.7, 1.0, "rank_accuracy")

    mixed_8 = result.decision_for("mixed", 8)
    assert mixed_8.admitted_system == "TilePO"
    assert mixed_8.recommended_policy.startswith("tilepo_")
    assert mixed_8.predicted_tok_gain_pct > 10.0
    assert mixed_8.confidence >= 0.5
    assert mixed_8.dominant_factor in {
        "observed_tilepo_gain",
        "vram_pressure_reduction",
        "latency_tail_reduction",
    }


def test_abundant_vram_profile_falls_back_more_often() -> None:
    constrained = predict_from_summary(SUMMARY, RTX_5090_DDR, admit_threshold_pct=3.0)
    abundant = predict_from_summary(SUMMARY, ABUNDANT_VRAM, admit_threshold_pct=3.0)
    assert abundant.summary["admit_tilepo"] < constrained.summary["admit_tilepo"]
    assert abundant.summary["fallback_kt"] >= 5
    assert abundant.summary["mean_predicted_tok_gain_pct"] < constrained.summary["mean_predicted_tok_gain_pct"]


def test_transfer_bandwidth_affects_prediction() -> None:
    normal = predict_from_summary(SUMMARY, RTX_5090_DDR, admit_threshold_pct=3.0)
    slow = predict_from_summary(SUMMARY, SLOW_TRANSFER, admit_threshold_pct=3.0)
    assert slow.summary["mean_predicted_tok_gain_pct"] < normal.summary["mean_predicted_tok_gain_pct"]


def test_extrapolates_unseen_expert_budget_with_probe_recommendation() -> None:
    result = predict_from_summary(
        SUMMARY,
        RTX_5090_DDR,
        admit_threshold_pct=3.0,
        target_pairs=[("mixed", 12)],
        allow_extrapolation=True,
    )
    mixed_12 = result.decision_for("mixed", 12)
    assert mixed_12.evidence_mode == "extrapolated"
    assert mixed_12.admitted_system == "KT"
    assert mixed_12.observed_tok_gain_pct is None
    assert mixed_12.observed_p95_reduction_pct is None
    assert mixed_12.confidence < 0.7
    assert mixed_12.probe_recommended is True
    assert mixed_12.probe_recommendation["runs"][0]["system"] == "KT"
    assert mixed_12.probe_recommendation["runs"][0]["experts_per_layer"] == 12
    assert mixed_12.probe_recommendation["runs"][1]["system"] == "TilePO"
    assert "extrapolated" in mixed_12.explanation


def test_measured_decisions_do_not_serialize_probe_fields() -> None:
    result = predict_from_summary(SUMMARY, RTX_5090_DDR, admit_threshold_pct=3.0)
    measured = result.to_dict()["decisions"][0]
    assert measured["evidence_mode"] == "measured"
    assert "nearest_experts" not in measured
    assert "probe_recommended" not in measured
    assert "probe_plan" not in measured
    assert "probe_recommendation" not in measured


def test_extrapolated_probe_uses_tilepo_candidate_policy_after_fallback() -> None:
    result = predict_from_summary(
        SUMMARY,
        ABUNDANT_VRAM,
        admit_threshold_pct=3.0,
        target_pairs=[("mixed", 12)],
        allow_extrapolation=True,
    )
    mixed_12 = result.decision_for("mixed", 12).to_dict()
    tilepo_probe = mixed_12["probe_recommendation"]["runs"][1]
    assert tilepo_probe["system"] == "TilePO"
    assert tilepo_probe["policy"].startswith("tilepo_")


def test_target_pair_does_not_expand_to_all_workloads() -> None:
    result = predict_from_summary(
        SUMMARY,
        RTX_5090_DDR,
        admit_threshold_pct=3.0,
        target_pairs=[("mixed", 12)],
        allow_extrapolation=True,
    )
    assert result.summary["groups"] == 11
    assert result.summary["extrapolated_groups"] == 1
    try:
        result.decision_for("long_context", 12)
    except KeyError:
        pass
    else:
        raise AssertionError("TMAP expanded a mixed:12 target to long_context:12")


def test_extrapolation_requires_explicit_opt_in() -> None:
    result = predict_from_summary(
        SUMMARY,
        RTX_5090_DDR,
        admit_threshold_pct=3.0,
        target_pairs=[("mixed", 12)],
    )
    try:
        result.decision_for("mixed", 12)
    except KeyError:
        pass
    else:
        raise AssertionError("TMAP extrapolated an unseen expert budget without explicit opt-in")


def test_python_api_rejects_mixed_target_modes() -> None:
    try:
        predict_from_summary(
            SUMMARY,
            RTX_5090_DDR,
            admit_threshold_pct=3.0,
            target_pairs=[("mixed", 12)],
            target_experts=[14],
            allow_extrapolation=True,
        )
    except ValueError as exc:
        assert "cannot combine target_pairs and target_experts" in str(exc)
    else:
        raise AssertionError("TMAP Python API accepted mixed target modes")


def test_rejects_non_v0_1_real_bf16_summary_shape() -> None:
    data = json.loads(SUMMARY.read_text())
    data["schema_version"] = "unexpected_schema"
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad_summary.json"
        bad.write_text(json.dumps(data))
        try:
            predict_from_summary(bad, RTX_5090_DDR, admit_threshold_pct=3.0)
        except ValueError as exc:
            assert "V0.1 BF16 summary" in str(exc)
        else:
            raise AssertionError("TMAP accepted a non-V0.1 summary")


def test_rejects_non_bf16_manifest_rows() -> None:
    summary = json.loads(SUMMARY.read_text())
    manifest = json.loads(MANIFEST.read_text())
    manifest["runs"][0]["dtype_counts"] = {"fp8": 1}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bad_manifest = tmp_path / "tilepo_ablation_manifest.json"
        bad_manifest.write_text(json.dumps(manifest))
        bad_summary = tmp_path / "tilepo_ablation_summary.json"
        summary["source_manifest"] = str(bad_manifest)
        bad_summary.write_text(json.dumps(summary))
        try:
            predict_from_summary(bad_summary, RTX_5090_DDR, admit_threshold_pct=3.0)
        except ValueError as exc:
            assert "BF16-only" in str(exc)
        else:
            raise AssertionError("TMAP accepted a non-BF16 manifest")


def test_checked_in_reports_are_fresh() -> None:
    cases = [
        (
            HardwareProfile.from_json_path(ROOT / "TMAP" / "hardware_profiles" / "rtx5090_ddr.json"),
            ROOT / "TMAP" / "reports" / "rtx5090_ddr",
        ),
        (
            HardwareProfile.from_json_path(ROOT / "TMAP" / "hardware_profiles" / "abundant_vram_ddr.json"),
            ROOT / "TMAP" / "reports" / "abundant_vram_ddr",
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for hardware, checked_in in cases:
            result = predict_from_summary(SUMMARY_REL, hardware, admit_threshold_pct=3.0)
            generated = tmp_root / hardware.name
            summary_path, report_path = write_prediction_outputs(result, generated)
            assert summary_path.read_text() == (checked_in / "tmap_prediction_summary.json").read_text()
            assert report_path.read_text() == (checked_in / "tmap_prediction_report.md").read_text()


def test_cli_writes_json_and_markdown_reports() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "tmap_predict"),
            "--summary",
            str(SUMMARY),
            "--hardware-profile",
            str(ROOT / "TMAP" / "hardware_profiles" / "rtx5090_ddr.json"),
            "--out-dir",
            str(out_dir),
            "--admit-threshold-pct",
            "3.0",
            "--target",
            "mixed:12",
            "--allow-extrapolation",
        ]
        subprocess.run(cmd, check=True)
        summary_path = out_dir / "tmap_prediction_summary.json"
        report_path = out_dir / "tmap_prediction_report.md"
        assert summary_path.exists()
        assert report_path.exists()
        data = json.loads(summary_path.read_text())
        assert data["schema_version"] == "tmap_v0_2_prediction_v1"
        assert data["summary"]["groups"] == 11
        assert data["summary"]["extrapolated_groups"] == 1
        mixed_12 = [
            item for item in data["decisions"] if item["workload"] == "mixed" and item["experts_per_layer"] == 12
        ][0]
        assert mixed_12["evidence_mode"] == "extrapolated"
        assert mixed_12["observed_tok_gain_pct"] is None
        assert mixed_12["probe_recommendation"]["runs"][1]["system"] == "TilePO"
        assert "TMAP V0.2 Prediction Report" in report_path.read_text()


def test_cli_rejects_mixed_target_modes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "tmap_predict"),
            "--summary",
            str(SUMMARY),
            "--hardware-profile",
            str(ROOT / "TMAP" / "hardware_profiles" / "rtx5090_ddr.json"),
            "--out-dir",
            str(out_dir),
            "--target",
            "mixed:12",
            "--target-experts",
            "14",
            "--allow-extrapolation",
        ]
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        assert completed.returncode == 2
        assert "cannot combine --target and --target-experts" in completed.stderr


def test_release_packaging_includes_tmap() -> None:
    package_script = (ROOT / "scripts" / "package_release.sh").read_text()
    match = re.search(r"for dir in (?P<dirs>[^;]+); do", package_script)
    assert match is not None
    packaged_dirs = set(match.group("dirs").split())
    assert {"TMAP", "tilepo"}.issubset(packaged_dirs)


def main() -> None:
    test_predicts_v0_1_matrix()
    test_abundant_vram_profile_falls_back_more_often()
    test_transfer_bandwidth_affects_prediction()
    test_extrapolates_unseen_expert_budget_with_probe_recommendation()
    test_measured_decisions_do_not_serialize_probe_fields()
    test_extrapolated_probe_uses_tilepo_candidate_policy_after_fallback()
    test_target_pair_does_not_expand_to_all_workloads()
    test_extrapolation_requires_explicit_opt_in()
    test_python_api_rejects_mixed_target_modes()
    test_rejects_non_v0_1_real_bf16_summary_shape()
    test_rejects_non_bf16_manifest_rows()
    test_checked_in_reports_are_fresh()
    test_cli_writes_json_and_markdown_reports()
    test_cli_rejects_mixed_target_modes()
    test_release_packaging_includes_tmap()
    print("TMAP tests passed")


if __name__ == "__main__":
    main()
