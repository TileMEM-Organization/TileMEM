from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from .schema import CandidateObservation, HardwareProfile, PredictionResult, TMAPDecision


SCHEMA_VERSION = "tmap_v0_2_prediction_v1"


def predict_from_summary(
    summary_path: Path | str,
    hardware: HardwareProfile,
    *,
    admit_threshold_pct: float = 3.0,
    target_experts: list[int] | None = None,
    target_pairs: list[tuple[str, int]] | None = None,
    allow_extrapolation: bool = False,
) -> PredictionResult:
    hardware.validate()
    summary_path = Path(summary_path)
    groups = _load_observations(summary_path)
    grouped: dict[tuple[str, int], list[CandidateObservation]] = defaultdict(list)
    for observation in groups:
        grouped[(observation.workload, observation.experts_per_layer)].append(observation)

    decisions = []
    rank_hits = 0
    for key in sorted(grouped):
        observations = grouped[key]
        kt = _single_kt(observations)
        candidates = [item for item in observations if item.policy != "kt_expert"]
        if not candidates:
            continue
        observed_best = max(candidates, key=lambda item: item.tok_per_sec)
        scored = [_score_candidate(kt, candidate, hardware) for candidate in candidates]
        selected = max(scored, key=lambda item: item["policy_score"])
        decision = _decision_from_score(
            kt=kt,
            selected=selected,
            observed_best=observed_best,
            admit_threshold_pct=admit_threshold_pct,
        )
        if selected["candidate"].policy_key == observed_best.policy_key:
            rank_hits += 1
        decisions.append(decision)

    requested_targets = _normalize_targets(target_experts=target_experts, target_pairs=target_pairs, decisions=decisions)
    if requested_targets and allow_extrapolation:
        measured_keys = {(decision.workload, decision.experts_per_layer) for decision in decisions}
        for workload, experts_per_layer in requested_targets:
            workload_decisions = [decision for decision in decisions if decision.workload == workload]
            key = (workload, experts_per_layer)
            if key not in measured_keys:
                decisions.append(
                    _extrapolate_decision(
                        workload_decisions=workload_decisions,
                        experts_per_layer=experts_per_layer,
                        admit_threshold_pct=admit_threshold_pct,
                    )
                )
        decisions.sort(key=lambda item: (item.workload, item.experts_per_layer, item.evidence_mode))

    admit_count = sum(1 for item in decisions if item.admitted_system == "TilePO")
    fallback_count = len(decisions) - admit_count
    measured_count = sum(1 for item in decisions if item.evidence_mode == "measured")
    extrapolated_count = sum(1 for item in decisions if item.evidence_mode == "extrapolated")
    summary = {
        "groups": len(decisions),
        "measured_groups": measured_count,
        "extrapolated_groups": extrapolated_count,
        "admit_tilepo": admit_count,
        "fallback_kt": fallback_count,
        "rank_accuracy": round(rank_hits / measured_count, 4) if measured_count else 0.0,
        "mean_predicted_tok_gain_pct": round(mean([item.predicted_tok_gain_pct for item in decisions]), 4)
        if decisions
        else 0.0,
        "mean_predicted_p95_reduction_pct": round(mean([item.predicted_p95_reduction_pct for item in decisions]), 4)
        if decisions
        else 0.0,
    }
    return PredictionResult(
        schema_version=SCHEMA_VERSION,
        hardware=hardware,
        source_summary=str(summary_path),
        admit_threshold_pct=admit_threshold_pct,
        summary=summary,
        decisions=decisions,
    )


def _normalize_targets(
    *,
    target_experts: list[int] | None,
    target_pairs: list[tuple[str, int]] | None,
    decisions: list[TMAPDecision],
) -> list[tuple[str, int]]:
    if target_pairs and target_experts:
        raise ValueError("cannot combine target_pairs and target_experts; use one target mode per run")
    if target_pairs:
        return sorted(set(target_pairs))
    if not target_experts:
        return []
    workloads = sorted({decision.workload for decision in decisions})
    return [(workload, expert) for workload in workloads for expert in sorted(set(target_experts))]


def _load_observations(summary_path: Path) -> list[CandidateObservation]:
    data = json.loads(summary_path.read_text())
    _validate_v0_1_summary(data, summary_path)
    observations = []
    for group in data.get("groups", []):
        metrics = group.get("metrics", {})
        observations.append(
            CandidateObservation(
                workload=str(group["workload"]),
                experts_per_layer=int(group["experts_per_layer"]),
                policy=str(group["policy"]),
                async_planning=str(group.get("async_planning", "off")),
                system=str(group["system"]),
                tok_per_sec=_median(metrics, "tok_per_sec"),
                p95_ms=_median(metrics, "p95_ms"),
                p99_ms=_median(metrics, "p99_ms"),
                gpu_peak_gib=_median(metrics, "gpu_peak_gib"),
                cpu_ram_peak_gib=_median(metrics, "cpu_ram_peak_gib"),
            )
        )
    if not observations:
        raise ValueError(f"TMAP summary has no groups: {summary_path}")
    return observations


def _validate_v0_1_summary(data: dict[str, Any], summary_path: Path) -> None:
    requested = data.get("requested", {})
    policies = set(requested.get("policies", []))
    required_policies = {"kt_expert", "tilepo_coarse", "tilepo_fine", "tilepo_hybrid"}
    if data.get("schema_version") != "tilepo_ablation_report_v1":
        raise ValueError(f"TMAP requires a V0.1 BF16 summary shape: {summary_path}")
    if requested.get("require_real") is not True:
        raise ValueError(f"TMAP requires a V0.1 BF16 summary with real evidence: {summary_path}")
    if not required_policies.issubset(policies):
        raise ValueError(f"TMAP requires V0.1 BF16 policy coverage: {summary_path}")
    manifest_path = _resolve_manifest_path(summary_path, str(data.get("source_manifest", "")))
    manifest = json.loads(manifest_path.read_text())
    _validate_v0_1_bf16_manifest(manifest, manifest_path, requested)


def _resolve_manifest_path(summary_path: Path, source_manifest: str) -> Path:
    if not source_manifest:
        raise ValueError(f"TMAP requires a V0.1 BF16 source manifest: {summary_path}")
    source = Path(source_manifest)
    candidates = [source] if source.is_absolute() else [
        Path.cwd() / source,
        summary_path.parent / source,
        summary_path.parent / source.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ValueError(f"TMAP cannot find V0.1 BF16 source manifest: {source_manifest}")


def _validate_v0_1_bf16_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    requested: dict[str, Any],
) -> None:
    if manifest.get("schema_version") != "tilepo_public_manifest_v1":
        raise ValueError(f"TMAP requires the V0.1 BF16 public manifest: {manifest_path}")
    if manifest.get("actual_result_rows") != manifest.get("expected_result_rows"):
        raise ValueError(f"TMAP requires a complete V0.1 BF16 public manifest: {manifest_path}")

    expected_shape = {
        "selected_workloads": sorted(requested.get("workloads", [])),
        "selected_experts": sorted(requested.get("experts", [])),
        "selected_policies": sorted(requested.get("policies", [])),
        "selected_async_modes": sorted(requested.get("async_modes", [])),
    }
    for field, expected in expected_shape.items():
        if sorted(manifest.get(field, [])) != expected:
            raise ValueError(f"TMAP requires manifest/summary V0.1 BF16 shape agreement: {field}")

    runs = manifest.get("runs", [])
    if len(runs) != manifest.get("actual_result_rows"):
        raise ValueError(f"TMAP requires all V0.1 BF16 rows in the public manifest: {manifest_path}")
    for index, run in enumerate(runs):
        if run.get("evidence_level") != "real" or run.get("simulated") is not False:
            raise ValueError(f"TMAP requires real BF16 evidence rows; row {index} is not real")
        dtype_counts = run.get("dtype_counts", {})
        if set(dtype_counts) != {"bf16"} or float(dtype_counts.get("bf16", 0.0)) <= 0.0:
            raise ValueError(f"TMAP requires BF16-only calibration rows; row {index} has {dtype_counts}")


def _median(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, {}).get("median")
    if value is None:
        raise ValueError(f"missing metric median: {key}")
    return float(value)


def _single_kt(observations: list[CandidateObservation]) -> CandidateObservation:
    kt = [item for item in observations if item.policy == "kt_expert"]
    if len(kt) != 1:
        raise ValueError(f"expected one KT baseline, got {len(kt)}")
    return kt[0]


def _score_candidate(
    kt: CandidateObservation,
    candidate: CandidateObservation,
    hardware: HardwareProfile,
) -> dict[str, Any]:
    observed_tok_gain = (candidate.tok_per_sec / kt.tok_per_sec - 1.0) * 100.0
    observed_p95_reduction = (1.0 - candidate.p95_ms / kt.p95_ms) * 100.0

    pressure = _capacity_pressure(kt.gpu_peak_gib, hardware.vram_capacity_gib)
    bandwidth_gap = _bandwidth_gap(hardware)
    latency_gap = _latency_gap(hardware)
    memory_modifier = pressure * (0.72 + 0.18 * bandwidth_gap + 0.10 * latency_gap)
    async_bonus = 1.04 if candidate.async_planning == "on" else 1.0
    policy_risk = _policy_risk(candidate.policy, candidate.async_planning)
    memory_delta_bonus = _memory_delta_bonus(kt, candidate, hardware)
    transfer_penalty = _transfer_penalty(candidate, hardware)

    predicted_tok_gain = observed_tok_gain * memory_modifier * async_bonus
    predicted_tok_gain += 0.12 * observed_p95_reduction + memory_delta_bonus
    predicted_tok_gain -= policy_risk + transfer_penalty

    predicted_p95_reduction = observed_p95_reduction * (0.62 + 0.38 * pressure)
    predicted_p95_reduction += 0.04 * max(0.0, observed_tok_gain)
    predicted_p95_reduction -= 0.5 * policy_risk

    confidence = _confidence(
        pressure=pressure,
        observed_tok_gain=observed_tok_gain,
        observed_p95_reduction=observed_p95_reduction,
        predicted_tok_gain=predicted_tok_gain,
    )
    return {
        "candidate": candidate,
        "observed_tok_gain": observed_tok_gain,
        "observed_p95_reduction": observed_p95_reduction,
        "predicted_tok_gain": predicted_tok_gain,
        "predicted_p95_reduction": predicted_p95_reduction,
        "confidence": confidence,
        "dominant_factor": _dominant_factor(pressure, observed_tok_gain, observed_p95_reduction, memory_delta_bonus),
        "risk": _risk_label(policy_risk, transfer_penalty, candidate),
        "policy_score": predicted_tok_gain + 0.20 * predicted_p95_reduction + 1.25 * confidence,
    }


def _capacity_pressure(gpu_peak_gib: float, vram_capacity_gib: float) -> float:
    # TMAP v0.1 intentionally models a two-tier VRAM/DRAM system.  Below ~8% of
    # VRAM use, KT already has enough fast-memory headroom and TilePO should be
    # conservative.  Between 8% and 16%, pressure ramps toward full admission.
    ratio = gpu_peak_gib / vram_capacity_gib
    return _clamp((ratio - 0.08) / 0.08, 0.0, 1.0)


def _bandwidth_gap(hardware: HardwareProfile) -> float:
    return _clamp(math.log2(hardware.vram_bandwidth_gbps / hardware.dram_bandwidth_gbps) / 5.0, 0.0, 1.0)


def _latency_gap(hardware: HardwareProfile) -> float:
    dram_ns = hardware.dram_latency_ns
    vram_ns = hardware.vram_latency_ns
    return _clamp(math.log10(max(1.0, dram_ns / vram_ns)) / 3.0, 0.0, 1.0)


def _policy_risk(policy: str, async_planning: str) -> float:
    if policy == "tilepo_coarse":
        base = 0.45
    elif policy == "tilepo_hybrid":
        base = 0.85
    elif policy == "tilepo_fine":
        base = 1.20
    else:
        base = 1.0
    if async_planning == "on":
        base *= 0.82
    return base


def _memory_delta_bonus(kt: CandidateObservation, candidate: CandidateObservation, hardware: HardwareProfile) -> float:
    delta = kt.gpu_peak_gib - candidate.gpu_peak_gib
    if delta <= 0:
        return -0.2 * min(2.5, abs(delta))
    return min(2.5, delta / hardware.vram_capacity_gib * 100.0)


def _transfer_penalty(candidate: CandidateObservation, hardware: HardwareProfile) -> float:
    transfer_ratio = candidate.cpu_ram_peak_gib / max(0.001, hardware.dram_capacity_gib)
    latency_penalty = _clamp(hardware.transfer_latency_us / 50.0, 0.0, 1.0)
    bandwidth_penalty = _clamp((64.0 / hardware.transfer_bandwidth_gbps - 1.0) / 8.0, 0.0, 4.0)
    return transfer_ratio * 4.0 + latency_penalty * 0.25 + bandwidth_penalty


def _confidence(
    *,
    pressure: float,
    observed_tok_gain: float,
    observed_p95_reduction: float,
    predicted_tok_gain: float,
) -> float:
    signal = 0.45 * pressure
    signal += 0.25 * _clamp(observed_tok_gain / 25.0, 0.0, 1.0)
    signal += 0.20 * _clamp(observed_p95_reduction / 20.0, 0.0, 1.0)
    signal += 0.10 * _clamp(predicted_tok_gain / 15.0, 0.0, 1.0)
    return round(_clamp(signal, 0.05, 0.95), 4)


def _dominant_factor(
    pressure: float,
    observed_tok_gain: float,
    observed_p95_reduction: float,
    memory_delta_bonus: float,
) -> str:
    if pressure < 0.15:
        return "low_vram_pressure"
    if memory_delta_bonus > 0.7:
        return "vram_pressure_reduction"
    if observed_p95_reduction > observed_tok_gain:
        return "latency_tail_reduction"
    return "observed_tilepo_gain"


def _risk_label(policy_risk: float, transfer_penalty: float, candidate: CandidateObservation) -> str:
    if transfer_penalty > 0.8:
        return "transfer_overhead"
    if candidate.policy == "tilepo_fine" and policy_risk >= 1.0:
        return "fragmentation_overhead"
    if candidate.async_planning == "off":
        return "exposed_planning"
    return "low"


def _decision_from_score(
    *,
    kt: CandidateObservation,
    selected: dict[str, Any],
    observed_best: CandidateObservation,
    admit_threshold_pct: float,
) -> TMAPDecision:
    candidate: CandidateObservation = selected["candidate"]
    predicted_tok_gain = float(selected["predicted_tok_gain"])
    predicted_p95_reduction = float(selected["predicted_p95_reduction"])
    admitted = predicted_tok_gain >= admit_threshold_pct and selected["confidence"] >= 0.35
    admitted_system = "TilePO" if admitted else "KT"
    policy = candidate.policy_key if admitted else "kt_expert"
    explanation = (
        f"TMAP compares KT with {candidate.policy_key} using V0.1 observed gain "
        f"({selected['observed_tok_gain']:.2f}% tok/s, {selected['observed_p95_reduction']:.2f}% p95) "
        f"and two-tier VRAM/DRAM hardware modifiers."
    )
    if not admitted:
        explanation += " Predicted gain is below admission threshold, so fallback KT is recommended."

    return TMAPDecision(
        workload=kt.workload,
        experts_per_layer=kt.experts_per_layer,
        admitted_system=admitted_system,
        recommended_policy=policy,
        fallback_policy="kt_expert",
        tilepo_candidate_policy=candidate.policy_key,
        predicted_tok_gain_pct=predicted_tok_gain,
        predicted_p95_reduction_pct=predicted_p95_reduction,
        observed_tok_gain_pct=float(selected["observed_tok_gain"]),
        observed_p95_reduction_pct=float(selected["observed_p95_reduction"]),
        confidence=float(selected["confidence"]),
        dominant_factor=str(selected["dominant_factor"]),
        risk=str(selected["risk"]),
        explanation=explanation,
    )


def _extrapolate_decision(
    *,
    workload_decisions: list[TMAPDecision],
    experts_per_layer: int,
    admit_threshold_pct: float,
) -> TMAPDecision:
    if len(workload_decisions) < 2:
        raise ValueError(f"TMAP needs at least two measured points to extrapolate {experts_per_layer}")

    nearest = sorted(workload_decisions, key=lambda item: abs(item.experts_per_layer - experts_per_layer))[:2]
    nearest.sort(key=lambda item: item.experts_per_layer)
    lower, upper = nearest
    predicted_tok_gain = _linear_extrapolate(
        lower.experts_per_layer,
        lower.predicted_tok_gain_pct,
        upper.experts_per_layer,
        upper.predicted_tok_gain_pct,
        experts_per_layer,
    )
    predicted_p95_reduction = _linear_extrapolate(
        lower.experts_per_layer,
        lower.predicted_p95_reduction_pct,
        upper.experts_per_layer,
        upper.predicted_p95_reduction_pct,
        experts_per_layer,
    )
    distance = min(abs(experts_per_layer - lower.experts_per_layer), abs(experts_per_layer - upper.experts_per_layer))
    confidence = round(_clamp(min(lower.confidence, upper.confidence) * (0.72 ** max(1, distance / 2.0)), 0.05, 0.65), 4)

    admitted = predicted_tok_gain >= admit_threshold_pct and predicted_p95_reduction >= 0.0 and confidence >= 0.45
    selected_policy = upper.tilepo_candidate_policy if upper.experts_per_layer <= experts_per_layer else lower.tilepo_candidate_policy
    recommended_policy = selected_policy if admitted else "kt_expert"
    explanation = (
        f"TMAP extrapolated {workload_decisions[0].workload}/{experts_per_layer} from measured "
        f"experts {lower.experts_per_layer} and {upper.experts_per_layer}. This is a quick-planning "
        "estimate, not a measured calibration point."
    )
    if not admitted:
        explanation += " Fallback KT is recommended until a short probe confirms TilePO."

    probe_recommendation = {
        "reason": "extrapolated_expert_budget",
        "request_count": 5,
        "repeats": 1,
        "runs": [
            {
                "system": "KT",
                "policy": "kt_expert",
                "workload": workload_decisions[0].workload,
                "experts_per_layer": experts_per_layer,
            },
            {
                "system": "TilePO",
                "policy": selected_policy,
                "workload": workload_decisions[0].workload,
                "experts_per_layer": experts_per_layer,
            },
        ],
    }

    return TMAPDecision(
        workload=workload_decisions[0].workload,
        experts_per_layer=experts_per_layer,
        admitted_system="TilePO" if admitted else "KT",
        recommended_policy=recommended_policy,
        fallback_policy="kt_expert",
        tilepo_candidate_policy=selected_policy,
        predicted_tok_gain_pct=predicted_tok_gain,
        predicted_p95_reduction_pct=predicted_p95_reduction,
        observed_tok_gain_pct=None,
        observed_p95_reduction_pct=None,
        confidence=confidence,
        dominant_factor="expert_budget_extrapolation",
        risk="extrapolation_uncertainty",
        explanation=explanation,
        evidence_mode="extrapolated",
        nearest_experts=[lower.experts_per_layer, upper.experts_per_layer],
        probe_recommended=True,
        probe_plan=f"Run KT {experts_per_layer} and TilePO {experts_per_layer} with request_count=5, repeats=1 before production.",
        probe_recommendation=probe_recommendation,
    )


def _linear_extrapolate(x0: int, y0: float, x1: int, y1: float, target: int) -> float:
    if x0 == x1:
        return y0
    slope = (y1 - y0) / (x1 - x0)
    return y1 + slope * (target - x1)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
