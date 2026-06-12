from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    vram_capacity_gib: float
    vram_bandwidth_gbps: float
    vram_latency_ns: float
    dram_capacity_gib: float
    dram_bandwidth_gbps: float
    dram_latency_ns: float
    transfer_bandwidth_gbps: float
    transfer_latency_us: float

    def validate(self) -> None:
        if not self.name:
            raise ValueError("hardware profile name is required")
        for field in (
            "vram_capacity_gib",
            "vram_bandwidth_gbps",
            "vram_latency_ns",
            "dram_capacity_gib",
            "dram_bandwidth_gbps",
            "dram_latency_ns",
            "transfer_bandwidth_gbps",
            "transfer_latency_us",
        ):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HardwareProfile":
        profile = cls(
            name=str(data["name"]),
            vram_capacity_gib=float(data["vram_capacity_gib"]),
            vram_bandwidth_gbps=float(data["vram_bandwidth_gbps"]),
            vram_latency_ns=float(data["vram_latency_ns"]),
            dram_capacity_gib=float(data["dram_capacity_gib"]),
            dram_bandwidth_gbps=float(data["dram_bandwidth_gbps"]),
            dram_latency_ns=float(data["dram_latency_ns"]),
            transfer_bandwidth_gbps=float(data["transfer_bandwidth_gbps"]),
            transfer_latency_us=float(data["transfer_latency_us"]),
        )
        profile.validate()
        return profile

    @classmethod
    def from_json_path(cls, path: Path | str) -> "HardwareProfile":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vram_capacity_gib": self.vram_capacity_gib,
            "vram_bandwidth_gbps": self.vram_bandwidth_gbps,
            "vram_latency_ns": self.vram_latency_ns,
            "dram_capacity_gib": self.dram_capacity_gib,
            "dram_bandwidth_gbps": self.dram_bandwidth_gbps,
            "dram_latency_ns": self.dram_latency_ns,
            "transfer_bandwidth_gbps": self.transfer_bandwidth_gbps,
            "transfer_latency_us": self.transfer_latency_us,
        }


@dataclass(frozen=True)
class CandidateObservation:
    workload: str
    experts_per_layer: int
    policy: str
    async_planning: str
    system: str
    tok_per_sec: float
    p95_ms: float
    p99_ms: float
    gpu_peak_gib: float
    cpu_ram_peak_gib: float

    @property
    def policy_key(self) -> str:
        if self.policy == "kt_expert":
            return "kt_expert"
        return f"{self.policy}_async_{self.async_planning}"


@dataclass(frozen=True)
class TMAPDecision:
    workload: str
    experts_per_layer: int
    admitted_system: str
    recommended_policy: str
    fallback_policy: str
    tilepo_candidate_policy: str
    predicted_tok_gain_pct: float
    predicted_p95_reduction_pct: float
    observed_tok_gain_pct: float | None
    observed_p95_reduction_pct: float | None
    confidence: float
    dominant_factor: str
    risk: str
    explanation: str
    evidence_mode: str = "measured"
    nearest_experts: list[int] | None = None
    probe_recommended: bool = False
    probe_plan: str = ""
    probe_recommendation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        observed_tok_gain = None if self.observed_tok_gain_pct is None else round(self.observed_tok_gain_pct, 4)
        observed_p95_reduction = (
            None if self.observed_p95_reduction_pct is None else round(self.observed_p95_reduction_pct, 4)
        )
        data = {
            "workload": self.workload,
            "experts_per_layer": self.experts_per_layer,
            "admitted_system": self.admitted_system,
            "recommended_policy": self.recommended_policy,
            "fallback_policy": self.fallback_policy,
            "predicted_tok_gain_pct": round(self.predicted_tok_gain_pct, 4),
            "predicted_p95_reduction_pct": round(self.predicted_p95_reduction_pct, 4),
            "observed_tok_gain_pct": observed_tok_gain,
            "observed_p95_reduction_pct": observed_p95_reduction,
            "confidence": round(self.confidence, 4),
            "dominant_factor": self.dominant_factor,
            "risk": self.risk,
            "explanation": self.explanation,
            "evidence_mode": self.evidence_mode,
        }
        if self.evidence_mode == "extrapolated":
            data.update(
                {
                    "nearest_experts": self.nearest_experts or [],
                    "probe_recommended": self.probe_recommended,
                    "probe_plan": self.probe_plan,
                    "probe_recommendation": self.probe_recommendation or {},
                }
            )
        return data


@dataclass(frozen=True)
class PredictionResult:
    schema_version: str
    hardware: HardwareProfile
    source_summary: str
    admit_threshold_pct: float
    summary: dict[str, Any]
    decisions: list[TMAPDecision]

    def decision_for(self, workload: str, experts_per_layer: int) -> TMAPDecision:
        for decision in self.decisions:
            if decision.workload == workload and decision.experts_per_layer == experts_per_layer:
                return decision
        raise KeyError(f"no TMAP decision for {workload}/{experts_per_layer}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hardware": self.hardware.to_dict(),
            "source_summary": self.source_summary,
            "admit_threshold_pct": self.admit_threshold_pct,
            "summary": self.summary,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }
