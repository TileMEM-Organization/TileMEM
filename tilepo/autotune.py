from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .mir import Backend, DeploymentMode, TileDType
from .profiling import ProfileSummary


@dataclass(frozen=True)
class DeploymentPlan:
    name: str
    deployment_mode: DeploymentMode | str
    gpu_cache_budget_gib: float
    experts_per_layer: int
    intermediate_tile: int
    hidden_tile: int
    dtype_policy: str
    hotset_coverage_threshold: float
    backend: Backend | str
    prewarm_policy: str
    miss_policy: str
    expected_hotset_coverage: float
    pareto_rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "deployment_mode": self.deployment_mode.value if hasattr(self.deployment_mode, "value") else self.deployment_mode,
            "gpu_cache_budget_gib": self.gpu_cache_budget_gib,
            "experts_per_layer": self.experts_per_layer,
            "intermediate_tile": self.intermediate_tile,
            "hidden_tile": self.hidden_tile,
            "dtype_policy": self.dtype_policy,
            "hotset_coverage_threshold": self.hotset_coverage_threshold,
            "backend": self.backend.value if hasattr(self.backend, "value") else self.backend,
            "prewarm_policy": self.prewarm_policy,
            "miss_policy": self.miss_policy,
            "expected_hotset_coverage": self.expected_hotset_coverage,
            "pareto_rank": self.pareto_rank,
        }


def autotune_from_profile(summary: ProfileSummary, gpu_cache_budget_gib: float) -> list[DeploymentPlan]:
    budgets = sorted((int(k), v) for k, v in summary.hotset_coverage.items())
    best_budget = budgets[-1][0] if budgets else 1
    tight_budget = budgets[0][0] if budgets else 1
    speed_budget = _first_budget_at_or_above(budgets, 0.80) or best_budget
    balanced_budget = _first_budget_at_or_above(budgets, 0.70) or speed_budget
    memory_budget = tight_budget
    safe_budget = min(speed_budget, best_budget)
    high_churn = summary.expert_churn_rate > 0.65 or summary.routing_entropy > 0.85

    plans = [
        DeploymentPlan(
            "speed_mode",
            DeploymentMode.SPEED,
            gpu_cache_budget_gib,
            speed_budget,
            256,
            128,
            TileDType.MXFP4.value,
            0.80,
            Backend.CUDA,
            "hotset",
            "fallback",
            _coverage(budgets, speed_budget),
            0,
        ),
        DeploymentPlan(
            "memory_mode",
            DeploymentMode.MEMORY,
            max(0.5, gpu_cache_budget_gib * 0.55),
            memory_budget,
            128,
            64,
            TileDType.FP8.value,
            0.55,
            Backend.CUDA,
            "lazy",
            "fallback",
            _coverage(budgets, memory_budget),
            0,
        ),
        DeploymentPlan(
            "balanced_mode",
            DeploymentMode.BALANCED,
            max(1.0, gpu_cache_budget_gib * 0.75),
            balanced_budget,
            192,
            96,
            TileDType.FP8.value if high_churn else TileDType.MXFP4.value,
            0.70,
            Backend.CUDA,
            "hotset",
            "fallback",
            _coverage(budgets, balanced_budget),
            0,
        ),
        DeploymentPlan(
            "safe_mode",
            DeploymentMode.SAFE,
            gpu_cache_budget_gib,
            safe_budget,
            128,
            64,
            TileDType.BF16.value,
            0.0,
            Backend.TILELANG,
            "none",
            "kt_fallback",
            _coverage(budgets, safe_budget),
            0,
        ),
    ]
    return _rank_pareto(plans)


def select_runtime_plan(
    plans: list[DeploymentPlan],
    locality_pass: bool,
    correctness_pass: bool,
    memory_pressure_high: bool,
) -> DeploymentPlan:
    by_name = {plan.name: plan for plan in plans}
    if not correctness_pass:
        return DeploymentPlan(
            "kt_fallback",
            "safe",
            0.0,
            0,
            0,
            0,
            "kt",
            0.0,
            Backend.KT_FALLBACK,
            "none",
            "kt_fallback",
            0.0,
            99,
        )
    if not locality_pass:
        return by_name["safe_mode"]
    if memory_pressure_high:
        return by_name["balanced_mode"]
    return by_name["speed_mode"]


def _first_budget_at_or_above(budgets: list[tuple[int, float]], threshold: float) -> int | None:
    for budget, coverage in budgets:
        if coverage >= threshold:
            return budget
    return None


def _coverage(budgets: list[tuple[int, float]], budget: int) -> float:
    for candidate, coverage in budgets:
        if candidate == budget:
            return coverage
    return 0.0


def _rank_pareto(plans: list[DeploymentPlan]) -> list[DeploymentPlan]:
    ranked = []
    for plan in plans:
        # The four release modes optimize different axes, so none should be
        # removed by a cross-objective dominance check at this stage.
        rank = 0
        ranked.append(
            DeploymentPlan(
                plan.name,
                plan.deployment_mode,
                plan.gpu_cache_budget_gib,
                plan.experts_per_layer,
                plan.intermediate_tile,
                plan.hidden_tile,
                plan.dtype_policy,
                plan.hotset_coverage_threshold,
                plan.backend,
                plan.prewarm_policy,
                plan.miss_policy,
                plan.expected_hotset_coverage,
                rank,
            )
        )
    order = {"speed_mode": 0, "memory_mode": 1, "balanced_mode": 2, "safe_mode": 3}
    return sorted(ranked, key=lambda plan: order[plan.name])
