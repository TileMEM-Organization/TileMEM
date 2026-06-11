from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProfileSummary:
    workload_label: str
    routing_entropy: float
    hotset_coverage: dict[str, float]
    expert_hits: dict[str, int]
    layer_hotness: dict[str, int]
    expert_churn_rate: float
    source_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_label": self.workload_label,
            "routing_entropy": self.routing_entropy,
            "hotset_coverage": self.hotset_coverage,
            "expert_hits": self.expert_hits,
            "layer_hotness": self.layer_hotness,
            "expert_churn_rate": self.expert_churn_rate,
            "source_files": self.source_files,
        }


def profile_directory(profile_dir: Path | str, workload_label: str, expert_budgets: list[int]) -> ProfileSummary:
    profile_dir = Path(profile_dir)
    rows = _load_rows(profile_dir)
    filtered = [row for row in rows if row.get("workload", workload_label) == workload_label]
    if not filtered:
        filtered = rows
    expert_hits: Counter[str] = Counter()
    layer_hotness: Counter[str] = Counter()
    request_experts: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(filtered):
        count = int(row.get("count", row.get("hits", 1)))
        routes = row.get("route_trace")
        if routes:
            for route in routes:
                key = f"{int(route['layer'])}:{int(route['expert'])}"
                expert_hits[key] += int(route.get("count", 1))
                layer_hotness[str(int(route["layer"]))] += int(route.get("count", 1))
                request_experts[str(row.get("request_id", index))].add(key)
        elif "layer" in row and "expert" in row:
            key = f"{int(row['layer'])}:{int(row['expert'])}"
            expert_hits[key] += count
            layer_hotness[str(int(row["layer"]))] += count
            request_experts[str(row.get("request_id", index))].add(key)
    total_hits = sum(expert_hits.values())
    entropy = _normalized_entropy(list(expert_hits.values()))
    hotset_coverage = {
        str(budget): _coverage_for_budget(expert_hits, max(1, int(budget)))
        for budget in sorted(set(expert_budgets))
    }
    churn = _churn_rate([request_experts[key] for key in sorted(request_experts)])
    return ProfileSummary(
        workload_label=workload_label,
        routing_entropy=entropy if total_hits else 0.0,
        hotset_coverage=hotset_coverage,
        expert_hits=dict(sorted(expert_hits.items())),
        layer_hotness=dict(sorted(layer_hotness.items())),
        expert_churn_rate=churn,
        source_files=[str(path) for path in sorted(profile_dir.glob("*")) if path.is_file()],
    )


def _load_rows(profile_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(profile_dir.glob("**/*")):
        if not path.is_file() or path.suffix not in {".jsonl", ".log", ".json"}:
            continue
        text = path.read_text(errors="ignore")
        if path.suffix == ".json":
            data = json.loads(text)
            if isinstance(data, dict) and "runs" in data:
                rows.extend(data["runs"])
            elif isinstance(data, list):
                rows.extend(data)
            elif isinstance(data, dict):
                rows.append(data)
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts:
        p = count / total
        entropy -= p * math.log(p)
    return entropy / math.log(len(counts))


def _coverage_for_budget(expert_hits: Counter[str], budget: int) -> float:
    total = sum(expert_hits.values())
    if total <= 0:
        return 0.0
    covered = sum(count for _, count in expert_hits.most_common(budget))
    return covered / total


def _churn_rate(request_sets: list[set[str]]) -> float:
    if len(request_sets) <= 1:
        return 0.0
    churns = []
    for prev, curr in zip(request_sets, request_sets[1:]):
        union = prev | curr
        if not union:
            churns.append(0.0)
            continue
        churns.append(1.0 - len(prev & curr) / len(union))
    return sum(churns) / len(churns)

