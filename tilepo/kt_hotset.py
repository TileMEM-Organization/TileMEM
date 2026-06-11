"""KT hotset export helpers for TilePO."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

import torch


EXPECTED_SHAPE = (1, 16, 64)
RANK_BANDED = "rank_banded"


def parse_expert_list(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        experts = [int(item) for item in value.split(",") if item]
    else:
        experts = [int(item) for item in value]
    if not experts:
        raise ValueError("experts must contain at least one budget")
    return experts


def export_kt_hotset(
    source_path: Path | str,
    output_path: Path | str,
    summary_output_path: Path | str | None = None,
    *,
    workload: str,
    experts: Iterable[int],
    strategy: str = RANK_BANDED,
) -> dict:
    if strategy != RANK_BANDED:
        raise ValueError(f"unsupported KT hotset strategy: {strategy}")

    source = Path(source_path)
    output = Path(output_path)
    expert_budgets = parse_expert_list(experts)
    source_dict = _torch_load(source)
    if not isinstance(source_dict, dict):
        raise ValueError(f"{source} must contain a torch-saved dict")
    if "logical_count" not in source_dict:
        raise ValueError(f"{source} is missing logical_count")

    source_counts = _normalize_logical_count(source_dict["logical_count"], source)
    ranked_counts, bands = rank_banded_logical_count(source_counts)
    payload = {
        "logical_count": ranked_counts,
        "tilemem_hotset_source": str(source),
        "tilemem_hotset_workload": workload,
        "tilemem_hotset_strategy": strategy,
        "tilemem_hotset_experts": expert_budgets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    summary = {
        "shape": list(ranked_counts.shape),
        "dtype": str(ranked_counts.dtype),
        "workload": workload,
        "strategy": strategy,
        "source_path": str(source),
        "output_path": str(output),
        "experts": expert_budgets,
        "per_layer_top4": bands[4],
        "per_layer_top8": bands[8],
        "per_layer_top16": bands[16],
    }
    if summary_output_path is not None:
        summary_output = Path(summary_output_path)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def rank_banded_logical_count(logical_count: torch.Tensor) -> tuple[torch.Tensor, dict[int, list[list[int]]]]:
    counts = _normalize_logical_count(logical_count, Path("<logical_count>"))
    ranked = torch.empty(EXPECTED_SHAPE, dtype=torch.float32, device="cpu")
    top_by_budget: dict[int, list[list[int]]] = {4: [], 8: [], 16: []}

    for layer in range(EXPECTED_SHAPE[1]):
        order = _rank_layer(counts[0, layer])
        top_by_budget[4].append(order[:4])
        top_by_budget[8].append(order[:8])
        top_by_budget[16].append(order[:16])
        for rank, expert in enumerate(order):
            ranked[0, layer, expert] = _rank_banded_score(rank, layer, expert)

    return ranked, top_by_budget


def _normalize_logical_count(value: object, source: Path) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"{source} logical_count must be a torch tensor")
    tensor = value.detach().cpu().to(torch.float32)
    if tensor.ndim == 2 and tuple(tensor.shape) == EXPECTED_SHAPE[1:]:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3 and tuple(tensor.shape[1:]) == EXPECTED_SHAPE[1:] and tensor.shape[0] > 1:
        tensor = tensor.sum(dim=0, keepdim=True)
    if tuple(tensor.shape) != EXPECTED_SHAPE:
        raise ValueError(f"{source} logical_count must have shape {EXPECTED_SHAPE}, got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{source} logical_count must contain only finite values")
    return tensor


def _rank_layer(layer_counts: torch.Tensor) -> list[int]:
    return sorted(range(layer_counts.numel()), key=lambda expert: (-float(layer_counts[expert]), expert))


def _rank_banded_score(rank: int, layer: int, expert: int) -> float:
    if rank < 4:
        band_base = 3.0
    elif rank < 8:
        band_base = 2.0
    elif rank < 16:
        band_base = 1.0
    else:
        band_base = 0.0

    rank_bonus = (EXPECTED_SHAPE[2] - rank) * 1.0e-4
    layer_bonus = (EXPECTED_SHAPE[1] - 1 - layer) * 1.0e-7
    expert_bonus = (EXPECTED_SHAPE[2] - 1 - expert) * 1.0e-9
    return band_base + rank_bonus + layer_bonus + expert_bonus


def _torch_load(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
