from __future__ import annotations

import math
from typing import Any


REQUIRED_MANIFEST_FIELDS = {
    "tile_offsets",
    "tile_bytes",
    "tile_dtype_map",
    "scale_offsets",
    "gpu_hot_tiles",
    "fallback_chain",
    "backend_priority",
    "runtime_gates",
    "checksum",
}


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing:
        raise ValueError("manifest missing fields: " + ", ".join(missing))
    return manifest


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    if not a or not b:
        return []
    cols = len(b[0])
    inner = len(b)
    if any(len(row) != inner for row in a):
        raise ValueError("left matrix has incompatible shape")
    if any(len(row) != cols for row in b):
        raise ValueError("right matrix is ragged")
    out: list[list[float]] = []
    for row in a:
        out_row = []
        for col in range(cols):
            out_row.append(sum(float(row[k]) * float(b[k][col]) for k in range(inner)))
        out.append(out_row)
    return out


def quantize_value(value: float, dtype: str) -> float:
    if dtype == "bf16":
        return float(value)
    if dtype == "fp8":
        return round(float(value) * 16.0) / 16.0
    if dtype == "mxfp4":
        return round(float(value) * 4.0) / 4.0
    raise ValueError(f"unsupported dtype {dtype}")


def quantized_matmul(a: list[list[float]], b: list[list[float]], dtype: str) -> list[list[float]]:
    qa = [[quantize_value(x, dtype) for x in row] for row in a]
    qb = [[quantize_value(x, dtype) for x in row] for row in b]
    return matmul(qa, qb)


def routed_moe(
    hidden: list[float],
    gate_up: list[list[list[float]]],
    down: list[list[list[float]]],
    expert_ids: list[int],
    router_scores: list[float],
    dtype: str,
) -> list[float]:
    if len(expert_ids) != len(router_scores):
        raise ValueError("expert_ids and router_scores must have the same length")
    if not hidden:
        return []
    if not gate_up or not down:
        raise ValueError("gate_up and down weights must not be empty")
    hidden_values = [quantize_value(x, dtype) for x in hidden]
    output = [0.0 for _ in hidden_values]
    for expert, score in zip(expert_ids, router_scores):
        expert_index = int(expert)
        if expert_index < 0 or expert_index >= len(gate_up) or expert_index >= len(down):
            raise ValueError(f"expert id out of range: {expert_index}")
        gate_up_matrix = _quantize_matrix(gate_up[expert_index], dtype)
        down_matrix = _quantize_matrix(down[expert_index], dtype)
        fused = _vec_matmul(hidden_values, gate_up_matrix)
        if len(fused) % 2 != 0:
            raise ValueError("gate_up projection width must be even")
        inter = len(fused) // 2
        activated = [_silu(fused[i]) * fused[i + inter] for i in range(inter)]
        expert_out = _vec_matmul(activated, down_matrix)
        if len(expert_out) != len(output):
            raise ValueError("down projection output width must match hidden size")
        for index, value in enumerate(expert_out):
            output[index] += float(score) * value
    return output


def _quantize_matrix(matrix: list[list[float]], dtype: str) -> list[list[float]]:
    return [[quantize_value(value, dtype) for value in row] for row in matrix]


def _vec_matmul(vector: list[float], matrix: list[list[float]]) -> list[float]:
    if len(vector) != len(matrix):
        raise ValueError("vector and matrix have incompatible shape")
    if not matrix:
        return []
    cols = len(matrix[0])
    if any(len(row) != cols for row in matrix):
        raise ValueError("matrix is ragged")
    return [sum(float(vector[row]) * float(matrix[row][col]) for row in range(len(vector))) for col in range(cols)]


def _silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))
