from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tilepo.mir import RuntimeMode


class TileMEMKTPatch:
    def __init__(self, runtime: Any, mode: RuntimeMode, verify_abs_threshold: float = 1e-2) -> None:
        self.runtime = runtime
        self.mode = mode
        self.verify_abs_threshold = verify_abs_threshold
        self.metrics = {
            "shadow_evaluations": 0,
            "verify_pass_count": 0,
            "verify_fail_count": 0,
            "max_abs_error": 0.0,
            "serve_count": 0,
            "kt_fallback_count": 0,
        }

    def run(self, request: dict[str, Any], kt_call: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        if self.mode == RuntimeMode.SHADOW:
            kt_output = kt_call(request)
            self.metrics["shadow_evaluations"] += 1
            return kt_output

        if self.mode == RuntimeMode.VERIFY:
            kt_output = kt_call(request)
            tilemem_output = self.runtime.execute({**request, "kt_output": kt_output.get("output")})
            error = _max_abs_error(kt_output.get("output"), tilemem_output.get("output"))
            self.metrics["max_abs_error"] = max(float(self.metrics["max_abs_error"]), error)
            if error <= self.verify_abs_threshold:
                self.metrics["verify_pass_count"] += 1
            else:
                self.metrics["verify_fail_count"] += 1
            return kt_output

        if self.mode == RuntimeMode.SERVE:
            tilemem_output = self.runtime.execute(request)
            if tilemem_output.get("source") == "kt_fallback":
                self.metrics["kt_fallback_count"] += 1
                return kt_call(request)
            self.metrics["serve_count"] += 1
            return tilemem_output

        raise ValueError(f"unsupported runtime mode {self.mode}")


def _max_abs_error(a: Any, b: Any) -> float:
    if a is None or b is None:
        return float("inf")
    flat_a = _flatten(a)
    flat_b = _flatten(b)
    if len(flat_a) != len(flat_b):
        return float("inf")
    if not flat_a:
        return 0.0
    return max(abs(float(x) - float(y)) for x, y in zip(flat_a, flat_b))


def _flatten(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return []

