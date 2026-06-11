from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationResult:
    dtype: str
    max_abs_error: float
    max_rel_error: float
    pass_gate: bool | None = None
    fallback_dtype: str | None = None


def calibration_gate(
    result: CalibrationResult,
    abs_threshold: float,
    rel_threshold: float,
) -> CalibrationResult:
    passed = result.max_abs_error <= abs_threshold and result.max_rel_error <= rel_threshold
    fallback = None
    if not passed:
        if result.dtype == "mxfp4":
            fallback = "fp8"
        elif result.dtype == "fp8":
            fallback = "bf16"
        else:
            fallback = "kt"
    return CalibrationResult(
        dtype=result.dtype,
        max_abs_error=result.max_abs_error,
        max_rel_error=result.max_rel_error,
        pass_gate=passed,
        fallback_dtype=fallback,
    )

