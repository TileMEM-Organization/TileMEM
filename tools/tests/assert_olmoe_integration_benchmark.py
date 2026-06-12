#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


EXPECTED_CASES = {
    "bf16_fp32_baseline",
    "fp8_external_cuda_packed",
    "fp6_external_cuda_packed",
    "fp4_external_cuda_packed",
}

EXPECTED_KERNELS = {
    "fp8": ("kernels/gemm_fp8.cu", "tilemem_launch_gemm_fp8"),
    "fp6": ("kernels/gemm_fp6.cu", "tilemem_launch_gemm_fp6"),
    "fp4": ("kernels/gemm_fp4.cu", "tilemem_launch_gemm_fp4"),
}


def test_cli_writes_offline_json_and_markdown() -> None:
    with TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "olmoe_benchmark"
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "benchmark_olmoe_integration_interface"),
                "--out-dir",
                str(out_dir),
                "--iterations",
                "7",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        stdout_summary = json.loads(proc.stdout)

        summary_path = out_dir / "olmoe_integration_summary.json"
        markdown_path = out_dir / "olmoe_integration_report.md"
        assert summary_path.exists()
        assert markdown_path.exists()

        summary = json.loads(summary_path.read_text())
        assert stdout_summary == summary
        assert summary["schema_version"] == "tilepo_olmoe_integration_benchmark_v1"
        assert summary["offline"] is True
        assert summary["real_cuda_sources"] is True
        assert summary["simulated_external_cuda"] is False
        assert "actual __global__ CUDA kernels" in summary["disclaimer"]

        cases = {case["name"]: case for case in summary["cases"]}
        assert set(cases) == EXPECTED_CASES
        assert summary["iterations"] == 7
        assert set(summary["kernel_compile"].keys()) == {"fp8", "fp6", "fp4"}
        assert "nvcc_available" in summary["cuda_environment"]
        assert "gpu_available" in summary["cuda_environment"]

        baseline = cases["bf16_fp32_baseline"]
        fp8 = cases["fp8_external_cuda_packed"]
        fp6 = cases["fp6_external_cuda_packed"]
        fp4 = cases["fp4_external_cuda_packed"]

        assert baseline["backend"] == "kt_fallback"
        assert baseline["format"] == "bf16_accum_fp32"
        assert baseline["kt_fallback_tiles"] == baseline["total_tiles"]
        assert baseline["external_backend_tiles"] == 0
        assert baseline["unsupported_fallback_tiles"] == 0
        assert baseline["dispatch_success_tiles"] == baseline["total_tiles"]
        assert baseline["cuda_kernel_source"] is None

        for precision, external in (("fp8", fp8), ("fp6", fp6), ("fp4", fp4)):
            assert external["backend"] == "olmoe_external_cuda"
            assert external["external_backend_tiles"] == external["total_tiles"]
            assert external["dispatch_success_tiles"] == external["total_tiles"]
            assert external["unsupported_fallback_tiles"] == 0
            assert external["estimated_payload_bytes"] < baseline["estimated_payload_bytes"]
            assert external["payload_reduction_vs_bf16"] > 0.0
            assert external["cuda_kernel_source"] == EXPECTED_KERNELS[precision][0]
            assert external["c_abi_launch_function"] == EXPECTED_KERNELS[precision][1]
            assert external["quality_claim"] == "external_developer_owned"
            assert external["compile_status"] == summary["kernel_compile"][precision]["status"]
            assert external["cuda_runtime"] is not None

        assert fp8["estimated_payload_bytes"] > fp6["estimated_payload_bytes"] > fp4["estimated_payload_bytes"]
        if summary["cuda_environment"]["nvcc_available"]:
            assert all(item["status"] == "compiled" for item in summary["kernel_compile"].values())
        if summary["cuda_environment"]["nvcc_available"] and summary["cuda_environment"]["gpu_available"]:
            assert all(
                cases[f"{precision}_external_cuda_packed"]["cuda_runtime"]["status"] == "success"
                for precision in ("fp8", "fp6", "fp4")
            )
            assert all(
                cases[f"{precision}_external_cuda_packed"]["cuda_runtime"]["avg_ms"] > 0.0
                for precision in ("fp8", "fp6", "fp4")
            )
            assert all(
                cases[f"{precision}_external_cuda_packed"]["cuda_runtime"]["gflops"] > 0.0
                for precision in ("fp8", "fp6", "fp4")
            )
        else:
            if not summary["cuda_environment"]["nvcc_available"]:
                assert summary["cuda_environment"]["compile_skipped_reason"] == "nvcc not found"
                assert all(item["status"] == "skipped_no_nvcc" for item in summary["kernel_compile"].values())

        report_text = markdown_path.read_text()
        assert "does not claim GPU runtime speed" in report_text
        assert "customer-owned kernels" in report_text


def test_cuda_sources_define_global_kernels_and_c_abi_launchers() -> None:
    for _precision, (relative_path, launch_function) in EXPECTED_KERNELS.items():
        source_path = ROOT / relative_path
        assert source_path.exists(), f"missing {relative_path}"
        source = source_path.read_text()
        assert "__global__" in source
        assert 'extern "C"' in source
        assert launch_function in source
        assert "cudaError_t" in source
        assert "<<<" in source and ">>>" in source


def main() -> None:
    test_cli_writes_offline_json_and_markdown()
    test_cuda_sources_define_global_kernels_and_c_abi_launchers()
    print("OLMoE integration benchmark tests passed")


if __name__ == "__main__":
    main()
