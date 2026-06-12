#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]

TILEMEM_OWNS = {
    "tile_splitting_and_id_management",
    "tile_dtype_tag_format_metadata",
    "scale_metadata_address_size_layout",
    "backend_capability_registration",
    "manifest_generation",
    "runtime_dispatch_tile_handle",
    "fallback_chain",
    "external_kernel_contract",
}

EXTERNAL_OWNS = {
    "concrete_fp8_fp6_fp4_kernels",
    "model_structure_adapter",
    "weight_quantization",
    "calibration_method",
    "quality_evaluation",
    "backend_specific_layout",
    "cuda_tilelang_triton_rocm_implementation",
}

PRECISIONS = {"fp8", "fp6", "fp4"}


def test_customer_demo_outputs_machine_checkable_contract() -> None:
    with TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "customer_demo"
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "examples" / "customer_integration_end_to_end.py"),
                "--out-dir",
                str(out_dir),
                "--iterations",
                "3",
                "--m",
                "8",
                "--n",
                "64",
                "--k",
                "64",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        stdout_payload = json.loads(proc.stdout)

        json_path = out_dir / "customer_integration_end_to_end.json"
        markdown_path = out_dir / "customer_integration_end_to_end.md"
        assert json_path.exists()
        assert markdown_path.exists()
        payload = json.loads(json_path.read_text())
        assert payload == stdout_payload

        assert payload["schema_version"] == "tilemem_customer_integration_demo_v1"
        assert set(payload["tilemem_responsibilities"]) == TILEMEM_OWNS
        assert set(payload["external_developer_responsibilities"]) == EXTERNAL_OWNS
        assert set(payload["kernel_contract_by_precision"]) == PRECISIONS

        for item in payload["tilemem_responsibilities"].values():
            assert item["owner"] == "TileMEM"
            assert item["status"] == "provided"
            assert item["evidence"]

        for item in payload["external_developer_responsibilities"].values():
            assert item["owner"] == "External developer"
            assert item["status"] in {
                "customer_owned_sample_provided",
                "customer_owned_contract_declared",
            }
            assert item["evidence"]

        capability = payload["tilemem_responsibilities"]["backend_capability_registration"]["evidence"]
        assert capability["name"] == "olmoe_external_cuda"
        assert set(capability["formats"]) == {"fp8_e4m3_sample", "fp6_s6_sample", "fp4_s4_sample"}

        manifest = payload["tilemem_responsibilities"]["manifest_generation"]["evidence"]
        assert manifest["schema_version"] == "tilepo_manifest_v1"
        assert manifest["total_tiles"] == 1024
        assert manifest["checksum"]

        fallback = payload["tilemem_responsibilities"]["fallback_chain"]["evidence"]
        assert fallback["manifest_fallback_chain"] == ["bf16", "kt_fallback"]

        for precision, contract in payload["kernel_contract_by_precision"].items():
            assert contract["dtype_tag"] == precision
            assert contract["stable_key"].startswith("L")
            assert contract["weight_offset"] >= 0
            assert contract["weight_bytes"] > 0
            assert contract["scale_offset"] >= 0
            assert contract["scale_bytes"] > 0
            assert contract["scale_layout"].startswith("block_n")
            assert contract["backend"] == "olmoe_external_cuda"
            assert contract["dispatchable"] is True
            assert contract["fallback"] == {"dtype": "bf16", "backend": "kt_fallback"}
            assert Path(ROOT / contract["kernel_source"]).exists()
            assert contract["c_abi_launch_function"] == f"tilemem_launch_gemm_{precision}"

        cuda_cases = {
            case["dtype"]: case
            for case in payload["benchmark_summary"]["cases"]
            if case["backend"] == "olmoe_external_cuda"
        }
        assert set(cuda_cases) == PRECISIONS
        for precision, case in cuda_cases.items():
            assert case["cuda_runtime"] is not None
            if payload["benchmark_summary"]["cuda_environment"]["gpu_available"]:
                assert case["cuda_runtime"]["status"] == "success"

        markdown = markdown_path.read_text()
        assert "TileMEM owns" in markdown
        assert "External developer owns" in markdown
        assert "not claim model quality" in markdown


def main() -> None:
    test_customer_demo_outputs_machine_checkable_contract()
    print("Customer integration end-to-end example tests passed")


if __name__ == "__main__":
    main()
