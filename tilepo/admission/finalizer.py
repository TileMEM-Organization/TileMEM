from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


class FinalizerError(RuntimeError):
    pass


@dataclass(frozen=True)
class FinalizerResult:
    manifest_path: Path
    readme_path: Path
    claim_checklist_path: Path
    checksum_path: Path
    summary: dict[str, Any]


def finalize_tilepo_deployment(
    *,
    selected_plan_path: Path | str,
    fallback_plan_path: Path | str,
    admission_report_path: Path | str,
    manifest_paths: list[Path | str],
    out_dir: Path | str,
    release_name: str,
    extra_paths: list[Path | str] | None = None,
) -> FinalizerResult:
    selected_plan_path = Path(selected_plan_path)
    fallback_plan_path = Path(fallback_plan_path)
    admission_report_path = Path(admission_report_path)
    manifest_paths = [Path(path) for path in manifest_paths]
    extra_paths = [Path(path) for path in (extra_paths or [])]
    out_dir = Path(out_dir)
    selected_plan = _load_json(selected_plan_path)
    gate = selected_plan.get("gate", {"status": "UNKNOWN", "failures": ["selected plan has no gate"]})
    if gate.get("status") != "PASS":
        raise FinalizerError(f"cannot finalize non-PASS selected plan: {gate}")

    out_dir.mkdir(parents=True, exist_ok=True)
    copied = _copy_inputs(
        out_dir,
        [selected_plan_path, fallback_plan_path, admission_report_path] + extra_paths + manifest_paths,
    )
    readme_path = out_dir / "README.md"
    claim_checklist_path = out_dir / "CLAIM_CHECKLIST.md"
    checksum_path = out_dir / "raw_files.sha256"
    readme_path.write_text(_readme(release_name, selected_plan))
    claim_checklist_path.write_text(_claim_checklist(selected_plan))
    copied.extend([readme_path, claim_checklist_path])
    checksum_path.write_text(_checksum_text(copied))

    summary = {
        "schema_version": "tilepo_v0_1_deployment_release_v1",
        "release_name": release_name,
        "gate": gate,
        "selected_plan": selected_plan_path.name,
        "fallback_plan": fallback_plan_path.name,
        "admission_report": admission_report_path.name,
        "files": sorted(path.name for path in copied),
        "checksum_file": checksum_path.name,
    }
    manifest_path = out_dir / "tilepo_v0_1_release_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return FinalizerResult(
        manifest_path=manifest_path,
        readme_path=readme_path,
        claim_checklist_path=claim_checklist_path,
        checksum_path=checksum_path,
        summary=summary,
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FinalizerError(f"missing required file: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise FinalizerError(f"JSON file is not an object: {path}")
    return data


def _copy_inputs(out_dir: Path, paths: list[Path]) -> list[Path]:
    copied: list[Path] = []
    used_names: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FinalizerError(f"missing required file: {path}")
        dest = out_dir / _unique_name(path.name, used_names)
        used_names.add(dest.name)
        if path.resolve() != dest.resolve():
            shutil.copy2(path, dest)
        copied.append(dest)
    return copied


def _unique_name(name: str, used_names: set[str]) -> str:
    if name not in used_names:
        return name
    path = Path(name)
    for index in range(2, 1000):
        candidate = f"{path.stem}_{index}{path.suffix}"
        if candidate not in used_names:
            return candidate
    raise FinalizerError(f"too many duplicate input names for {name}")


def _checksum_text(paths: list[Path]) -> str:
    lines = [f"{_sha256(path)}  {path.name}" for path in sorted(paths)]
    return "\n".join(lines) + ("\n" if lines else "")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readme(release_name: str, selected_plan: dict[str, Any]) -> str:
    selections = selected_plan.get("selections", {})
    lines = [
        f"# {release_name}",
        "",
        "This package contains a TilePO V0.1 offline admission result.",
        "",
        "TilePO is enabled only for workloads and expert budgets admitted by the",
        "BF16 evidence gate. KT remains the fallback path.",
        "",
        "## Selected Plans",
        "",
        "| Workload | System | Expert Budget | Precision |",
        "| --- | --- | ---: | --- |",
    ]
    for workload, selection in sorted(selections.items()):
        lines.append(
            f"| `{workload}` | {selection.get('selected_system', 'n/a')} | "
            f"{selection.get('selected_expert_budget', 'n/a')} | "
            f"{selection.get('serving_precision', 'BF16')} |"
        )
    return "\n".join(lines) + "\n"


def _claim_checklist(selected_plan: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# TilePO V0.1 Claim Checklist",
            "",
            "Can say:",
            "",
            "- TilePO V0.1 is an offline admission scheduler for VRAM/DRAM MoE residency.",
            "- The selected deployment plan is BF16-only unless a separate quality gate is provided.",
            "- TilePO is admitted only for measured workload/budget pairs with positive evidence.",
            "- KT fallback is retained for non-admitted workload/budget regions.",
            "",
            "Cannot say:",
            "",
            "- TilePO universally beats KT/SGLang.",
            "- TilePO improves full-VRAM deployments where all experts already fit on GPU.",
            "- Low-bit FP8/MXFP4 serving quality is proven by this BF16 package.",
            "",
            f"Selected plan schema: `{selected_plan.get('schema_version', 'unknown')}`",
            "",
        ]
    )
