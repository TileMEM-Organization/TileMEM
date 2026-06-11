from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def finalize_release(manifest_path: Path | str, package_dir: Path | str, extra_paths: list[Path] | None = None) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in [manifest_path] + list(extra_paths or []):
        if not path.exists():
            continue
        dest = package_dir / path.name
        if path.resolve() != dest.resolve():
            shutil.copy2(path, dest)
        copied.append(dest)
    checksum_path = package_dir / "raw_files.sha256"
    lines = []
    for path in sorted(copied):
        lines.append(f"{_sha256(path)}  {path.name}")
    checksum_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    release_manifest = {
        "schema_version": "tilepo_release_manifest_v1",
        "source_manifest": str(manifest_path),
        "package_dir": str(package_dir),
        "files": [path.name for path in copied],
        "checksum_file": checksum_path.name,
    }
    (package_dir / "results_manifest.json").write_text(json.dumps(release_manifest, indent=2, sort_keys=True) + "\n")
    return release_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

