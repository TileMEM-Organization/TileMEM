#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m compileall -q tilepo
python3 tools/tests/assert_tilepo_ablation.py
bash scripts/reproduce_ablation.sh

PACKAGE_DIR="publish/TileMEM_TilePO_V0_1_20260611"
if [[ -f "$PACKAGE_DIR/SHA256SUMS" ]]; then
  (cd "$PACKAGE_DIR" && sha256sum -c SHA256SUMS)
fi

if [[ -f "publish/TileMEM_TilePO_V0_1_20260611.tar.gz.sha256" ]]; then
  (cd publish && sha256sum -c TileMEM_TilePO_V0_1_20260611.tar.gz.sha256)
fi

echo "TileMEM / TilePO artifact verification passed."
