#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m compileall -q tilepo TMAP
python3 tools/tests/assert_tilepo_ablation.py
python3 tools/tests/assert_tmap.py
bash scripts/reproduce_ablation.sh

PACKAGE_DIR="publish/TileMEM_TilePO_V0_1_20260611"
for required in \
  "$PACKAGE_DIR/TMAP/README.md" \
  "$PACKAGE_DIR/tools/tmap_predict" \
  "$PACKAGE_DIR/tools/tests/assert_tmap.py"; do
  if [[ ! -f "$required" ]]; then
    echo "missing packaged TMAP artifact: $required" >&2
    exit 1
  fi
done

if [[ -f "$PACKAGE_DIR/SHA256SUMS" ]]; then
  (cd "$PACKAGE_DIR" && sha256sum -c SHA256SUMS)
fi

if [[ -f "publish/TileMEM_TilePO_V0_1_20260611.tar.gz.sha256" ]]; then
  (cd publish && sha256sum -c TileMEM_TilePO_V0_1_20260611.tar.gz.sha256)
  tar -tzf publish/TileMEM_TilePO_V0_1_20260611.tar.gz \
    TileMEM_TilePO_V0_1_20260611/TMAP/README.md \
    TileMEM_TilePO_V0_1_20260611/tools/tmap_predict \
    TileMEM_TilePO_V0_1_20260611/tools/tests/assert_tmap.py >/dev/null
fi

echo "TileMEM / TilePO artifact verification passed."
