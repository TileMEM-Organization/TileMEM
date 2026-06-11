#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m pip install -e .
python3 - <<'PY'
import tilepo
print("TileMEM / TilePO import OK:", tilepo.__name__)
PY
