#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MANIFEST="${1:-evidence/ablation/tilepo_ablation_manifest.json}"
OUT_DIR="${2:-build/reproduced_ablation_report}"

python3 tools/report_tilepo_ablation \
  --manifest "$MANIFEST" \
  --out-dir "$OUT_DIR" \
  --workloads mixed,long_context \
  --experts 2,4,6,8,10 \
  --policies kt_expert,tilepo_coarse,tilepo_fine,tilepo_hybrid \
  --async-modes off,on \
  --repeats 3 \
  --require-real

export TILEPO_REPRO_OUT_DIR="$OUT_DIR"
export TILEPO_REPRO_MANIFEST="$MANIFEST"
python3 - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["TILEPO_REPRO_OUT_DIR"])
manifest_path = Path(os.environ["TILEPO_REPRO_MANIFEST"])
summary = json.loads((out_dir / "tilepo_ablation_summary.json").read_text())
manifest = json.loads(manifest_path.read_text())
print("TilePO V0.1 ablation gate:", summary["gate"]["status"])
print(f"Rows: {manifest.get('actual_result_rows')}/{manifest.get('expected_result_rows')}")
print("Groups:", len(summary["groups"]))
if summary["gate"]["status"] != "PASS":
    raise SystemExit(1)
if manifest.get("actual_result_rows") != 210 or manifest.get("expected_result_rows") != 210:
    raise SystemExit("unexpected V0.1 row count")
if len(summary["groups"]) != 70:
    raise SystemExit("unexpected V0.1 group count")
PY
