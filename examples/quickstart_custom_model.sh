#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${TILEMEM_MODEL_PATH:?Set TILEMEM_MODEL_PATH=/path/to/moe/checkpoint first.}"

bash scripts/reproduce_with_model.sh \
  --model-path "$TILEMEM_MODEL_PATH" \
  --plan "${TILEMEM_PLAN:-configs/models/olmoe_1b_7b_example.tmem}" \
  --workloads "${TILEMEM_WORKLOAD:-mixed}" \
  --experts "${TILEMEM_EXPERTS:-2,4,6,8,10}" \
  --out-dir "${TILEMEM_OUT_DIR:-build/custom_model_run}"
