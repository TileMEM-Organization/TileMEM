#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${TILEMEM_MODEL_PATH:-}"
PLAN="${TILEMEM_PLAN:-configs/models/olmoe_1b_7b_example.tmem}"
WORKLOADS="${TILEMEM_WORKLOAD:-mixed}"
EXPERTS="${TILEMEM_EXPERTS:-2,4,6,8,10}"
OUT_DIR="${TILEMEM_OUT_DIR:-build/custom_model_run}"
REPEATS="${TILEMEM_REPEATS:-3}"
REQUEST_COUNT="${TILEMEM_REQUEST_COUNT:-5}"
WARMUP_REQUEST_COUNT="${TILEMEM_WARMUP_REQUEST_COUNT:-1}"
OUTPUT_TOKENS="${TILEMEM_OUTPUT_TOKENS:-8}"
SYSTEMS="${TILEMEM_SYSTEMS:-B,C}"
EXECUTE_FLAG="--execute"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --plan)
      PLAN="$2"
      shift 2
      ;;
    --workloads)
      WORKLOADS="$2"
      shift 2
      ;;
    --experts)
      EXPERTS="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --request-count)
      REQUEST_COUNT="$2"
      shift 2
      ;;
    --dry-run)
      EXECUTE_FLAG="--dry-run-commands"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODEL_PATH" ]]; then
  echo "Missing model path. Set TILEMEM_MODEL_PATH or pass --model-path." >&2
  exit 2
fi

if [[ ! -e "$MODEL_PATH" ]]; then
  echo "Model path does not exist: $MODEL_PATH" >&2
  exit 2
fi

python3 tools/run_tilepo_sweep \
  --mode serve \
  --c-mode kt_native \
  --plan "$PLAN" \
  --out-dir "$OUT_DIR" \
  --workloads "$WORKLOADS" \
  --experts "$EXPERTS" \
  --systems "$SYSTEMS" \
  --repeats "$REPEATS" \
  --request-count "$REQUEST_COUNT" \
  --warmup-request-count "$WARMUP_REQUEST_COUNT" \
  --output-tokens "$OUTPUT_TOKENS" \
  --model-dir "$MODEL_PATH" \
  --min-linux-available-gib 8 \
  --skip-existing-success \
  --require-real \
  "$EXECUTE_FLAG"
