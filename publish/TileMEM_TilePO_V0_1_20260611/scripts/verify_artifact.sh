#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m compileall -q tilemem tilepo TMAP
python3 tools/tests/assert_agent_skills.py
python3 tools/tests/assert_tilemem_sdk.py
python3 tools/tests/assert_tilemem_cli.py
python3 tools/tests/assert_checkpoint_integration.py
python3 tools/tests/assert_tilemem_industrial_quickstart.py
python3 tools/tests/assert_public_mir_interface.py
python3 tools/tests/assert_integration_interface.py
python3 tools/tests/assert_olmoe_integration_benchmark.py
python3 tools/tests/assert_customer_integration_end_to_end.py
python3 tools/tests/assert_tilepo_ablation.py
python3 tools/tests/assert_tmap.py
bash scripts/reproduce_ablation.sh

PACKAGE_DIR="publish/TileMEM_TilePO_V0_1_20260611"
for required in \
  "$PACKAGE_DIR/TMAP/README.md" \
  "$PACKAGE_DIR/.codex/skills/tilemem-environment-setup/SKILL.md" \
  "$PACKAGE_DIR/.codex/skills/tilemem-environment-setup/agents/openai.yaml" \
  "$PACKAGE_DIR/.codex/skills/tilemem-acceleration-path/SKILL.md" \
  "$PACKAGE_DIR/.codex/skills/tilemem-acceleration-path/agents/openai.yaml" \
  "$PACKAGE_DIR/.codex/skills/tilemem-backend-precision-path/SKILL.md" \
  "$PACKAGE_DIR/.codex/skills/tilemem-backend-precision-path/agents/openai.yaml" \
  "$PACKAGE_DIR/docs/customer_integration_end_to_end_example_20260613.md" \
  "$PACKAGE_DIR/docs/tilemem_checkpoint_integration.md" \
  "$PACKAGE_DIR/docs/tilemem_python_sdk_quickstart.md" \
  "$PACKAGE_DIR/configs/models/model_spec_template.json" \
  "$PACKAGE_DIR/tilemem/__init__.py" \
  "$PACKAGE_DIR/tilemem/checkpoint.py" \
  "$PACKAGE_DIR/tilemem/sdk.py" \
  "$PACKAGE_DIR/tilepo/model_interface.py" \
  "$PACKAGE_DIR/tilepo/integration.py" \
  "$PACKAGE_DIR/tilepo/mir/io.py" \
  "$PACKAGE_DIR/examples/olmoe_external_cuda_backend.py" \
  "$PACKAGE_DIR/examples/customer_integration_end_to_end.py" \
  "$PACKAGE_DIR/examples/tilemem_checkpoint_integration.py" \
  "$PACKAGE_DIR/examples/tilemem_industrial_quickstart.py" \
  "$PACKAGE_DIR/kernels/gemm_fp8.cu" \
  "$PACKAGE_DIR/kernels/gemm_fp6.cu" \
  "$PACKAGE_DIR/kernels/gemm_fp4.cu" \
  "$PACKAGE_DIR/tools/benchmark_olmoe_integration_interface" \
  "$PACKAGE_DIR/tools/tilemem" \
  "$PACKAGE_DIR/tools/tests/assert_agent_skills.py" \
  "$PACKAGE_DIR/tools/tests/assert_integration_interface.py" \
  "$PACKAGE_DIR/tools/tests/assert_tilemem_cli.py" \
  "$PACKAGE_DIR/tools/tests/assert_checkpoint_integration.py" \
  "$PACKAGE_DIR/tools/tests/assert_tilemem_sdk.py" \
  "$PACKAGE_DIR/tools/tests/assert_tilemem_industrial_quickstart.py" \
  "$PACKAGE_DIR/tools/tests/assert_olmoe_integration_benchmark.py" \
  "$PACKAGE_DIR/tools/tests/assert_customer_integration_end_to_end.py" \
  "$PACKAGE_DIR/tools/tilemem_checkpoint_prepare" \
  "$PACKAGE_DIR/tools/tmap_predict" \
  "$PACKAGE_DIR/tools/tests/assert_public_mir_interface.py" \
  "$PACKAGE_DIR/tools/tests/assert_tmap.py"; do
  if [[ ! -f "$required" ]]; then
    echo "missing packaged TMAP artifact: $required" >&2
    exit 1
  fi
done

"$PACKAGE_DIR/tools/tilemem" doctor --json >/dev/null

if [[ -f "$PACKAGE_DIR/SHA256SUMS" ]]; then
  (cd "$PACKAGE_DIR" && sha256sum -c SHA256SUMS)
fi

if [[ -f "publish/TileMEM_TilePO_V0_1_20260611.tar.gz.sha256" ]]; then
  (cd publish && sha256sum -c TileMEM_TilePO_V0_1_20260611.tar.gz.sha256)
  tar -tzf publish/TileMEM_TilePO_V0_1_20260611.tar.gz \
    TileMEM_TilePO_V0_1_20260611/TMAP/README.md \
    TileMEM_TilePO_V0_1_20260611/.codex/skills/tilemem-environment-setup/SKILL.md \
    TileMEM_TilePO_V0_1_20260611/.codex/skills/tilemem-environment-setup/agents/openai.yaml \
    TileMEM_TilePO_V0_1_20260611/.codex/skills/tilemem-acceleration-path/SKILL.md \
    TileMEM_TilePO_V0_1_20260611/.codex/skills/tilemem-acceleration-path/agents/openai.yaml \
    TileMEM_TilePO_V0_1_20260611/.codex/skills/tilemem-backend-precision-path/SKILL.md \
    TileMEM_TilePO_V0_1_20260611/.codex/skills/tilemem-backend-precision-path/agents/openai.yaml \
    TileMEM_TilePO_V0_1_20260611/docs/customer_integration_end_to_end_example_20260613.md \
    TileMEM_TilePO_V0_1_20260611/docs/tilemem_checkpoint_integration.md \
    TileMEM_TilePO_V0_1_20260611/docs/tilemem_python_sdk_quickstart.md \
    TileMEM_TilePO_V0_1_20260611/configs/models/model_spec_template.json \
    TileMEM_TilePO_V0_1_20260611/tilemem/__init__.py \
    TileMEM_TilePO_V0_1_20260611/tilemem/checkpoint.py \
    TileMEM_TilePO_V0_1_20260611/tilemem/sdk.py \
    TileMEM_TilePO_V0_1_20260611/tilepo/model_interface.py \
    TileMEM_TilePO_V0_1_20260611/tilepo/integration.py \
    TileMEM_TilePO_V0_1_20260611/tilepo/mir/io.py \
    TileMEM_TilePO_V0_1_20260611/examples/olmoe_external_cuda_backend.py \
    TileMEM_TilePO_V0_1_20260611/examples/customer_integration_end_to_end.py \
    TileMEM_TilePO_V0_1_20260611/examples/tilemem_checkpoint_integration.py \
    TileMEM_TilePO_V0_1_20260611/examples/tilemem_industrial_quickstart.py \
    TileMEM_TilePO_V0_1_20260611/kernels/gemm_fp8.cu \
    TileMEM_TilePO_V0_1_20260611/kernels/gemm_fp6.cu \
    TileMEM_TilePO_V0_1_20260611/kernels/gemm_fp4.cu \
    TileMEM_TilePO_V0_1_20260611/tools/benchmark_olmoe_integration_interface \
    TileMEM_TilePO_V0_1_20260611/tools/tilemem \
    TileMEM_TilePO_V0_1_20260611/tools/tests/assert_agent_skills.py \
    TileMEM_TilePO_V0_1_20260611/tools/tests/assert_integration_interface.py \
    TileMEM_TilePO_V0_1_20260611/tools/tests/assert_tilemem_cli.py \
    TileMEM_TilePO_V0_1_20260611/tools/tests/assert_checkpoint_integration.py \
    TileMEM_TilePO_V0_1_20260611/tools/tests/assert_tilemem_sdk.py \
    TileMEM_TilePO_V0_1_20260611/tools/tests/assert_tilemem_industrial_quickstart.py \
    TileMEM_TilePO_V0_1_20260611/tools/tests/assert_olmoe_integration_benchmark.py \
    TileMEM_TilePO_V0_1_20260611/tools/tests/assert_customer_integration_end_to_end.py \
    TileMEM_TilePO_V0_1_20260611/tools/tilemem_checkpoint_prepare \
    TileMEM_TilePO_V0_1_20260611/tools/tmap_predict \
    TileMEM_TilePO_V0_1_20260611/tools/tests/assert_public_mir_interface.py \
    TileMEM_TilePO_V0_1_20260611/tools/tests/assert_tmap.py >/dev/null
fi

echo "TileMEM / TilePO artifact verification passed."
