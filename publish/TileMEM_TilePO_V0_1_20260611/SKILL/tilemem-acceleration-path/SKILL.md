---
name: tilemem-acceleration-path
description: Use when connecting a MoE model or checkpoint to TileMEM, compiling MIR/manifests, preparing checkpoint artifacts, running TilePO/KT comparisons, using TMAP, or choosing a TileMEM acceleration path.
---

# TileMEM Acceleration Path

## Overview

Treat acceleration as a staged evidence chain: health check -> model/checkpoint metadata -> MIR/manifest/tile handles -> dry-run backend command -> TMAP narrowing -> same-budget KT vs TilePO benchmark.

## Required First Checks

```bash
tools/tilemem doctor
tools/tilemem verify --quick
python3 examples/tilemem_checkpoint_integration.py
python3 examples/tilemem_industrial_quickstart.py \
  --out-json build/tilemem_industrial_quickstart.json
```

Verify TileMEM import, checkpoint topology inference, manifest/tile map generation, BF16 fallback availability, and backend dry-run command construction before running long experiments.

## Connect A Checkpoint

Use dry-run first:

```bash
tools/tilemem checkpoint prepare \
  --checkpoint-dir /path/to/hf_moe_checkpoint \
  --out-dir build/my_moe_checkpoint_artifact \
  --backend sglang \
  --dry-run
```

For KT-native:

```bash
tools/tilemem checkpoint prepare \
  --checkpoint-dir /path/to/hf_moe_checkpoint \
  --out-dir build/my_moe_checkpoint_artifact_kt \
  --backend kt_native \
  --dry-run
```

Inspect `model_spec.json`, `model.mir.json`, `model.manifest.json`, `checkpoint_weight_map.json`, `tile_checkpoint_map.json`, and `checkpoint_artifact_summary.json`. Only use `--execute` after the generated command, backend binary, model path, tile map, and fallback path are correct.

## Compile And Predict

Compile a public spec or `.tmem` plan:

```bash
tools/tilemem compile \
  --model-spec configs/models/model_spec_template.json \
  --out-dir build/my_moe_compile

tools/tilemem compile \
  --plan configs/models/model_template.tmem \
  --out-dir build/my_moe_plan_compile
```

Use TMAP to reduce scan size:

```bash
tools/tilemem tmap predict \
  --summary evidence/ablation/tilepo_ablation_summary.json \
  --hardware-profile TMAP/hardware_profiles/rtx5090_ddr.json \
  --out-dir build/tmap_my_gpu \
  --target mixed:8
```

For unseen budgets, require explicit extrapolation and call it a planning estimate:

```bash
tools/tilemem tmap predict \
  --summary evidence/ablation/tilepo_ablation_summary.json \
  --hardware-profile TMAP/hardware_profiles/rtx5090_ddr.json \
  --out-dir build/tmap_mixed_12 \
  --target mixed:12 \
  --allow-extrapolation
```

## Benchmark Decision Rule

Compare at the same expert budget:

- KT expert placement
- TilePO coarse
- TilePO fine
- TilePO hybrid
- async planning off/on when supported

Choose TilePO only when it beats KT at the same expert budget on throughput without p95/p99 regression. If TMAP predicts low gain, confidence is low, or VRAM is abundant enough that KT wins, recommend KT fallback or a shorter targeted probe.

## Guardrails

- Do not claim universal speedup; TilePO benefits are workload/hardware/budget dependent.
- Do not treat TMAP as exact tok/s prediction; it predicts policy preference and confidence.
- Do not skip manifest and tile map inspection for real checkpoints.
- Do not benchmark TilePO against KT with different expert budgets.
- Keep BF16/KT fallback available while integrating a model.
