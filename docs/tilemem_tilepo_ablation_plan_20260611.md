# TilePO V0.1 Ablation Plan

**Date:** 2026-06-11  
**Goal:** isolate expert-level placement, tile granularity, hybrid tiling, and
async planning metadata while recording both VRAM and DRAM/CPU memory.

## Matrix

Workloads:

- `mixed`
- `long_context`

The default V0.1 offline script runs the core `mixed,long_context` matrix.
Additional workloads can be enabled with
`TILEPO_WORKLOADS=mixed,long_context,profile_matched,long_output` after the
core matrix is reviewed.

GPU expert budgets:

```text
2,4,6,8,10
```

The original full budget list can still be enabled with
`TILEPO_EXPERTS=1,2,4,6,8,10,12,14,16`.

Repeated serving protocol:

```text
repeats = 3
request_count = 5
warmup_request_count = 1
output_tokens = 8
precision = BF16
```

## Ablations

| Policy | System path | Async modes | Meaning |
| --- | --- | --- | --- |
| `kt_expert` | KT/SGLang B | `off` | expert-level KT placement baseline |
| `tilepo_coarse` | TilePO C hook | `off,on` | coarse tile manifest, near expert-level tile count |
| `tilepo_fine` | TilePO C hook | `off,on` | fine tile manifest with many shards |
| `tilepo_hybrid` | TilePO C hook | `off,on` | hot experts coarse, cold experts fine |

The V0.1 TilePO policies compile different `.tmem` plans per expert budget. This
matters because the manifest tile count changes with both the policy and the
expert budget.

## Metrics

The V0.1 report groups by:

```text
workload, experts_per_layer, policy, async_planning, system
```

and reports:

- `tok_per_sec`
- `p95_ms`
- `p99_ms`
- `gpu_peak_gib`
- `cpu_ram_peak_gib`

`gpu_peak_gib` is sampled from `nvidia-smi` during measured requests. The
`cpu_ram_peak_gib` field is the benchmark process tree peak RSS proxy recorded
by the existing OpenAI varprompt bench.

## One-Click Offline Run

```bash
bash build/tilepo_ablation_20260611/run_all_offline.sh
```

Default core matrix size:

```text
2 workloads x 5 expert budgets x 3 repeats x 7 policy/system modes = 210 rows
```

Full optional matrix size:

```text
3 workloads x 9 expert budgets x 3 repeats x 7 policy/system modes = 567 rows
```

Status:

```bash
bash build/tilepo_ablation_20260611/status_offline.sh
```

Final outputs:

```text
build/tilepo_ablation_20260611/tilepo_ablation_manifest.json
build/tilepo_ablation_20260611/report/tilepo_ablation_summary.json
build/tilepo_ablation_20260611/report/tilepo_ablation_report.md
build/tilepo_ablation_20260611/tilepo_ablation.completed.json
```

## Claim Boundary

This V0.1 matrix is designed to separate:

- KT expert-level placement,
- TilePO coarse tile planning,
- TilePO fine tile planning,
- TilePO hot/cold hybrid tile planning,
- async planning on/off metadata and runtime probe accounting.

The current SGLang hook remains conservative: it observes MoE execution and
returns KT/SGLang's original BF16 output. Therefore V0.1 can measure serving
throughput impact, memory footprint, hook/probe overhead, and manifest tile
granularity, but it should not be described as complete native CUDA MoE
replacement unless a future replacement path is explicitly enabled and verified.
