# TileMEM / TilePO V0.1 Technical Report

**Date:** 2026-06-11  
**Project:** TileMEM  
**System:** TilePO  
**Scope:** BF16 profile-guided tile-level placement/admission for MoE serving

## Abstract

TilePO is a BF16 profile-guided tile-level placement/admission system for
mixture-of-experts serving. It studies whether MoE inference should expose
placement decisions below the expert level while keeping the serving comparison
fair: same model, same expert budget, and BF16 KT-native execution. The V0.1
artifact reports a V0.1 ablation over `mixed` and `long_context`, expert budgets
2/4/6/8/10, three repeats, and five measured requests per row. The released
public manifest contains 210/210 successful real rows and passes the TilePO V0.1
gate.

## Method

TilePO uses workload/profile evidence to choose a placement/admission policy
for MoE expert weights. The public V0.1 policy space contains:

- `kt_expert`: KT expert-level placement baseline.
- `tilepo_coarse`: coarse tile policy for throughput-oriented execution.
- `tilepo_fine`: fine tile policy for more granular admission.
- `tilepo_hybrid`: hot experts use coarse residency while cold experts use
  finer admission.
- async planning `off/on`: whether planning work is exposed synchronously or
  hidden behind the serving path.

The implementation uses `.tmem` plans, a typed MIR/manifest path, runtime
metrics, and report gates. The public comparison remains BF16 and does not
promote low-bit execution as a serving-quality claim.

## V0.1 Evidence

```text
Rows: 210 / 210 real success
Gate: PASS
Workloads: mixed, long_context
Expert budgets: 2, 4, 6, 8, 10
Repeats: 3
Request count: 5
Serving path: BF16 / KT-native
```

| Workload | Experts | Best TilePO vs KT tok/s | p95 improvement |
| --- | ---: | ---: | ---: |
| `long_context` | 2 | +20.81% | +15.81% |
| `long_context` | 4 | +26.42% | +18.73% |
| `long_context` | 6 | +28.31% | +17.58% |
| `long_context` | 8 | +31.16% | +29.14% |
| `long_context` | 10 | +26.22% | +21.47% |
| `mixed` | 2 | +10.59% | +7.48% |
| `mixed` | 4 | +17.23% | +12.37% |
| `mixed` | 6 | +21.95% | +15.98% |
| `mixed` | 8 | +31.42% | +20.23% |
| `mixed` | 10 | +12.95% | +6.93% |

## Interpretation

The V0.1 evidence supports adaptive TilePO policy selection, not a blanket claim
that fine-grained tiles are always better. In the tested matrix, TilePO finds
policies that outperform KT expert-level placement under the same expert budget
on both `mixed` and `long_context`. Coarse+async is often strong for throughput,
fine tiles can help selected mixed regimes, and hybrid policies expose useful
boundaries rather than dominating everywhere.

## Claim Boundary

V0.1 does not claim:

- full native CUDA replacement of KT/SGLang MoE kernels;
- FP8/MXFP4 serving-quality gains;
- universal wins across all models, GPUs, or serving systems;
- that fine-grained tile splitting alone explains all gains.

## Reproduction

Offline artifact verification:

```bash
bash examples/quickstart_offline.sh
```

Real BF16 evaluation with a compatible MoE model:

```bash
export TILEMEM_MODEL_PATH=/path/to/moe/checkpoint
bash examples/quickstart_custom_model.sh
```
