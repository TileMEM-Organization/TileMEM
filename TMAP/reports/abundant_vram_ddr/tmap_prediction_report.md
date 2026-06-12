# TMAP V0.2 Prediction Report

TMAP is a two-tier Tile Memory Allocation Predictor for TileMEM. This
report uses V0.1 TilePO/KT measurements as calibration samples and
applies a VRAM/DRAM hardware profile to predict relative policy
preference and conservative fallback decisions.

## Hardware Profile

- Name: `abundant_vram_ddr`
- VRAM: 96.00 GiB, 2200.00 GB/s, 300.00 ns
- DRAM: 128.00 GiB, 95.00 GB/s, 90000.00 ns
- Transfer: 64.00 GB/s, 12.00 us

## Summary

- Groups: 10
- Measured groups: 10
- Extrapolated groups: 0
- Admit TilePO: 1
- Fallback KT: 9
- TilePO candidate-rank accuracy against V0.1 observed best TilePO tok/s: 0.40
- Mean predicted tok/s gain: 1.47%
- Mean predicted p95 reduction: 10.80%

## Decisions

| Workload | Experts | Evidence | Admit | Recommended policy | Pred. tok/s gain | Pred. p95 reduction | Confidence | Probe | Factor | Risk |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| long_context | 2 | measured | KT | kt_expert | 1.50% | 10.42% | 0.37 | no | low_vram_pressure | low |
| long_context | 4 | measured | KT | kt_expert | 1.81% | 12.44% | 0.45 | no | low_vram_pressure | low |
| long_context | 6 | measured | KT | kt_expert | 1.68% | 11.85% | 0.44 | no | low_vram_pressure | low |
| long_context | 8 | measured | TilePO | tilepo_coarse_async_on | 3.05% | 19.03% | 0.47 | no | low_vram_pressure | low |
| long_context | 10 | measured | KT | kt_expert | 2.05% | 14.07% | 0.46 | no | low_vram_pressure | exposed_planning |
| mixed | 2 | measured | KT | kt_expert | 0.39% | 4.22% | 0.14 | no | low_vram_pressure | low |
| mixed | 4 | measured | KT | kt_expert | 0.98% | 7.74% | 0.28 | no | low_vram_pressure | low |
| mixed | 6 | measured | KT | kt_expert | 1.46% | 10.47% | 0.39 | no | low_vram_pressure | low |
| mixed | 8 | measured | KT | kt_expert | 2.00% | 13.56% | 0.46 | no | low_vram_pressure | low |
| mixed | 10 | measured | KT | kt_expert | -0.22% | 4.22% | 0.17 | no | low_vram_pressure | low |

## Boundary

TMAP V0.2 predicts relative policy preference, not exact serving
throughput. It is calibrated from V0.1 BF16 samples and uses a
two-tier VRAM/DRAM model only. Extrapolated expert budgets are
quick-planning estimates and must be validated with a short probe.
Mixed precision and multi-tier memory are out of scope for this version.
