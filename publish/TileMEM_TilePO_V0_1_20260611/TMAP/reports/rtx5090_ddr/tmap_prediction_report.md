# TMAP V0.2 Prediction Report

TMAP is a two-tier Tile Memory Allocation Predictor for TileMEM. This
report uses V0.1 TilePO/KT measurements as calibration samples and
applies a VRAM/DRAM hardware profile to predict relative policy
preference and conservative fallback decisions.

## Hardware Profile

- Name: `rtx5090_ddr`
- VRAM: 32.00 GiB, 1792.00 GB/s, 350.00 ns
- DRAM: 128.00 GiB, 95.00 GB/s, 90000.00 ns
- Transfer: 64.00 GB/s, 12.00 us

## Summary

- Groups: 10
- Measured groups: 10
- Extrapolated groups: 0
- Admit TilePO: 10
- Fallback KT: 0
- TilePO candidate-rank accuracy against V0.1 observed best TilePO tok/s: 0.80
- Mean predicted tok/s gain: 21.60%
- Mean predicted p95 reduction: 16.24%

## Decisions

| Workload | Experts | Evidence | Admit | Recommended policy | Pred. tok/s gain | Pred. p95 reduction | Confidence | Probe | Factor | Risk |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| long_context | 2 | measured | TilePO | tilepo_coarse_async_on | 15.32% | 14.57% | 0.77 | no | observed_tilepo_gain | low |
| long_context | 4 | measured | TilePO | tilepo_coarse_async_on | 22.57% | 18.31% | 0.91 | no | observed_tilepo_gain | low |
| long_context | 6 | measured | TilePO | tilepo_coarse_async_on | 28.91% | 18.33% | 0.95 | no | observed_tilepo_gain | low |
| long_context | 8 | measured | TilePO | tilepo_coarse_async_on | 33.93% | 30.05% | 0.95 | no | observed_tilepo_gain | low |
| long_context | 10 | measured | TilePO | tilepo_coarse_async_off | 27.03% | 22.18% | 0.95 | no | observed_tilepo_gain | exposed_planning |
| mixed | 2 | measured | TilePO | tilepo_hybrid_async_off | 6.93% | 6.59% | 0.54 | no | observed_tilepo_gain | exposed_planning |
| mixed | 4 | measured | TilePO | tilepo_hybrid_async_on | 14.74% | 11.88% | 0.76 | no | observed_tilepo_gain | low |
| mixed | 6 | measured | TilePO | tilepo_coarse_async_on | 22.56% | 16.28% | 0.91 | no | observed_tilepo_gain | low |
| mixed | 8 | measured | TilePO | tilepo_fine_async_on | 32.44% | 20.41% | 0.95 | no | observed_tilepo_gain | low |
| mixed | 10 | measured | TilePO | tilepo_fine_async_off | 11.53% | 3.77% | 0.69 | no | observed_tilepo_gain | fragmentation_overhead |

## Boundary

TMAP V0.2 predicts relative policy preference, not exact serving
throughput. It is calibrated from V0.1 BF16 samples and uses a
two-tier VRAM/DRAM model only. Extrapolated expert budgets are
quick-planning estimates and must be validated with a short probe.
Mixed precision and multi-tier memory are out of scope for this version.
