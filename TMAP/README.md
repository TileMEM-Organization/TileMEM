# TMAP: Two-Tier Tile Memory Allocation Predictor

TMAP is the TileMEM hardware-aware prediction module. V0.1 is intentionally
limited to a two-tier VRAM/DRAM memory model and BF16 TilePO V0.1 calibration
samples.

TMAP does not claim exact serving throughput prediction. It predicts relative
policy preference and conservative admission decisions:

- use KT expert-level placement;
- use TilePO coarse/fine/hybrid placement;
- use async planning on/off;
- fallback to KT when the predicted TilePO gain is below threshold.

## Inputs

- TilePO V0.1 ablation summary, for example
  `evidence/ablation/tilepo_ablation_summary.json`.
- Hardware profile JSON with VRAM/DRAM capacity, bandwidth, latency, and
  transfer cost.

Example profile:

```json
{
  "name": "rtx5090_ddr",
  "vram_capacity_gib": 32.0,
  "vram_bandwidth_gbps": 1792.0,
  "vram_latency_ns": 350.0,
  "dram_capacity_gib": 128.0,
  "dram_bandwidth_gbps": 95.0,
  "dram_latency_ns": 90000.0,
  "transfer_bandwidth_gbps": 64.0,
  "transfer_latency_us": 12.0
}
```

## Run

```bash
tools/tmap_predict \
  --summary evidence/ablation/tilepo_ablation_summary.json \
  --hardware-profile TMAP/hardware_profiles/rtx5090_ddr.json \
  --out-dir build/tmap_rtx5090_ddr
```

Expected outputs:

- `tmap_prediction_summary.json`
- `tmap_prediction_report.md`

To reduce customer bring-up time, TMAP can also run an explicit quick-planning
extrapolation mode for expert budgets that are not yet in the calibration
matrix:

```bash
tools/tmap_predict \
  --summary evidence/ablation/tilepo_ablation_summary.json \
  --hardware-profile TMAP/hardware_profiles/rtx5090_ddr.json \
  --out-dir build/tmap_rtx5090_ddr_mixed12 \
  --target mixed:12 \
  --allow-extrapolation
```

Extrapolated decisions are marked with `evidence_mode=extrapolated`, carry
lower confidence, and include a short probe plan. They are intended to avoid a
full matrix sweep, not to replace the final production smoke test.
Use `--target-experts 12` only when you intentionally want to scan the same
expert budget for every measured workload.

Checked-in V0.1 experiment reports are provided for two profiles:

- `TMAP/reports/rtx5090_ddr`: predicts `10/10` TilePO admissions with 0 KT
  fallbacks and 0.80 TilePO candidate-rank accuracy against observed best
  TilePO tok/s.
- `TMAP/reports/abundant_vram_ddr`: predicts `1/10` TilePO admissions and 9 KT
  fallbacks, illustrating that TilePO's marginal benefit shrinks when VRAM is
  abundant relative to the observed working set.

## Method

TMAP follows the same systems style used by GPU infrastructure stacks: an
analytical cost model, hardware profile parameters, empirical calibration, and a
conservative fallback gate.

The V0.1 score uses:

```text
predicted gain = observed TilePO gain from V0.1 samples
               * VRAM pressure modifier
               * bandwidth/latency modifier
               + tail-latency signal
               - transfer/fragmentation/planning risk
```

This is a relative predictor. It is designed to answer whether TilePO is likely
to be admitted over KT for a given two-tier hardware profile, not to predict an
exact `tok/s` value.

## Boundaries

- VRAM/DRAM two-tier only.
- BF16 V0.1 calibration only.
- Unseen expert budgets require `--allow-extrapolation` and are reported as
  quick-planning estimates with probe recommendations.
- Mixed precision, F4/F6/F8, and multi-tier memory are future work.
- TMAP should be validated against new measurements before production use on a
  new hardware/model pair.
