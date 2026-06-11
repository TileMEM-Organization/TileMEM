<p align="center">
  <img src="docs/assets/tilemem-tilepo-logo.png" alt="TileMEM TilePO logo" width="760">
</p>

# TileMEM / TilePO

TileMEM is an open MoE serving optimization project. TilePO is its BF16
profile-guided tile-level placement/admission system for high-throughput MoE
serving.

This repository is the V0.1 priority artifact for TilePO. It contains the
method description, source code, V0.1 evidence, public manifests, reproducibility
scripts, and checksum tooling needed to make the result public, citable, and
verifiable .

## What TilePO Claims

TilePO studies when MoE expert weights should be admitted, retained, or
organized at tile granularity under fixed GPU expert budgets. V0.1 evaluates
TilePO in a BF16 KT-native serving path and compares against KT expert-level
placement under the same expert budget.

Safe claim:

> To the best of the author's knowledge, TilePO is among the first open artifact
> systems to publicly disclose and evaluate BF16 profile-guided tile-level
> placement/admission for MoE serving under same expert-budget KT baselines.

Boundary:

- no full native CUDA MoE replacement claim;
- no FP8/MXFP4 serving-quality claim;
- no universal win claim across all models, GPUs, or serving systems;
- no claim that fine-grained tiles alone explain every win.

## V0.1 Headline Evidence

The strongest V0.1 evidence is the V0.1 ablation matrix:

```text
Workloads: mixed, long_context
Experts: 2, 4, 6, 8, 10
Policies: kt_expert, tilepo_coarse, tilepo_fine, tilepo_hybrid
Async planning: off, on
Repeats: 3
Request count: 5
Rows: 210 / 210 real success
Gate: PASS
Serving precision: BF16 / KT-native path
```

<p align="center">
  <img src="docs/assets/tilepo-throughput-improvement.png" alt="TilePO throughput improvement over KT across expert budgets" width="860">
</p>

<p align="center">
  <img src="docs/assets/tilepo-p95-improvement.png" alt="TilePO p95 latency improvement over KT across expert budgets" width="860">
</p>

## Quickstart: Offline Verification

This does not require a GPU or model checkpoint. It verifies the released V0.1
manifest and regenerates the V0.1 report from evidence files.

```bash
git clone https://github.com/TerminusAkivili/TileMEM.git
cd TileMEM
python3 -m pip install -e .
bash examples/quickstart_offline.sh
```

Expected outcome:

```text
TilePO V0.1 ablation gate: PASS
Rows: 210/210
Groups: 70
```

## Quickstart: Bring Your Own MoE Model

Real BF16 serving evaluation requires a compatible MoE checkpoint and the local
KT/SGLang runtime environment.

```bash
export TILEMEM_MODEL_PATH=/path/to/moe/checkpoint
export TILEMEM_PLAN=configs/models/olmoe_1b_7b_example.tmem
export TILEMEM_WORKLOAD=mixed
export TILEMEM_EXPERTS=2,4,6,8,10
export TILEMEM_OUT_DIR=build/custom_model_run

bash examples/quickstart_custom_model.sh
```

The public model interface is explicit:

- model path comes from `TILEMEM_MODEL_PATH` or `--model-path`;
- plan comes from a `.tmem` config;
- default precision is BF16;
- TilePO does not silently switch to FP8/MXFP4.

## Repository Layout

```text
tilepo/            TilePO Python implementation.
tools/             Reporters, sweep runners, V0.1 plan renderers, tests.
configs/          Replaceable model, workload, and plan examples.
evidence/ablation/
                  V0.1 report, summary, and public manifest.
paper/            Technical report snapshot.
scripts/          Verification, packaging, and real-run wrappers.
examples/         User-facing quickstarts.
publish/          Generated release packet.
```

## Priority Path

V0.1 is designed for this priority chain:

1. public GitHub repository;
2. GitHub tag and release: `v0.1-priority-2026-06-11`;
3. Zenodo DOI for the release archive;
4. Software Heritage SWHID plus SHA256 checksums.

See [docs/tilemem_tilepo_v0_1_priority_roadmap_20260611.md](docs/tilemem_tilepo_v0_1_priority_roadmap_20260611.md).

## Citation

Before Zenodo DOI is minted:

```bibtex
@software{tilemem_tilepo_v0_1_2026,
  title   = {TileMEM TilePO v0.1: BF16 Profile-Guided Tile-Level Placement/Admission for MoE Serving},
  author  = {TerminusAkivili},
  year    = {2026},
  version = {v0.1-priority-2026-06-11},
  url     = {https://github.com/TerminusAkivili/TileMEM}
}
```

After Zenodo release, replace `url` with the DOI field in `CITATION.cff`.

## License

The V0.1 artifact is released under the MIT License. Review the license before
public release if a different open-source policy is required.
