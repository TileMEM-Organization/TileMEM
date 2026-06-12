# TilePO Priority Disclosure

**Disclosure date:** 2026-06-11<br>
**Project:** TileMEM<br>
**Algorithm/system:** TilePO<br>
**Primary release target:** `v0.1-priority-2026-06-11`<br>
**Current PDF-included record:** `v0.1.1-pdf-2026-06-12`

## Disclosure Statement

TilePO is a BF16 profile-guided tile-level placement/admission system for MoE
serving. It uses workload/profile evidence and fixed expert budgets to choose
how expert weights should be admitted, retained, or organized at tile granularity
while preserving a KT-native BF16 serving path.

The purpose of the V0.1 priority release is to establish a public priority
record for the TilePO idea, implementation shape, benchmark evidence, public
manifests, and reproducibility path. The later V0.1.1 PDF-included record adds
the generated technical report PDF to the archived artifact chain while
preserving the original V0.1 priority anchor. These records are not dependent on
arXiv, ChinaXiv, OSF, or TechRxiv acceptance.

## Safe Priority Claim

To the best of the author's knowledge, TilePO is among the first open artifact
systems to publicly disclose and evaluate BF16 profile-guided tile-level
placement/admission for MoE serving under same expert-budget KT baselines.

Chinese wording:

据作者所知，TilePO 是较早公开披露并开源评测 BF16 条件下、面向 MoE
推理的 profile-guided tile-level placement/admission 系统之一，并在同专家
budget 的 KT baseline 下给出真实实验、消融和可复现 artifact。

## What Is Disclosed

- Problem: MoE serving needs VRAM/DRAM placement decisions beyond dense-model
  GPU scheduling and beyond coarse expert-level residency alone.
- Method: profile-guided tile-level placement/admission for MoE experts.
- Interface: `.tmem` DSL, typed MIR, compiled manifest, runtime metrics, and
  BF16 serving wrappers.
- Evidence: V0.1 BF16 same-budget ablation over `mixed` and `long_context`.
- Reproducibility: offline manifest verification plus real-run wrappers with
  replaceable model paths.

## What Is Not Claimed

- TilePO does not claim full native CUDA replacement of KT/SGLang MoE kernels.
- TilePO does not claim FP8 or MXFP4 serving-quality gains.
- TilePO does not claim universal wins across all models, GPUs, workloads, or
  serving systems.
- TilePO does not claim that fine-grained tile splitting alone explains all
  performance gains.
- TilePO does not make an absolute "first tile idea in history" claim.

## V0.1 Evidence Summary

```text
Rows: 210 / 210 real success
Gate: PASS
Workloads: mixed, long_context
Expert budgets: 2, 4, 6, 8, 10
Repeats: 3
Request count: 5
Serving precision: BF16 / KT-native path
```

V0.1 supports the following precise claim:

> Under the V0.1 BF16 same-budget matrix, TilePO policies outperform KT
> expert-level placement on `mixed` and `long_context`, while reporting VRAM,
> CPU/DRAM, p95, p99, and policy boundaries.

## Public Identifiers

Original priority anchor, 2026-06-11:

```text
GitHub repository: https://github.com/TerminusAkivili/TileMEM
GitHub release:    https://github.com/TerminusAkivili/TileMEM/releases/tag/v0.1-priority-2026-06-11
Git tag:           v0.1-priority-2026-06-11
Zenodo DOI:        https://doi.org/10.5281/zenodo.20646195
Concept DOI:       10.5281/zenodo.20646194
SWHID:             swh:1:snp:073ee68e366c28f478e81db109056b68f9b146ab
Tarball SHA256:    4592f09fb451c5d0fe998d9f4fb83ab774100ddba72dc580ef1c5772b7b70f3b
```

Current PDF-included record, 2026-06-12:

```text
GitHub release:    https://github.com/TerminusAkivili/TileMEM/releases/tag/v0.1.1-pdf-2026-06-12
Git tag:           v0.1.1-pdf-2026-06-12
Git commit:        b864a2375e7f05ce41200a43c90c2016fd738590
Zenodo DOI:        https://doi.org/10.5281/zenodo.20648132
Concept DOI:       10.5281/zenodo.20646194
SWHID:             swh:1:snp:1b8b452d7fc26b52c53ae79a2dcfed0e98984389
PDF SHA256:        97aef412b58835eeed318aeb4a439aec4e87990252f92dfb0e2adac1cae770d7
Tarball SHA256:    e48a32ddfa081a7ab4a327fa56044abd6eb8c0abba75666e5db8438ea91db5dc
```

## Verification

Offline verification:

```bash
bash examples/quickstart_offline.sh
```

Real BF16 run with a user-supplied model:

```bash
export TILEMEM_MODEL_PATH=/path/to/moe/checkpoint
bash examples/quickstart_custom_model.sh
```
