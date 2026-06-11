# TileMEM / TilePO V0.1 Priority Roadmap

**Date:** 2026-06-11  
**Project name:** TileMEM  
**Algorithm/system name:** TilePO  
**Release target:** `v0.1-priority-2026-06-11`  
**Primary goal:** establish a public, citable, and verifiable priority record for TilePO without depending on edu email, arXiv endorsement, ChinaXiv access, OSF, or TechRxiv acceptance.

## 1. Positioning

TileMEM is the project. TilePO is the algorithm/system inside TileMEM.

TilePO should be presented as:

> A BF16 profile-guided tile-level placement/admission system for high-throughput MoE serving.

The V0.1 release is not a final conference submission and not a full native CUDA
MoE replacement claim. It is a priority disclosure and artifact release. Its job
is to make the idea, implementation shape, benchmark evidence, public manifests,
and reproducibility path public under stable timestamps and content hashes.

## 2. Why This Route

The normal preprint path is blocked or uncertain because:

- arXiv requires endorsement for new authors in many categories;
- ChinaXiv may require account/email conditions that are not currently
  available;
- OSF and TechRxiv can still involve moderation, format, domain, or identity
  friction.

Therefore V0.1 uses an independent artifact-first priority path:

1. GitHub public repository.
2. GitHub tag and release.
3. Zenodo DOI for the release archive.
4. Software Heritage SWHID plus SHA256 checksums.

These four steps do not require a school email or advisor. They also create a
strong public evidence chain: timestamped source, release archive, DOI, permanent
software snapshot, and checksum-verifiable files.

## 3. Priority Claim

The release should avoid fragile "first in the world" language. The safe and
strong claim is:

> To the best of the author's knowledge, TilePO is among the first open artifact
> systems to publicly disclose and evaluate BF16 profile-guided tile-level
> placement/admission for MoE serving under same expert-budget KT baselines.

Chinese version:

> 据作者所知，TilePO 是较早公开披露并开源评测 BF16 条件下、面向 MoE
> 推理的 profile-guided tile-level placement/admission 系统之一，并在同专家
> budget 的 KT baseline 下给出真实实验、消融和可复现 artifact。

This wording protects the priority record while avoiding an absolute novelty
claim that could be attacked by broad prior art around MoE placement, offloading,
caching, tiling, prefetching, or memory hierarchy scheduling.

## 4. Scope of V0.1

V0.1 should include:

- TilePO method definition.
- MIR / manifest / placement/admission design.
- BF16-only fair comparison statement.
- V0.1 ablation evidence.
- Public manifests and summary JSON files.
- Reproducibility scripts.
- Claim checklist.
- SHA256 checksums.

V0.1 should not claim:

- full native CUDA replacement of KT/SGLang MoE kernels;
- FP8 or MXFP4 serving-quality gains;
- universal win across all models, GPUs, or serving systems;
- that fine-grained tile splitting alone explains all wins;
- patent-like absolute priority over all tile, cache, or offload ideas.

## 5. Evidence to Preserve

The V0.1 priority packet should preserve the V0.1 ablation because it is the
strongest fair evidence so far:

```text
Workloads: mixed, long_context
Experts: 2, 4, 6, 8, 10
Repeats: 3
Request count: 5
Rows: 210 / 210 real success
Gate: PASS
Serving precision: BF16 / KT-native path
```

Summary of the safe V0.1 result:

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

Interpretation:

> TilePO wins in the V0.1 BF16 same-budget matrix by selecting better
> profile-guided tile/admission policies than KT expert-level placement. The
> data supports adaptive tile policy scheduling, not a blanket claim that finer
> tiles are always better.

## 6. Artifact Layout

Create a clean priority package:

```text
publish/TileMEM_TilePO_V0_1_20260611/
  README.md
  PRIORITY_DISCLOSURE.md
  RELEASE_NOTES.md
  CLAIM_CHECKLIST.md
  SHA256SUMS
  paper/
    tilemem_tilepo_v0_1_technical_report.md
  docs/
    tilemem_tilepo_v0_1_priority_roadmap.md
    tilemem_tilepo_ablation_report.md
  configs/
    models/
      olmoe_1b_7b_example.tmem
      model_template.tmem
    workloads/
      mixed.json
      long_context.json
  evidence/
    ablation/
      tilepo_ablation_report.md
      tilepo_ablation_summary.json
      tilepo_ablation_manifest.json
      tilepo_ablation.completed.json
  scripts/
    reproduce_ablation.sh
    reproduce_with_model.sh
    verify_artifact.sh
  examples/
    quickstart_offline.sh
    quickstart_custom_model.sh
```

The root repository should also contain the source tree, tools, configs, and
tests needed to make the release credible. The publish packet is the clean
reader-facing subset.

## 7. Out-of-Box Model Replacement Interface

"Out of the box" means the repository can be used in two modes:

1. offline artifact verification without a GPU or model checkpoint;
2. real BF16 serving evaluation with a user-supplied MoE model path.

The public repository should not hard-code the original local model path as the
only usable path. It should expose a small model replacement interface:

```bash
export TILEMEM_MODEL_PATH=/path/to/moe/checkpoint
export TILEMEM_MODEL_NAME=olmoe_1b_7b
export TILEMEM_PLAN=configs/models/olmoe_1b_7b_example.tmem
export TILEMEM_WORKLOAD=mixed
export TILEMEM_EXPERTS=2,4,6,8,10
export TILEMEM_OUT_DIR=build/custom_model_run
```

The expected user-facing commands are:

```bash
bash examples/quickstart_offline.sh
bash examples/quickstart_custom_model.sh
```

`quickstart_offline.sh` should verify the released evidence from existing
manifests:

- install/import the Python package;
- read the V0.1 merged manifest;
- regenerate or validate the V0.1 report;
- check gate status;
- check SHA256 files.

`quickstart_custom_model.sh` should run a real evaluation only when the user
provides a model checkpoint:

```bash
bash scripts/reproduce_with_model.sh \
  --model-path "$TILEMEM_MODEL_PATH" \
  --plan "$TILEMEM_PLAN" \
  --workloads "$TILEMEM_WORKLOAD" \
  --experts "$TILEMEM_EXPERTS" \
  --out-dir "$TILEMEM_OUT_DIR"
```

The model interface should be conservative:

- default to BF16;
- require the user to pass the model path explicitly;
- keep the model config in `.tmem` rather than hidden Python constants;
- record model path, model name, workload, expert budget, GPU peak, and
  CPU/DRAM peak in the manifest;
- fail clearly if the model is not a compatible MoE checkpoint;
- never silently switch to FP8/MXFP4.

This makes TilePO usable by other developers: they can keep the TilePO
placement/admission pipeline while replacing the model, workload, expert
budgets, and output directory.

## 8. Step 1: GitHub Public Repository

Goal: make TileMEM/TilePO publicly visible with source, reports, and evidence.

Repository name:

```text
TileMEM
```

Recommended repository subtitle:

```text
TilePO: BF16 profile-guided tile-level placement/admission for MoE serving.
```

Root `README.md` should include:

- one-sentence project definition;
- V0.1 priority disclosure statement;
- V0.1 headline result table;
- "works without GPU" offline verification command;
- "bring your own MoE model" command using `TILEMEM_MODEL_PATH`;
- quick reproduction commands;
- artifact paths;
- claim boundaries;
- citation block;
- license.

Minimum push path:

```bash
git remote add origin git@github.com:TerminusAkivili/TileMEM.git
git add README.md docs publish tilepo tools configs kernels pyproject.toml CMakeLists.txt
git commit -m "Release TileMEM TilePO v0.1 priority artifact"
git push -u origin main
```

If the repository already has a remote, use:

```bash
git remote -v
git push origin main
```

## 9. Step 2: GitHub Tag and Release

Goal: create a public timestamped release that points to a fixed code state and
artifact package.

Tag:

```text
v0.1-priority-2026-06-11
```

Tag command:

```bash
git tag -a v0.1-priority-2026-06-11 \
  -m "TileMEM TilePO v0.1 priority disclosure"
git push origin v0.1-priority-2026-06-11
```

Release title:

```text
TileMEM TilePO v0.1: Priority Disclosure and BF16 MoE Serving Artifact
```

Release summary:

```text
This release publicly discloses TilePO, a BF16 profile-guided tile-level
placement/admission system for MoE serving. It includes source code, a technical
report, V0.1 evidence, public manifests, reproducibility scripts, and SHA256
checksums. The release does not claim full native CUDA MoE replacement or
low-bit serving-quality gains.
```

Optional CLI release command:

```bash
gh release create v0.1-priority-2026-06-11 \
  publish/TileMEM_TilePO_V0_1_20260611.tar.gz \
  publish/TileMEM_TilePO_V0_1_20260611.tar.gz.sha256 \
  --title "TileMEM TilePO v0.1: Priority Disclosure and BF16 MoE Serving Artifact" \
  --notes-file publish/TileMEM_TilePO_V0_1_20260611/RELEASE_NOTES.md
```

## 10. Step 3: Zenodo DOI

Goal: archive the GitHub release and obtain a DOI.

Implementation path:

1. Log in to Zenodo.
2. Connect GitHub.
3. Enable the `TileMEM` repository.
4. Create or re-sync the GitHub release.
5. Confirm Zenodo minted a DOI.
6. Add the DOI badge and citation entry back to `README.md`.

Recommended citation after DOI exists:

```bibtex
@software{tilemem_tilepo_v0_1_2026,
  title  = {TileMEM TilePO v0.1: BF16 Profile-Guided Tile-Level Placement/Admission for MoE Serving},
  author = {TerminusAkivili},
  year   = {2026},
  version = {v0.1-priority-2026-06-11},
  doi    = {ZENODO_DOI_AFTER_PUBLICATION},
  url    = {https://github.com/TerminusAkivili/TileMEM}
}
```

If Zenodo has trouble syncing from GitHub, manually upload the release tarball
and SHA256 file as a software artifact. The DOI still serves the priority goal.

## 11. Step 4: Software Heritage, SHA256, and Optional Timestamping

Goal: produce content-level permanence and verification.

Software Heritage:

1. Wait until the GitHub repository is public.
2. Submit the repository URL to Software Heritage Save Code Now.
3. Record the resulting SWHID in:
   - `README.md`
   - `PRIORITY_DISCLOSURE.md`
   - `PUBLISH_MANIFEST.md`

SHA256:

```bash
sha256sum publish/TileMEM_TilePO_V0_1_20260611.tar.gz \
  > publish/TileMEM_TilePO_V0_1_20260611.tar.gz.sha256

find publish/TileMEM_TilePO_V0_1_20260611 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > publish/TileMEM_TilePO_V0_1_20260611/SHA256SUMS
```

Optional extra timestamp:

- timestamp the final PDF hash;
- timestamp the release tarball hash;
- timestamp the merged V0.1 manifest hash.

This is optional because GitHub release, Zenodo DOI, SWHID, and SHA256 already
provide a strong public record.

## 12. Packaging Language

Use this short public description:

```text
TileMEM is an open MoE serving optimization project. TilePO is its BF16
profile-guided tile-level placement/admission system. TilePO studies when
expert weights should be admitted, retained, or organized at tile granularity
under fixed GPU expert budgets, and reports both wins and boundaries against
KT expert-level placement.
```

Use this short Chinese description:

```text
TileMEM 是一个面向 MoE 推理服务的开源优化项目。TilePO 是其中的 BF16
profile-guided tile-level placement/admission 系统，用于在固定 GPU expert
budget 下决定专家权重如何按 tile 粒度进入、保留和组织，并与 KT expert-level
placement 进行公平对比。
```

Avoid:

```text
TilePO proves fine-grained tiles are always better.
TilePO replaces KT/SGLang MoE kernels end to end.
TilePO is the first tile idea in history.
TilePO wins on every model and GPU.
```

## 13. Release Checklist

Before publishing:

- [ ] Root README names TileMEM as project and TilePO as algorithm/system.
- [ ] README includes offline verification and custom-model quickstart commands.
- [ ] Model path is controlled by `TILEMEM_MODEL_PATH` or an explicit
      `--model-path` flag.
- [ ] Example `.tmem` model config and model template are included.
- [ ] `PRIORITY_DISCLOSURE.md` states the idea and claim boundary.
- [ ] V0.1 report and public manifests are included.
- [ ] BF16-only fairness is explicitly stated.
- [ ] Low-bit probes are not promoted as serving gains.
- [ ] Release tarball is generated.
- [ ] SHA256 files are generated.
- [ ] `git status` is reviewed so unrelated local files are not accidentally
      published.
- [ ] License is selected.
- [ ] Author name, affiliation line, and contact email are correct.

After publishing:

- [ ] GitHub repository is public.
- [ ] GitHub tag exists.
- [ ] GitHub release exists.
- [ ] Zenodo DOI exists.
- [ ] Software Heritage SWHID exists.
- [ ] README contains DOI and SWHID.
- [ ] Release URL, DOI, SWHID, and SHA256 are recorded in
      `PRIORITY_DISCLOSURE.md`.

## 14. Success Definition

V0.1 is complete when an external reader can verify all of the following without
private chat logs or local-only files:

1. TilePO's method was publicly disclosed.
2. The release date is visible.
3. The source and artifacts are downloadable.
4. The V0.1 evidence is present.
5. The release has either a DOI or a public GitHub release URL.
6. The artifact hash matches the published SHA256.
7. The code snapshot can be independently archived or identified by SWHID.
8. A new user can run offline verification without the original local model.
9. A GPU user can replace the model path through the documented model interface.

At that point, TilePO has a public priority record even if arXiv, ChinaXiv, OSF,
and TechRxiv remain unavailable.

## 15. Immediate Next Actions

Recommended next implementation order:

1. Create `publish/TileMEM_TilePO_V0_1_20260611/`.
2. Write `PRIORITY_DISCLOSURE.md`, `CLAIM_CHECKLIST.md`, and `RELEASE_NOTES.md`.
3. Copy V0.1 report, summary, manifest, and completion JSON into the package.
4. Add `configs/models/model_template.tmem` and an OLMoE example config.
5. Add offline and custom-model quickstart scripts.
6. Add or update root `README.md` for public release.
7. Generate tarball and SHA256.
8. Review package contents.
9. Push to GitHub.
10. Create tag and release.
11. Mint Zenodo DOI.
12. Save to Software Heritage.

V0.1 should freeze the current priority evidence. Further engineering work such
as V7/V8 optimization should happen after this record is public.
