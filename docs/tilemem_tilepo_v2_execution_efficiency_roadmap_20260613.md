# TileMEM / TilePO V2 Execution Efficiency Roadmap

**Date:** 2026-06-13  
**Scope:** near-term TileMEM / TilePO runtime roadmap  
**Theme:** decouple fine-grained memory placement from hardware-efficient execution

## 1. Motivation

TilePO V0.1 shows that profile-guided tile-level placement/admission can beat
same-budget KT expert-level placement in the BF16 evidence matrix. The next
systems question is not whether tiles can be made smaller. The next question is
how TileMEM can use fine-grained placement without paying unnecessary execution
overhead.

The V2 direction is therefore:

> Make memory placement finer while keeping transfer, dispatch, and GEMM
> execution units large, predictable, and hardware-efficient.

This roadmap deliberately focuses on execution efficiency and avoids expanding
V2 into broader hardware or model-quality claims.

## 2. Workstream A: Adaptive Tile Granularity

V0.1 uses fixed policy variants such as coarse, fine, and hybrid tiles. V2
should make tile granularity adaptive rather than uniform across all experts.

Proposed policy shape:

| Region | Tile granularity | Reason |
| --- | --- | --- |
| hot experts | coarse | preserve GEMM/grouped-GEMM efficiency and reduce metadata overhead |
| warm experts | medium | balance reuse, residency, and dispatch cost |
| cold experts | fine | reduce VRAM footprint and improve admission flexibility |
| tail-latency mode | finer | admit only the required working set and pre-position smaller pieces |
| throughput mode | coarser | favor large execution units and stable grouped dispatch |

The key claim should be conservative:

> TileMEM should not make every expert equally fine-grained; it should choose
> granularity based on reuse, residency pressure, and the serving objective.

This directly addresses the criticism that overly fine tiles may hurt
throughput. Fine granularity is useful where it improves placement flexibility;
coarse granularity remains useful where execution efficiency dominates.

## 3. Workstream B: Runtime Tile Coalescing

Adaptive placement should not force equally fine-grained execution. Runtime tile
coalescing should merge multiple small admitted tiles into larger transfer and
dispatch units whenever the runtime can do so without changing residency
correctness.

Expected benefits:

- fewer transfer descriptors;
- fewer dispatch records;
- lower metadata overhead;
- fewer small kernel launches;
- better grouped-GEMM and batched execution efficiency;
- ability to keep cold-placement flexibility without fragmenting hot execution.

The core principle is:

> Fine-grained placement does not require fine-grained execution.

In V2, a tile can remain the accounting and placement unit while a coalesced
group becomes the transfer, prefetch, or execution unit.

## 4. Workstream C: Bubble-Aware Async Planning

TilePO planning should avoid the critical path. V2 should hide placement
planning behind GPU execution bubbles and use recent routing information to
speculatively prepare the next tile map.

Proposed mechanism:

- maintain a rolling histogram over the most recent `N` routing steps;
- generate a speculative next-step tile map from the histogram;
- keep double-buffered tile maps;
- execute the current step with map `A`;
- asynchronously prepare map `B`;
- swap maps only at safe boundaries;
- fall back to the current stable plan when prediction confidence is low.

This workstream targets throughput loss from CPU decision making and GPU
synchronization. It is separate from the memory-capacity benefit of tile
placement.

The intended V2 claim is:

> Bubble-aware async planning hides TilePO decisions behind existing execution
> gaps instead of inserting a CPU planning dependency into the serving critical
> path.

## 5. Suggested V2 Evaluation

V2 should be evaluated against the same-budget KT baseline and the V0.1 TilePO
policies. Minimum useful evidence:

- throughput and p95/p99 latency for fixed expert budgets;
- metadata overhead and tile-map planning overhead;
- number of dispatch records before and after coalescing;
- transfer count and transferred bytes;
- grouped-GEMM batch shape distribution;
- planning time on and off the critical path;
- ablations for adaptive granularity, coalescing, and bubble-aware planning.

The success criterion is not only higher token throughput. V2 should show that
TileMEM can preserve most of the memory-placement benefit of fine tiles while
recovering the execution efficiency of larger units.

## 6. Summary

V2 should make TileMEM more mature by separating three concepts:

1. **placement unit:** the small object used for VRAM/DRAM admission;
2. **transfer unit:** the coalesced object moved across the memory hierarchy;
3. **execution unit:** the grouped object dispatched to GEMM or serving kernels.

This separation is the cleanest answer to the central systems objection:

> TileMEM does not need to choose between fine placement and efficient execution.
> It can use fine placement where memory pressure demands it and coalesce tiles
> when the hardware needs larger, more regular work units.
