# STProtein2 v2.1 benchmark and evidence-driven optimization

Goal: assess the latest executable model on the established paired-data
benchmarks across error, correlation and spatial metrics, then improve failures
with reference-only selection. SOTA is unproven until complete matched coverage,
appropriate competitors and independent confirmation support it. No artificial
target score or guaranteed dominance is assumed.

## Frozen candidate

Use `balanced_C0_hierarchical_none_v1.json`: 4000 genes, 31 signed centered-log
antibodies, fixed alpha=.5 spatial anchor, 8-member Student, 32 state/16 niche
codes, protein-conditioned CSNN, hierarchical source/slice/unique-spot
initialization, weight-side modulation and representation checkpointing. No
Teacher for this registered candidate; the EMA version remains a matched control.
Seeds2025/2026/2027, maximum80 epochs, warm8/ramp8, validation every4,
patience6 and the existing four-choice reference OOF gate rule.

## Two established tracks

1. README seven slices / four conservative source groups. All GEO slices stay
   together. Query-group holdout leaves three reference groups, below the frozen
   gate requirement of four. Both neural gates therefore close. The actual
   Student is still trained with reference-selected epochs, and the final
   prediction must equal its fixed spatial B0 exactly. This track tests the
   latest deployable behavior; it cannot demonstrate a prototype benefit.
2. Extended ten slices / six conservative source groups. Hold out all slices of
   the query group. Five remaining groups permit the existing strict nested
   gate/epoch selection (31 neural fits per outer,186 per seed). No eligibility,
   loss, target, gate threshold or preprocessing changes relative to v2.1.

The benchmark adapter permits in-fit C0 training on two source groups in the
first track's epoch-selection fits. Ridge has its own internal source CV and
can train there. This is a cohort-feasibility adaptation, not a different C0
loss. Cross-fitted CF0 still requires at least three sources per neural fit and
six at cohort level. The old matched C0/CF0 namespace remains unchanged.

Priority: seed2025 README track, seed2025 extended track, remaining registered
seeds on both tracks. Every arm/query runs independent of its interim accuracy.
Query proteins are neither opened nor hashed by prediction. Only the separate
scorer reads frozen query ADT after the complete prediction marker. Retain all
negative outcomes; repeated development inspection does not become blind testing.

## Metrics and reporting

Recompute raw-label RMSE, MAE, mean per-protein Pearson and Spearman, spatial
edge-difference RMSE, per-protein bias/amplitude and slice-mean/within-slice MSE.
Use the same named spot/protein axes, graph and exact query/reference set.
Average slices within a source, then sources equally; seeds are optimization
repeats, not independent donors. Report undefined correlations explicitly.
Primary endpoint remains source-macro RMSE. Different metric winners are shown
without claiming that all metrics share one winner. Partial coverage is paired
to the exact same controls and excluded from complete-coverage global ranking.

Include available same-track Ridge, spatial B0, MLP, official TabM, RNA0,
DPAS-Graph, DGAT, previous STProtein2 variants and native-component MultiSpaceNet.
Original leave-slice three-seed DPAS paper-style runs form a different protocol
and cannot be pooled into the source-held common-input table.

## Resources and restart

Single controller lock; immutable code/config/input manifests and bounded fresh
OOM retries. Shared A40 GPU admission requires at least4608 MiB free in two
observations; each new worker sets a4 GiB PyTorch allocator ceiling before model
work. Record free memory, device and limits. Do not modify or stop the unrelated
ThetaCheck process. A terminal child and a fresh CUDA OOM are required before
retrying. An observation timeout never authorizes a duplicate job. Completed
fits are validated and reused. CPU jobs may proceed independently.

Shared GPU timings describe execution, not a speed ranking. Existing CPU and
CUDA correctness tests support the representation kernel; no65k original-kernel
parity or end-to-end streaming claim is added here.

## Optimization after diagnosis

Keep benchmark outcomes and subsequent candidates in new versioned artifacts.
Use source-held reference validation for initialization, Teacher, residual loss
or anchor alternatives. Changes motivated by outer development error require a
new independent paired confirmation after final model freezing. The failed
low-dimensional context experiment does not justify expanding B3 on the same
data. Do not use query protein statistics, remove difficult proteins/sources,
change metric definitions or relax gates to obtain a favorable ranking.
