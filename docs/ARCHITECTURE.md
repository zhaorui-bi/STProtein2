# Current STProtein2 architecture

The current v2.1 configuration is `configs/latest.json`. It uses C0 in-fit training residual targets, reference-only hierarchical initialization, no EMA teacher, and full-graph execution. The 8 RNA members, 32 state prototypes, 16 niche prototypes and 4 CSNN experts are configuration defaults, not fixed mathematical requirements.

## Prediction components

1. Fit RNA feature selection, normalization, standardization and QC transformations using allowed references only. Gene availability distinguishes a missing feature from a measured zero.
2. Fit an independent Ridge RNA-to-protein anchor with reference-only model selection. Freeze its weights and bias during neural training.
3. Add the fixed spatial basis: `B_sp = B_RNA + 0.5 * center(neighbor_mean(B_RNA) - B_RNA)`. Center over non-isolated spots and assign zero spatial correction to isolated spots.
4. Encode RNA, availability and QC with the shared ensemble. The node-wise RNA encoder does not receive the spatial graph. Its averaged representation is `h`.
5. State memory uses soft assignments from `h`. Niche memory uses `h`, neighboring `h`, neighboring state assignments and QC. Own-niche memory replaces neighboring inputs with the spot's own representation and assignments. Prototypes represent learned expression states, not annotated cell types.
6. State residual is a QC-gated difference between `f_state(h, M_state)` and `f_state(h, 0)`.
7. Each cooperative sheaf expert produces a difference between its graph-context and self-only computation. Spot/state/niche features and protein embeddings determine a separate expert mixture for each spot and protein. Add the neighborhood-versus-own-niche readout difference, apply the local QC gate, and center across non-isolated spots.
8. Select binary `gamma_state` and `gamma_local` per protein from reference-held-out evidence. Evaluate Base/State/Local/Both choices. A correction requires at least four validation units, mean relative MSE gain of at least 1%, and positive gains in at least 60% of units. If both switches are zero, the output equals `B_sp` exactly.

The neural direct protein head is auxiliary training supervision; it is not the deployed frozen base. In the no-teacher configuration, the retained teacher-related configuration fields do not activate an EMA teacher. Optional partition execution and teacher alternatives remain separately configured code paths.

## Data separation

The outer query group is excluded from all reference fitting, initialization, epoch selection and gate selection. Every gate fold also excludes its held group from its model's training and inner early stopping. Query RNA and coordinates are available for prediction. Query ADT is read only by a separate scoring command.

Group labels specify the unit of evaluation. Source-held and slice-held protocols answer different questions. Multiple slices from one donor do not constitute independent donors.

## Target panel

The default panel is sorted in this order:

```text
ACTA2 BCL2 CCR7 CD14 CD163 CD19 CD27 CD274 CD3E CD4 CD40 CD68
CD8A CEACAM8 CR2 CXCR5 EPCAM FCGR3A HLA_DRA ITGAM ITGAX KRT5
MS4A1 PAX5 PCNA PDCD1 PECAM1 PTPRC_1 PTPRC_2 SDC1 VIM
```

The protein reader applies `log1p` to raw ADT counts and subtracts each spot's mean across these 31 targets. Predictions are signed values in this target space.

## Implementation locations

| Component | Module |
|---|---|
| Data contract and preprocessing | `stprotein2/data.py` |
| RNA ensemble and prototype memories | `stprotein2/models/prototypes.py` |
| State, niche and expert-mixture residuals | `stprotein2/models/residual.py` |
| Cooperative sheaf propagation | `stprotein2/models/sheaf.py` |
| Independently frozen anchor | `stprotein2/models/anchor.py` and `stprotein2/training/anchors.py` |
| Fixed spatial base | `stprotein2/models/spatial.py` |
| Hierarchical initialization | `stprotein2/training/initialization.py` |
| Current neural training | `stprotein2/training/` |
| Current nested cohort runner | `stprotein2/training/nested.py` |
| Validation switches | `stprotein2/gates.py` |
| Model export and RNA-only prediction | `stprotein2/inference/` |

Structural differences from earlier methods are implemented. Their individual accuracy gains require matched ablations and independent evaluation; this snapshot does not establish SOTA or improvement for every dataset/protein/metric.
