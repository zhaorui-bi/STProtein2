# Offline evaluation

These repository tools evaluate completed predictions against measured ADT. They are excluded from the core wheel. Install the core with `python -m pip install -e .`, then run commands from the repository root.

## Completed training runs

```bash
python -m evaluation run --registry /path/registry.json --output runs/query \
  --report-dir reports/query --extended --cluster-counts 4 6 8
```

`--output` identifies a completed training run containing its manifest, completion marker and `predictions/<query>/prediction.npz`. The registry must identify the original query RNA and measured ADT. `--report-dir` selects a separate report destination; omitting it preserves the earlier behavior of writing scores in the run directory. The flag-only `python -m evaluation --registry ... --output ...` remains supported.

Reports include `predictions/<query>/scores.json`, `development_scores.json`, `scoring/manifest.json` and `score_complete.json` beneath the report destination.

## Standalone predictions

```bash
python -m evaluation prediction --registry /path/application_registry.json \
  --prediction predictions/new_slice --output reports/new_slice \
  --extended --cluster-counts 4 6 8
```

`--prediction` is the complete output of `stprotein2 predict`. The registry must contain a sample whose name matches `predict --sample`, with the original RNA file and measured ADT. No bundle, training data or cohort grouping is needed. The command verifies prediction checksums, RNA identity and protein/spot axes before scoring; evaluation writes to a separate report directory and preserves the prediction files.

Outputs are `scores.json`, `manifest.json` and `score_complete.json`. An identical repeated call verifies and reuses the completed report. Changed inputs/settings require a fresh report directory; corrupted reports are rejected.

## Metrics

Without `--extended`, both commands compute error/correlation metrics. Extended scoring adds spatial and domain metrics. Prespecify cluster counts before comparing methods; counts must be distinct integers at least two and smaller than the query spot count.

| Module | Available metrics |
|---|---|
| `metrics.evaluate` | Overall RMSE/MAE, macro protein RMSE, per-protein Pearson/Spearman, mean spot Pearson, bias, amplitude ratio and edge-difference RMSE |
| `spatial.extended_metrics` | Moran I/Geary C agreement, residual spatial diagnostics, ARI, NMI, homogeneity and completeness |
| `spatial.compare` | Paired query/seed comparisons with explicit wins, losses, ties and undefined values |

`mean_pearson_complete_panel` and `mean_spearman_complete_panel` are undefined (`null`) if any protein correlation is undefined. `mean_spot_pearson` is undefined if any spot correlation is undefined. Coverage counts accompany these metrics. The older `mean_pearson` field averages defined protein correlations and remains available for diagnostics. `macro_protein_rmse` averages each protein's RMSE; the overall RMSE pools all entries.

The evaluation graph uses symmetric physical k=6/radius=2.5 neighborhoods. Cluster-specific results are named `cluster_k<k>_*`; the unqualified ARI/NMI/homogeneity/completeness fields refer to k=6 when requested. Residual spatial statistics are diagnostics. Comparison requires every declared query/seed/metric and rejects duplicate coverage, invalid directions and invalid tolerances.

Arrays passed to the Python API must already share the same spot/protein order and row-centered `log1p` ADT transformation over the fixed 31-protein panel:

```python
from evaluation.metrics import evaluate
from evaluation.spatial import extended_metrics
from stprotein2.data import PANEL
from stprotein2.graph import physical_graph

graph = physical_graph(coordinates)
standard = evaluate(truth, prediction, PANEL, graph.edge_index)
spatial = extended_metrics(truth, prediction, graph, cluster_counts=(4, 6, 8))
```

Scoring manifests record evaluation code separately from the core implementation. Comparative experiments must declare datasets, grouped splits, seeds, metrics and model-selection budgets; the software does not establish dataset independence or a performance ranking by itself.
