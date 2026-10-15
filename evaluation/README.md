# Offline evaluation

These repository tools evaluate completed predictions against measured ADT. They are excluded from the installable `stprotein2` package. Install the core with `python -m pip install -e .`, then run commands from the repository root.

```bash
python -m evaluation --registry /path/registry.json --output runs/query
```

`--output` is a completed **training run** containing `manifest.json`, `prediction_complete.json` and `predictions/<query>/prediction.npz`. The registry must identify the same RNA files/barcodes and the measured query ADT. The command does not accept the standalone `stprotein2 predict` output schema. It verifies run/prediction checksums and axes before opening query ADT. Prediction and model selection remain separate from scoring.

Results are written to `predictions/<query>/scores.json`, `development_scores.json`, `scoring/manifest.json` and `score_complete.json`. The scoring manifest records evaluation code separately from the core implementation.

| Module | Available metrics |
|---|---|
| `metrics.evaluate` | RMSE, MAE, Pearson, Spearman, bias, amplitude ratio and optional edge-difference RMSE |
| `spatial.extended_metrics` | Moran I/Geary C agreement, residual spatial diagnostics, ARI, NMI, homogeneity and completeness at prespecified cluster counts |
| `spatial.compare` | Paired query/seed comparisons with explicit wins, losses, ties and undefined values |

Extended metrics are a Python API; the command above produces the standard error/correlation report. Arrays passed to the API must already share the same spot/protein order and target transformation:

```python
from evaluation.metrics import evaluate
from evaluation.spatial import extended_metrics
from stprotein2.data import PANEL
from stprotein2.graph import physical_graph

graph = physical_graph(coordinates)
standard = evaluate(truth, prediction, PANEL, graph.edge_index)
spatial = extended_metrics(truth, prediction, graph, cluster_counts=(4, 6, 8))
```

Targets are row-centered `log1p` ADT over the fixed 31-protein panel. Undefined correlations/spatial channels are reported explicitly. Comparative experiments must declare datasets, grouped splits, seeds, metrics and model-selection budgets before interpreting results. Development-source holdouts alone do not establish independent confirmation.
