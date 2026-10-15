# Public interface

## Commands

| Command | Purpose |
|---|---|
| `stprotein2 prepare` | Prepare independent reference-only anchors |
| `stprotein2 train` | Fit the current model, select gates and predict the held query group |
| `stprotein2 export` | Export a completed run into a self-contained bundle |
| `stprotein2 predict` | Predict one complete RNA slice using a bundle |
| `stprotein2 score` | Score frozen query predictions using measured ADT |

`python -m stprotein2` accepts the same commands. `stprotein2-train run` and `stprotein2-predict export|predict` remain console aliases.

## Python

```python
from stprotein2 import STProtein2, ModelConfig
from stprotein2.inference import export_run, load_bundle, predict_file

# Low-level model structure; a trained checkpoint supplies fitted weights/gates.
model = STProtein2(ModelConfig())

export_run("runs/query", "exports/query_model")
predict_file(
    "exports/query_model", "new_slice_rna.h5ad", "predictions/new_slice",
    sample="new_slice", device="cpu",
)
```

## Registry and groups

A registry is a JSON list with `name`, `rna` and `adt` fields:

```json
[
  {"name": "reference_a", "rna": "/path/reference_a_rna.h5ad", "adt": "/path/reference_a_adt.h5ad"},
  {"name": "reference_b", "rna": "/path/reference_b_rna.h5ad", "adt": "/path/reference_b_adt.h5ad"},
  {"name": "query", "rna": "/path/query_rna.h5ad", "adt": "/path/query_adt_not_used.h5ad"}
]
```

This excerpt illustrates the format. The current C0 cohort runner needs at least four cohort groups. Supply grouping as `{"groups": {"reference_a": "source_a", "reference_b": "source_b", "query": "held_source"}}`, including every sample. All slices in the query group are held out together. Reference ADT is required for fitting; query ADT can be absent during training and prediction and is needed only for scoring.

Use source/donor groups appropriate to the scientific question. Separate slice IDs do not establish donor independence. At least four reference groups after query exclusion are needed to generate gate evidence; otherwise the switches remain closed. Every inner fit refits preprocessing and initialization using its allowed references.

The archived configuration retains `anchor_cache_policy=reuse_frozen` and needs a matching completed cache. `configs/latest.json` uses `fit`, allowing anchors to be created on a fresh clone. Signed predictions represent row-centered `log1p` ADT over the fixed 31-target panel.

## Prediction inputs and outputs

RNA `.X` contains finite nonnegative integer counts. Gene identifiers and spot barcodes must be unique, and `.obsm['spatial']` supplies finite two-dimensional coordinates. Partial measured-gene overlap is reported; zero overlap is rejected. Supply one complete slice because cropping changes neighborhoods and centering.

Outputs include `prediction.h5ad`, `prediction.npz`, diagnostics, a manifest and a completion marker. Named protein/spot axes, component arrays and gene coverage accompany predictions. Bundles contain trusted Python preprocessing objects; checksums verify integrity.
