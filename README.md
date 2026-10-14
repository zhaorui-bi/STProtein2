# STProtein2

STProtein2 predicts a fixed panel of **31 spatial protein targets from RNA and spatial coordinates**, using a frozen spatial anchor and prototype-guided residual corrections. This repository contains the standalone implementation of the current **v2.1 architecture**, extracted from [STProtein](https://github.com/zhaorui-bi/STProtein).

![STProtein2 framework](fig/framework.png)

The current configuration uses an eight-member RNA ensemble, 32 state prototypes, 16 niche prototypes, four cooperative sheaf experts, hierarchical reference-only initialization, C0 in-fit residual targets, no EMA teacher, and full-graph execution. Its prediction is:

$$\widehat{Y}=B_{\mathrm{sp}}+\gamma_{\mathrm{state}}\Delta_{\mathrm{state}}+\gamma_{\mathrm{local}}\Delta_{\mathrm{local}}.$$

The independently fitted Ridge anchor and fixed centered spatial basis form $B_{\mathrm{sp}}$. State and local corrections have separate binary switches for each protein, selected using reference-only held-out evaluation. The switches require at least four validation units, mean relative MSE improvement of at least 1%, and improvement in at least 60% of units. Closed switches reproduce the frozen spatial base exactly.

This is a research implementation. Independent confirmation of neural residual gains and superiority over competing methods remains in progress. The framework illustration is schematic and contains no benchmark measurements.

## Install

Python 3.10 or newer is required. PyTorch, NumPy, SciPy, scikit-learn, pandas, AnnData, joblib and threadpoolctl are declared in `pyproject.toml`.

```bash
git clone https://github.com/zhaorui-bi/STProtein2.git
cd STProtein2
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

The default full-graph configuration does not require PyTorch Geometric, METIS or Leiden. Optional partition controls can be installed with `python -m pip install -e '.[test,partition]'`.

## Run a small standalone example

The example generates synthetic RNA and reference ADT files locally. Query ADT is absent. It reduces model size and uses three epochs to exercise installation, nested training, export and RNA-only inference; it is not an accuracy benchmark. Three reference groups are insufficient to open the validation switches, so the selected prediction falls back to its spatial base.

```bash
python examples/make_demo.py --output demo/data

stprotein2-train run \
  --registry demo/data/registry.json --groups demo/data/groups.json \
  --config configs/demo.json --query query \
  --output demo/run --anchor-cache demo/anchors --device cpu

stprotein2-predict export --run demo/run --output demo/bundle

stprotein2-predict predict \
  --bundle demo/bundle --rna demo/data/query_rna.h5ad \
  --sample query --output demo/prediction --device cpu
```

## Train with paired references

RNA input is an AnnData `.h5ad` with raw nonnegative integer counts in `.X`, unique gene IDs in `.var_names`, unique spot barcodes in `.obs_names`, and two-dimensional coordinates in `.obsm['spatial']`. ADT reference files contain raw counts for the named 31-antibody [panel](docs/ARCHITECTURE.md), with matching spot barcodes. The target is row-centered `log1p` ADT expression, and predictions may be negative.

A registry is a JSON list of records:

```json
[
  {"name": "reference_a", "rna": "/path/reference_a_rna.h5ad", "adt": "/path/reference_a_adt.h5ad"},
  {"name": "reference_b", "rna": "/path/reference_b_rna.h5ad", "adt": "/path/reference_b_adt.h5ad"},
  {"name": "query", "rna": "/path/query_rna.h5ad", "adt": "/path/query_adt_not_used.h5ad"}
]
```

The registry excerpt illustrates the format; a C0 run needs at least four cohort groups. Supply `groups.json` as `{"groups": {"reference_a": "source_a", "reference_b": "source_b", "query": "held_source"}}`, including every sample. All slices sharing the query group are held out together. Use source/donor groups appropriate to the scientific question; separate slice IDs do not establish donor independence. At least four reference groups after query exclusion are needed to generate gate evidence. Every inner fit refits preprocessing and initialization on its allowed reference subset.

```bash
stprotein2-train prepare \
  --registry /path/registry.json --groups /path/groups.json \
  --config configs/latest.json --query query \
  --output runs/query --anchor-cache cache/anchors --device cpu

stprotein2-train run \
  --registry /path/registry.json --groups /path/groups.json \
  --config configs/latest.json --query query \
  --output runs/query --anchor-cache cache/anchors --device cuda \
  --cuda-allocator-gib 8
```

`prepare` builds reference-only anchors on CPU; `run` also works without a separate preparation step. Use `--device cpu` for CPU training. The allocator setting caps the process's CUDA allocator; it does not guarantee that a particular dataset will fit.

`configs/latest.json` matches the current algorithm and sets `anchor_cache_policy` to `fit` for a fresh clone. The archived `balanced_C0_hierarchical_none_v1.json` retains the original `reuse_frozen` policy and requires a completed matching anchor cache.

## Export and predict

```bash
stprotein2-predict export --run runs/query --output exports/query_model

stprotein2-predict predict \
  --bundle exports/query_model --rna /path/new_slice_rna.h5ad \
  --sample new_slice --output predictions/new_slice --device cpu
```

Prediction requires only the exported bundle and one complete RNA slice. Output files include named `prediction.h5ad`, `prediction.npz`, component predictions, gene-coverage diagnostics and integrity manifests. The latest initialization and execution configuration is restored from the checkpoint. Bundles are trusted Python model artifacts; their checksums check integrity.

If query ADT is available, score completed predictions separately:

```bash
python scripts/score_extended_baseline.py --registry /path/registry.json --output runs/query
```

Additional spatial and domain metrics are available in `stprotein2_dpas_evaluation`.

## Validation and provenance

```bash
python -m pytest -q
```

Tests cover frozen-anchor behavior, nonzero corrections, sheaf operators, reference/query separation, nested epoch and gate selection, checkpoint recovery, full/partition parity, model export and RNA-only prediction. Optional native partition and CUDA tests depend on their runtimes. [Release validation](docs/RELEASE_VALIDATION.md) records the actual checks completed for this snapshot.

The release includes the current algorithm and the training/deployment dependency chain. Historical method alternatives required by that chain retain their original module names. Shared baseline helpers are used for artifact verification; external competitor training adapters require the original research workspace and are not standalone benchmark entry points here. Large datasets, fitted checkpoints, run caches, third-party source checkouts and the original STProtein model are not distributed.

The [source snapshot](docs/source_snapshot.json) records the original files and extraction changes. Dated protocol documents preserve development provenance; `configs/latest.json` and [the architecture description](docs/ARCHITECTURE.md) specify the configuration presented here. The framework [legend](fig/framework.caption.md) and [image-generation prompt](fig/framework.prompt.txt) accompany the figure.

## References and license

- [STProtein: predicting spatial protein expression from multi-omics data](https://arxiv.org/abs/2602.05811), Jiang et al., 2026.
- [Cooperative Sheaf Neural Networks](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ea1a7f7bc0fc14142106a84c94c826d0-Abstract-Conference.html), Ribeiro et al., ICLR 2026. The tensor implementation adapts the published operator; CSNN itself is prior work.
- [Upstream implementation notes](docs/UPSTREAM.json) describe the independent BatchEnsemble-style RNA adaptation and preprocessing lineage.

The source retains the [GPL v3 license](LICENSE) of the original repository.
