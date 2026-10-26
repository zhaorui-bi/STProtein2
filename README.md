# STProtein2 compact development base

The frozen compact development default is **B0: reference-fitted RNA Ridge plus the fixed physical spatial correction**. The registered six-source, three-seed comparison retained B0 under its parsimony rule; `own_mlp` had the lowest mean RMSE and remains an explicit comparison. See [the complete decision and negative results](docs/COMPACT_BASE_20261026.md). This freezes the starting point for further research; novelty, the full formal benchmark and reserved external confirmation remain open.

The first post-freeze source-conditional shrinkage experiment is complete. Both registered primary variants failed their continuation criteria; the [complete negative-result summary](docs/SOURCE_RISK_RESULT_20261026.md) preserves all controls and explains the decision.

The package keeps reference-only preprocessing, grouped Ridge selection, named 31-channel outputs, artifact identities and separate scoring. Both ordinary residual heads can also be trained with nested reference selection or exported from independently verified research checkpoints.

The validated B0 foundation is preserved at Git tag `baseline-b0-20261025`. The current branch extends that version; completed evidence and remaining requirements are tracked in [REFINEMENT_LOOP.md](docs/REFINEMENT_LOOP.md). Historical code, negative results and checkpoints remain in the original directories and Git history.

```text
stprotein2/
  data.py          # Reference-fitted genes/scaling and separate protein reader
  graph.py         # Canonical physical k6/radius2.5 graph
  anchor.py        # Reference source CV and Ridge
  smoothing.py     # Fixed mean-preserving spatial correction
  neural.py        # Optional compact own-RNA / neighborhood residuals
  portable.py      # Numeric preprocessing and frozen coefficients
  deployment.py    # Export independently verified development checkpoints
  training.py      # Optional C0 Student fitting and exact checkpoint resume
  selection.py     # Source-held epoch/gate selection and refitting
  gates.py         # Conservative reference-OOF binary switches
  pipeline.py      # Train a bundle and predict RNA-only slices
  io.py            # Atomic artifacts and integrity
  provenance.py    # Source identities
  cli.py           # train / export / predict
evaluation/        # Separate measured-protein scoring
tests/             # Data isolation, end-to-end and numerical parity
```

The default prediction is:

`B0 = B_RNA + 0.5 * center(neighbor_mean(B_RNA) - B_RNA)`.

Ridge coefficients and transformed RNA are multiplied in FP64 and rounded to FP32, following the stable evaluation policy used by the current research experiments. The spatial correction uses the existing FP64 implementation. The target is row-centered `log1p` raw ADT over the same named 31 channels. No prototype, sheaf, Teacher, expert router or partition executor is deployed. The B0 path needs no PyTorch; residual inference uses the optional `neural` extra.

## Run

```bash
python -m pip install -e '.[test]'
python examples/make_demo.py --output demo/data

python -m stprotein2 train \
  --registry demo/data/registry.json --groups demo/data/groups.json \
  --query query --genes 8 --output demo/bundle

python -m stprotein2 predict \
  --bundle demo/bundle --rna demo/data/query_rna.h5ad \
  --sample query --output demo/prediction

python -m pytest -q
```

For actual data use `--genes 4000`. `--query` excludes its complete source from all training and preprocessing; omit it when fitting a final reference-only bundle from an already frozen reference registry. Prediction requires only a bundle and one RNA slice. Its output includes `prediction.h5ad`, components in `prediction.npz`, gene coverage and integrity records.

`train` defaults to B0. For compact residuals, install `.[neural]` and use `--model own_mlp` or `--model neighbor_mlp`, following [TRAINING.md](docs/TRAINING.md). [PORTABLE_DEPLOYMENT.md](docs/PORTABLE_DEPLOYMENT.md) covers standalone inference and checkpoint conversion. Neither export nor conversion selects a new model.

Measured query ADT is scored only after prediction:

```bash
python -m evaluation --registry /path/paired_registry.json \
  --prediction /path/prediction --output /path/separate_report --extended
```

The synthetic demo deliberately has no query ADT and cannot be scored without independent measured labels. New bundles/caches are required in this branch; legacy bundles require their matching implementation. See [the scientific contract](docs/SCIENTIFIC_CONTRACT.md) for scope and subsequent innovations.

GPL v3 is retained from STProtein/STProtein2. Core lineage and extraction details are in [extraction.json](docs/extraction.json).
