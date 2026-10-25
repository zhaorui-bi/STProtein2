# STProtein2 clean foundation

This branch is the small **B0 reference baseline for the next STProtein2 iteration**. It retains reference-only RNA preprocessing, grouped Ridge selection, the fixed physical spatial basis, named 31-channel outputs, artifact identities and separate scoring. Neural residuals remain development controls in the original research checkout. This foundation is not a claim that the final predictive architecture or new mechanisms have been established.

The reduced core experiments and the clean implementation checks have passed. This branch is the validated B0 foundation; compact neural-model confirmation is tracked separately before selecting the next default model. Check `docs/core_readiness.json` for the bound evidence. Historical code, negative results and checkpoints remain in the original directories and Git history.

```text
stprotein2/
  data.py          # Reference-fitted genes/scaling and separate protein reader
  graph.py         # Canonical physical k6/radius2.5 graph
  anchor.py        # Reference source CV and Ridge
  smoothing.py     # Fixed mean-preserving spatial correction
  pipeline.py      # Train a bundle and predict RNA-only slices
  io.py            # Atomic artifacts and integrity
  provenance.py    # Source identities
  cli.py           # train / predict
evaluation/        # Separate measured-protein scoring
tests/             # Data isolation, end-to-end and numerical parity
```

The default prediction is:

`B0 = B_RNA + 0.5 * center(neighbor_mean(B_RNA) - B_RNA)`.

Ridge coefficients and transformed RNA are multiplied in FP64 and rounded to FP32, following the stable evaluation policy used by the current research experiments. The spatial correction uses the existing FP64 implementation. The target is row-centered `log1p` raw ADT over the same named 31 channels. No prototype, sheaf, Teacher, expert router or partition executor is imported or installed in this branch. PyTorch is not a dependency.

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

Measured query ADT is scored only after prediction:

```bash
python -m evaluation --registry /path/paired_registry.json \
  --prediction /path/prediction --output /path/separate_report --extended
```

The synthetic demo deliberately has no query ADT and cannot be scored without independent measured labels. New bundles/caches are required in this branch; legacy bundles require their matching implementation. See [the scientific contract](docs/SCIENTIFIC_CONTRACT.md) for scope and subsequent innovations.

GPL v3 is retained from STProtein/STProtein2. Core lineage and extraction details are in [extraction.json](docs/extraction.json).
