# Unified-package validation

The source-layout release is `0.2.0.dev0`; the algorithm configuration remains v2.1. All supported Python source now uses the `stprotein2` namespace.

## Completed checks

| Check | Result |
|---|---|
| Namespace, public commands and source-manifest coverage | Passed |
| Selected deployment/model/data/gate/spatial regression | **31 passed, 1 CUDA case skipped** |
| Complete local regression suite | **98 passed, 7 skipped**, one intentional duplicate-barcode warning; 991.67 seconds |
| Original versus relocated numerical operator/trainer ASTs | Identical after excluding imports for the six checked source files |
| Fixed nonzero-residual prediction comparison | Exact equality for prediction, spatial base, RNA base and all three correction components |
| Fixed-input parameter-gradient comparison | Exact equality for 72 gradient tensors |
| Original versus reorganized complete synthetic training | Exact equality for 72 checkpoint tensors and all six compared prediction components |
| Wheel namespace inventory | Only `stprotein2`; no former sibling packages |
| Repository versus installed-wheel source manifests | Exact equality |
| Public commands from outside the source checkout | Passed |
| Actual installed-wheel nested training, export and prediction | Passed |
| Query ADT absent during the installed example | Verified |
| Original run versus exported model prediction/components | Exact equality; 16 × 31 finite predictions |

The selected regression command was:

```bash
python -m pytest -q tests/test_package_layout.py tests/test_deploy.py \
  tests/test_architecture.py tests/test_data_training.py \
  tests/test_nested_gates.py tests/test_dpas_evaluation.py
```

It completed in 30.77 seconds in the CPU validation environment. The single AnnData warning is produced intentionally by the invalid duplicate-barcode input test. Numerical parity uses a representative fixed-input case with nonzero corrections; it is not an accuracy benchmark or a universal numerical proof.

An additional comparison against the first release's independently completed synthetic training run found exactly equal model checkpoint tensors and prediction components.

The installed-wheel example performed three reference-held-out inner fits and a final refit with the reduced demo configuration. Three reference groups keep the validation switches closed. Deployment regression separately checks selected nonzero corrections, deleted training/ADT files, full/adaptive agreement, missing genes and corrupted artifacts.

The complete local regression suite, including historical-control training, nested selection, serialization and recovery checks, completed with 98 passed and 7 skipped in 991.67 seconds. Skips concern CUDA and optional native partition runtimes. The same suite is configured in GitHub Actions; its remote run is a separate result.

## Integrity and compatibility

Imports, package paths, configuration locations and source snapshots were updated together. The current training/deployment modules do not import historical controls from `research/`. Model tensor names and computation are preserved. The package uses canonical installation-relative source paths in editable and wheel installations, with checksum checks retained for caches, runs and exported bundles.

Earlier 0.1 bundles remain tied to their original environment/source hashes. New training and export should use a fresh output/cache location for this layout release.

Machine-readable checks are in [layout_validation.json](layout_validation.json), and file relocation records are in [module_migration.json](module_migration.json). The initial release's validation evidence is preserved under `history/initial_release/`.
