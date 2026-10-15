# Standalone release validation

Validated on 2026-10-15 (Australia/Melbourne).

## Completed checks

| Check | Result |
|---|---|
| Build standalone wheel | Passed; protocol and scoring-source resources included |
| Deployment regression suite | **8 passed**, including the new balanced v2.1 checkpoint case |
| Installed-package imports outside both source checkouts | Passed |
| Actual reference-only nested training from the installed wheel | Passed; three inner fits followed by a final refit |
| Export latest no-teacher C0 model | Passed; restored `BalancedTrainConfig` and execution policy |
| Query RNA-only inference with query ADT absent | Passed; finite 16 × 31 predictions |
| Saved versus exported prediction and residual components | Exact array equality |
| Core model, residual trainer and nested-runner extraction | Byte-identical to original workspace snapshot |
| Staged artifact inventory | Code, configuration, documentation, tests and framework image |

The deployment suite runs nonzero state/local corrections, deletes the original reference/ADT files before inference, exercises full/adaptive execution, checks missing genes and named axes, and rejects modified artifacts. Its single warning comes from a deliberately duplicated-barcode input in an error-handling test.

The installed-wheel pipeline uses a reduced synthetic configuration to test packaging and inference. Three reference groups cannot activate the production validation switches, so the selected demo output falls back to its spatial base. The separate deployment regression includes nonzero selected corrections. These are engineering checks, not protein-accuracy results.

## Full regression status

The complete regression suite is included in `tests/` and configured in `.github/workflows/tests.yml`. Local full-suite attempts were not completed; no full-suite pass is claimed here. They include repeated nested refits across historical model variants. Release checks were scoped to the installation/export changes and the installed current training pipeline; the algorithm and trainer mathematics remain unchanged. The GitHub Actions run provides a separate full-suite result. CUDA and native partition cases require their corresponding runtimes.

## Runtime and extraction changes

The validation environment used Python 3.12.12, PyTorch 2.8.0+cu129 on CPU, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.0, AnnData 0.12.19 and joblib 1.6.0. The isolated wheel installation reused the verified scientific dependencies without processing the original editable-package hooks. Import checks confirmed that model code came from the installed standalone wheel.

The extraction preserves frozen-anchor, spatial-basis, prototype, sheaf, residual-training and gate-selection computation. Release adaptations restore the latest configuration and execution dispatch during deployment, package the provenance resources needed by installed training, and add the current training entry point. `configs/latest.json` permits new reference-only anchor fitting on a fresh clone; the original archived configuration remains available.

See [the source snapshot](source_snapshot.json) and [installed-wheel check record](installed_wheel_validation.json). The wheel SHA256 is `d17d2b6afb2a378fed0e6c6f300158a63dcce9ad66d9acc77f60da94f7d8f48b`.
