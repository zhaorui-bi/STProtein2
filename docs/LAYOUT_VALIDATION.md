# Software validation

The software version is `0.3.0.dev1`; the algorithm configuration remains v2.1. Core software, offline evaluation and research remain separate.

## Completed local checks

| Check | Result |
|---|---|
| Complete sequential regression suite | **129 passed, 7 skipped**, one intentional duplicate-barcode warning; 1342.13 seconds |
| Complete regression suite with four test workers | **129 passed, 7 skipped**, the same intentional warning; 511.46 seconds |
| Final checkout versus tested core/research/evaluation/test sources | Byte-identical |
| Standard and extended evaluation of actual exported predictions | Passed after deleting the fitted bundle and all training references |
| Completed report reuse and changed/corrupt inputs | Verified; incompatible inputs/settings and modified reports rejected |
| Existing training-run evaluation and separate report directories | Passed, including spatial/domain metrics and legacy entry point |
| Complete-panel and spot correlations | Match independent formulas; undefined coverage preserved |
| Invalid configurations/registries | Rejected before data access; relative paths independent of working directory |
| Lightweight version command | Works with scientific imports blocked |
| Wheel and source archive metadata | `twine check --strict`: passed for both distributions |
| Core wheel boundary and installed version | Only `stprotein2`; 38 Python files; installed metadata matches `--version` |
| Checkout versus installed-wheel source identity | Exact equality |
| Installed-wheel nested training, export and RNA-only prediction outside checkout | Passed; 16 × 31 finite predictions; query ADT absent |
| Trained run versus exported model | All six compared prediction components exactly equal |
| Previous versus current complete synthetic training | All 72 checkpoint tensors and six prediction components exactly equal |
| Fixed-input nonzero-residual parity | Six components and 72 parameter gradients exactly equal |
| Model modules and current trainer versus previous commit | Complete Python ASTs unchanged |

Both full-suite runs used a local filesystem copy of the final runtime/test sources. They ran with CUDA hidden and two OpenMP/OpenBLAS threads per process. Optional CUDA and native partition dependencies were unavailable. The timing reflects this validation environment and concurrent workloads; it is not a model performance benchmark.

Reproduce the standard suite with:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Parallel execution additionally needs `pytest-xdist` and uses `python -m pytest -n 4 -q`. Regression covers real training, nested group exclusion, serialization, frozen-source checks, recovery, evaluation and deployment. Engineering parity checks do not establish accuracy superiority over another method.

## Distribution checks

After building and installing a wheel, run from the checkout:

```bash
python scripts/validate_wheel.py --work-dir /tmp/stprotein2-wheel-check
```

Use a fresh directory outside the checkout. The script verifies core-only imports, source identity and version metadata, then performs nested demo training, export and RNA-only prediction. Three reference groups keep demo selection switches closed; separate deployment/parity tests exercise nonzero corrections.

CI builds the wheel from the source archive, validates both distributions, checks installed dependencies, runs the full suite and exercises the installed wheel outside the checkout on Python 3.10 and 3.12. Remote CI outcomes remain separate from these local results.

## Integrity and compatibility

Core artifacts bind only core source hashes. Research manifests separately bind experiment/evaluation code and dated protocols. Evaluation reports bind input, prediction and metric-setting identities and reject modified completed artifacts. The core wheel contains no evaluation/research code.

Core source hashes change in this version. Use fresh caches and output directories for new training; preserve matching environments for existing artifacts. Model equations and the v2.1 configuration are unchanged. Prior evidence remains under [history/](history/README.md), including [0.3.0.dev0](history/layout_0_3_0/) and [0.2](history/layout_0_2/).

The machine-readable record is [layout_validation.json](layout_validation.json).
