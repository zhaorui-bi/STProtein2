# Core-package boundary validation

The source-layout version is `0.3.0.dev0`; the algorithm configuration remains v2.1. Evaluation and research code live outside `stprotein2/` and are excluded from the core wheel.

## Completed local checks

| Check | Result |
|---|---|
| Focused boundary, deployment, architecture, data, gate and metric regression | 33 passed, 1 skipped; 246.82 seconds |
| Complete regression suite on a local copy of the current runtime sources | 100 passed, 7 skipped; 212.83 seconds |
| Additional research-export isolation regression | 1 passed; 7.36 seconds |
| Core dependency boundary | No imports of repository evaluation/research tools |
| Built wheel | Only `stprotein2`; 38 Python source files, no evaluation/research/protocol resources |
| Source archive | Includes evaluation, research, protocols, configs, examples and development tools |
| Repository versus installed-wheel core source identity | Exact equality |
| Core public API and command imports with repository tool imports blocked | Passed |
| Actual wheel training, export and RNA-only prediction outside the checkout | Passed; 16 × 31 finite predictions |
| Trained run versus exported model | All six compared prediction components exactly equal |
| Previous versus current complete synthetic training | All 72 checkpoint tensors and six compared prediction components exactly equal |
| Fixed-input nonzero-residual model parity | Six prediction components and 72 parameter gradients exactly equal |
| Core model/operator/trainer AST comparison with 0.2 | Six checked files unchanged after excluding imports |
| Historical joint fit, coefficient search and graph permutation functions | Function ASTs unchanged after relocation |
| Research identity and portable bundle boundary | Altered research code rejected at export; exported model loads with core sources only |

The complete suite and additional regression cover **101 passing tests in total**, with seven optional-runtime skips. The single AnnData warning is intentionally triggered by an invalid duplicate-barcode input. CUDA and optional native partition runtimes were unavailable for these checks.

The complete suite used `python -m pytest -q` on a local filesystem copy to avoid shared-filesystem overhead. Its core, research and evaluation implementations match the checkout (complete Python AST equality verified; a trailing blank line was removed during final diff cleanup). The additional research-export test was added afterward and run separately:

```bash
python -m pytest -q tests/test_deploy.py::test_research_export_checks_experiment_sources_but_bundle_needs_only_core
```

The distribution check is reproducible after installing a built wheel:

```bash
python scripts/validate_wheel.py --work-dir /tmp/stprotein2-wheel-check
```

Use a fresh destination. The script runs outside the source checkout, verifies the installed source identity and dependency boundary, then performs real nested demo training, export and inference with query ADT absent. The demo uses three reference groups, which keep selection switches closed. Separate deployment and numerical parity checks exercise nonzero selected corrections. These engineering checks do not establish comparative prediction accuracy.

## Integrity and compatibility

Core artifacts bind only core source hashes. Research manifests separately bind experiment/evaluation code and dated protocols; scoring manifests bind evaluation code. Research dependencies are verified when exporting those runs, while a deployed bundle contains only core runtime sources.

Use fresh cache/output directories for this version. Existing 0.1/0.2 artifacts remain tied to their matching environments and source hashes; integrity checks have not been bypassed. Previous 0.2 validation evidence is retained in [history/layout_0_2/](history/layout_0_2/), and initial extraction evidence remains in [history/initial_release/](history/initial_release/).

The machine-readable record is [layout_validation.json](layout_validation.json). GitHub Actions repeats the complete suite and tests an installed wheel outside the checkout; remote CI status is separate from these local results.
