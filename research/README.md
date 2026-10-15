# Research experiments

This directory retains historical model controls and experimental runners. It is excluded from the core wheel and is used from a source checkout after `python -m pip install -e '.[test]'`.

Use `stprotein2 train` with `configs/latest.json` for the supported model. The following modules preserve distinct experimental protocols and are not interchangeable training recipes:

| Directory/module | Purpose |
|---|---|
| `joint/` | Original jointly fitted neural base and residual model |
| `anchor/` | Independent frozen Ridge anchor control |
| `residuals/` | Matched in-fit/cross-fit residual-target experiments |
| `spatial/` | Spatial-basis residual controls |
| `teacher/`, `teacher_recovery/` | Teacher-loss and cache/recovery controls |
| `strict_nested.py`, `strict_cli.py` | Earlier strict balanced initialization/training protocol |
| `cohort_v1/` | Earlier cohort-level runner |
| `smoothing/` | Classical spatial shrinkage, reference-selected coefficient and permuted graphs |
| `provenance.py` | Experiment source identity and immutable snapshots |

Inspect a runner's arguments from the repository root:

```bash
python -m research.joint --help
python -m research.anchor --help
python -m research.residuals --help
python -m research.spatial --help
python -m research.teacher --help
python -m research.teacher_recovery --help
python -m research.strict_cli --help
python -m research.cohort_v1 --help
```

The smoothing experiment exposes `research.smoothing.experiment.run_query` as a Python API. Its protocol requires completed independent anchor caches. Review the [dated protocols](../docs/history/protocols/) and the relevant regression tests before running controls.

Research run manifests record the core source identity plus a separate identity covering research code, evaluation tools and dated protocol records. Editing research does not invalidate core training caches or deployed core bundles. Exporting a research run verifies its recorded research dependencies; the resulting inference bundle contains only core runtime sources. Use fresh outputs/caches after a source-layout change, preserving matching environments for historical runs.
