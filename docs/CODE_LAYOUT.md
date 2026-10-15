# Code organization

The repository has three explicit boundaries. Only the core package is installed into the wheel.

| Location | Responsibility | Distribution |
|---|---|---|
| `stprotein2/` | Reusable model, preprocessing, reference-only training, export and RNA-only inference | Core wheel and source archive |
| `evaluation/` | Measured-ADT scoring and comparative metrics | Source archive / checkout |
| `research/` | Baselines, ablations and historical experimental runners | Source archive / checkout |
| `docs/history/` | Dated protocols and earlier validation evidence | Source archive / checkout |

Dependencies point from evaluation/research to core. Core software does not import repository evaluation or research tools.

## Core pipeline

```text
stprotein2/cli.py
  → training/nested.py
      → training/anchors.py + data.py
      → training/initialization.py
      → training/trainer.py → models/spatial.py
      → gates.py
  → inference/bundle.py

evaluation/__main__.py → evaluation/score.py → completed predictions + measured ADT
research/             → core components and separate experimental protocols
```

| Core file/directory | Responsibility |
|---|---|
| `models/spatial.py` | Complete frozen-spatial-anchor model exposed as `STProtein2` |
| `models/anchor.py`, `models/residual.py` | Frozen Ridge layer, state/local heads and protein-conditioned expert mixture |
| `models/prototypes.py`, `models/sheaf.py` | RNA ensemble, prototype memories and cooperative sheaf propagation |
| `models/smoothing.py` | Mean-preserving spatial basis shared by training/inference |
| `training/nested.py`, `training/trainer.py` | Current grouped pipeline and balanced residual trainer |
| `training/initialization.py` | Reference-only hierarchical routing and memory initialization |
| `training/anchors.py`, `training/residual_targets.py`, `training/targets.py` | Independent anchor selection/caching and residual offsets |
| `training/config.py`, `training/basic.py`, `training/prediction.py` | Training configuration, shared helpers and prediction |
| `inference/` | Completed-run export, bundle loading and whole-slice RNA-only prediction |
| `execution/` | Runtime dispatch, exact-dependency partitions and refinement |
| `data.py`, `graph.py`, `gates.py` | Data contracts, spatial graphs, frozen routing and reference-selected switches |
| `io.py`, `paths.py`, `serialization.py`, `provenance.py` | Artifact handling, installation roots, content identities and core source snapshots |

Training configuration inheritance and inference loaders retain reusable support for model families exercised by the research controls. Their experimental fitting/selection orchestration stays in `research/`.

## Provenance

Core source hashes cover only `stprotein2/**/*.py` and match between a checkout and its built wheel. Repository edits to metrics, experiments or documentation do not change core cache/bundle identities. Research manifests additionally bind research/evaluation code and dated protocols; scoring records evaluation sources separately. Exported bundles snapshot only the runtime core.

The source-layout version is `0.3.0.dev0`; the algorithm configuration remains v2.1. Existing 0.1/0.2 artifacts remain tied to their matching source hashes/environments. New training and export require fresh output/cache locations. See [CHANGELOG.md](../CHANGELOG.md) for interface migration and [the validation record](LAYOUT_VALIDATION.md) for current checks.
