# Code organization

All distributed Python source belongs to `stprotein2`. The supported pipeline is:

```text
cli.py
  → training/nested.py
      → training/anchors.py + data.py
      → training/initialization.py
      → training/trainer.py → models/spatial.py
      → gates.py
  → inference/bundle.py
  → evaluation/score.py
```

## Models

| File | Responsibility |
|---|---|
| `models/spatial.py` | Complete frozen-spatial-anchor STProtein2 model |
| `models/anchor.py` | Frozen Ridge layer and independently anchored residual model |
| `models/residual.py` | State/local residual heads and protein-conditioned expert mixture |
| `models/prototypes.py` | RNA ensemble and state/niche prototype memories |
| `models/sheaf.py` | Cooperative sheaf propagation and graph-versus-self correction |
| `models/smoothing.py` | Centered spatial basis and classical spatial-control operators |

The public `from stprotein2 import STProtein2, ModelConfig` exposes the complete spatial model and its model configuration. Fitted inference uses `stprotein2 predict` or `stprotein2.predict_file`.

## Training

`training/nested.py` implements the current cohort pipeline, including fold-specific refits and reference-only selection. `trainer.py` trains the current model. `initialization.py` initializes routing and memories. `anchors.py`, `residual_targets.py` and `targets.py` construct the independent anchor and training offsets. `config.py` contains the training configuration hierarchy in one location. `prediction.py` runs the complete spatial model; `basic.py` supplies shared tensor, augmentation and initialization helpers.

## Deployment, evaluation and execution

`inference/bundle.py` handles completed-run export, checkpoint loading and RNA-only application. `evaluation/metrics.py` contains core error/correlation metrics; `evaluation/spatial.py` adds spatial and domain metrics. `evaluation/score.py` is the separate measured-ADT scoring interface. `execution/` holds runtime dispatch, exact-dependency partitions and refinement.

`io.py`, `paths.py`, `serialization.py` and `provenance.py` provide shared artifact handling, installation roots, content identities and source snapshots. Training and deployment use these common helpers rather than importing benchmark orchestration.

## Historical controls

`research/` retains the joint, anchor-only, residual-target, teacher and earlier cohort controls used by regression tests. The supported model/training/inference/evaluation code does not import `research`. External competitor adapters from the original multi-project workspace have been removed from this distribution; they were not standalone entry points.

The former top-level packages were consolidated as follows:

| Earlier directory | Current location |
|---|---|
| `stprotein2_anchor` | `models/anchor.py`, `training/anchors.py` |
| `stprotein2_spatial_anchor` | `models/spatial.py`, `training/targets.py`, `training/prediction.py` |
| `stprotein2_balanced_training` | `training/trainer.py`, `training/config.py` |
| `stprotein2_balanced_initialization` | `training/initialization.py` |
| `stprotein2_latest_benchmark_v2` | `training/nested.py`, `training/__main__.py` |
| `stprotein2_deploy` | `inference/` |
| `stprotein2_dpas_evaluation` | `evaluation/spatial.py` |
| `stprotein2_execution`, `stprotein2_partition`, `stprotein2_partition_cost` | `execution/` |
| Historical teacher/cohort training variants | `research/` |
| `stprotein2_resources` | `resources/` |

The source-layout version is `0.2.0.dev0`; the algorithm configuration remains v2.1. Earlier 0.1 exports record their original module paths and source hashes, and should be read with their matching 0.1 environment. New training and export use the unified layout; integrity checks are not bypassed to load an incompatible bundle.
