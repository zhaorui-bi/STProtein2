# Change log

## 0.4.0.dev0 — clean foundation candidate

- Extract the reference-only Ridge and fixed physical B0 into a small CPU package.
- Remove neural prototypes, sheaf, Teacher, conditional experts, partition executors and historical experiment runners from this candidate branch.
- Preserve the canonical preprocessing/target, grouped Ridge selection, stable anchor arithmetic, spatial operator, named outputs and separate evaluation.
- Keep legacy code and artifacts in the original checkouts. Candidate readiness is gated by the reduced core evidence and clean implementation validation.

This is a foundation for subsequent research, not a new accuracy improvement or the final STProtein2 architecture.

## 0.3.0.dev1

- Add evaluation of standalone prediction directories without a model/training dependency; preserve the training-run evaluation interface and allow separate report destinations.
- Expose optional spatial/domain metrics in both evaluation commands. Add complete-panel protein correlations, mean spot correlation, macro protein RMSE and explicit undefined coverage.
- Verify completed evaluation reports before reuse; reject incomplete paired comparisons and invalid metric directions/tolerances.
- Reject invalid numeric configurations before training and resolve relative registry paths against the registry directory.
- Add a lightweight `--version` command, one source for package version metadata, documentation navigation and troubleshooting.
- Validate distributions and dependencies and run CI on Python 3.10 and 3.12.

Model equations and the v2.1 algorithm configuration are unchanged. Core source checksums change; new training needs fresh cache/output directories, while existing artifacts need their matching environment.

## 0.3.0.dev0

- Move measured-ADT evaluation to repository-level `evaluation/` and historical experiments to `research/`.
- Keep the wheel limited to core model, data, training, execution and inference code. Move the historical joint trainer and spatial-control searches out of the core.
- Replace `stprotein2 score` with `python -m evaluation` from the source checkout. The four core commands are `prepare`, `train`, `export` and `predict`.
- Move historical protocol resources to `docs/history/protocols/`; separate research/evaluation provenance from core source identity.
- Add development, evaluation and research guides and validate wheel execution outside the checkout.

The algorithm configuration remains v2.1. This is a source-layout/interface change, not an accuracy improvement. Use fresh run/cache directories. Versions 0.1 and 0.2 artifacts retain their original source hashes and should be read in their matching environments.

## 0.2.0.dev0

Consolidated the first upload's sibling namespaces into `stprotein2`, introduced the unified public API and CLI, and verified numerical/deployment parity. Evidence is preserved in `docs/history/layout_0_2/`.

## 0.1.0.dev0

Initial standalone extraction. Evidence is preserved in `docs/history/initial_release/`.
