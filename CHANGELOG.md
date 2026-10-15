# Change log

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
