# Compact reference training

The optional trainer fits the ordinary own-RNA or neighborhood residual above the independent B0. It implements the confirmed C0 setting: no count/mask augmentation, no Teacher or prototypes, no learned graph operator, and reference-only nested epoch and gate selection. The completed three-seed comparison retained B0 as the compact development default; `own_mlp` remains the lowest-mean comparison. See [the frozen decision](COMPACT_BASE_20261026.md).

```bash
python -m pip install -e '.[neural]'
python -m stprotein2 train \
  --model own_mlp --config configs/compact_residual.json \
  --registry /path/to/reference_registry.json --groups /path/to/groups.json \
  --query development_query --genes 4000 --threads 2 --device cuda \
  --output /path/to/fresh_training_run

python -m stprotein2 export \
  --run /path/to/fresh_training_run --output /path/to/portable_bundle
python -m stprotein2 predict \
  --bundle /path/to/portable_bundle --rna /path/to/query_RNA.h5ad \
  --sample query --output /path/to/fresh_prediction --threads 2
```

Use `--device cpu` for CPU fitting. The `--model` choice can be changed to `neighbor_mlp` with the same configuration file. The CLI fixes the model head, gene count and thread count; model and gene values supplied in JSON must agree with the CLI. Other model/training fields use the declared dataclass defaults or the JSON values. Omit `--query` only when fitting all samples in an already frozen reference registry. At least three allowed reference sources are required. The separate scoring command remains `python -m evaluation`.

`--query` excludes its whole source from preprocessing, coefficient fitting, epoch selection, gate selection and input-file hashing. Its RNA and ADT files need not exist during training. Names and source labels are read from registry metadata. Within each allowed subset, Ridge selection refits genes and normalization inside every source CV fold. Residual training uses the resulting frozen coefficients and true reference protein targets.

One uniformly selected slice from each source is used per epoch, giving sources equal expected weight. The loss is the residual prediction MSE plus `0.2` times the per-member auxiliary protein MSE. The auxiliary head supervises true protein targets and is removed from the deployed network. A temporary initialization draw preserves the registered ordinary model's RNG sequence; there is no learned protein-embedding parameter.

The registered default has 80 maximum epochs, 8 warmup epochs, checkpoint eligibility from epoch 16, validation every 4 epochs, patience 6, AdamW learning rate 0.001, epsilon 1e-8, weight decay 1e-4 and gradient clipping at 1. Corrections start training after warmup. Inner validation chooses each source-held fit's epoch, then the integer floor of their median determines the refit budget.

Gate OOF predictions hold out an entire reference source from both fitting and epoch selection. Held labels are opened after the prediction file is saved. The unchanged rule requires four valid sources, at least 1% mean relative protein MSE gain, and improvement in at least 60% of sources. Ties prefer fewer correction components. With fewer than four allowed reference sources, gates stay closed. With five allowed sources, one outer development case contains 31 actual residual fits.

Training targets use the original FP32 CPU anchor followed by the FP64 NumPy spatial basis. Validation and gate prediction use the original FP32 device anchor and FP64 Torch spatial basis. Deployed inference retains the registered stable FP64 anchor accumulation and FP32 output. Keeping these roles explicit avoids changing the optimization experiment while stabilizing deployment.

Repeat the same training command to resume. Inputs, package sources, model/training settings, CPU/CUDA choice, hardware properties and matrix-precision flags are frozen. A changed identity requires a fresh output directory. Resume restores weights, AdamW state, sampling RNG, Torch dropout RNG and checkpoint selection state. Completed runs verify all nested artifacts before returning. Export strips training caches and auxiliary checkpoints; prediction needs only the portable bundle and query RNA.

Software validation includes an actual six-source CLI run with all 31 nested fits, query-file exclusion, exact interrupted/continuous training parity, completed-artifact rejection, export after deleting training data, RNA-only inference and independent scoring after deleting the model. The independent compatibility command additionally compares complete matched fits with the frozen research implementation:

```bash
python scripts/validate_clean_training.py \
  --research-root /path/to/STProtein/STProtein2 \
  --output /path/to/fresh_compatibility_check
```

Its small synthetic budget tests software equivalence; it does not establish biological accuracy, seed robustness, final architecture selection or novelty. The full-budget confirmation remains a separate registered experiment.
