# Matched Teacher and distillation controls

This new protocol isolates Teacher behavior without editing frozen B2, C0/CF0 or competitor experiments. The independent `stprotein2_teacher_ablation` package reuses the unchanged spatial-anchor Student, prediction function, prototype initialization and residual-target builders. Every mode retains both learned Student codebooks in the B2 configuration.

| Mode | Teacher representation | Prototype loss in Student objective | Teacher forward / centers |
|---|---|---|---|
| `ema` | EMA of Student after each update | Yes | Yes |
| `no_loss` | Same EMA | No; computed only as a diagnostic | Yes |
| `none` | No Teacher is constructed | No | No |
| `fixed` | Copy initialized Student after warm-up; weights stay fixed | Yes | Yes; centers continue updating |

Fixed Teacher means fixed representation weights, including codebooks. It does not mean fixed centering buffers. Teacher temperature schedules, center momentum and Student losses otherwise remain unchanged. Teacher parameters never receive gradients or enter the optimizer. Without the distillation loss, Student codebooks remain trainable through the supervised residual path. Initializing Student codebooks still occurs in `none` mode.

`ema` must reproduce the unchanged B2 trainer's weights, predictions, selected checkpoint and numerical losses on a matched deterministic CPU test. `no_loss` and `none` must produce exactly the same Student update sequence under identical seeds: Teacher evaluation consumes no stochastic dropout or sampling. They differ in computation, memory and available diagnostics. These two modes should not be counted as independent predictive candidates if this equality holds. Runtime on the shared node is descriptive only.

Training remains source-balanced, with one uniformly sampled reference slice per group per epoch, a frozen independently fitted RNA anchor and fixed spatial alpha 0.5. C0 uses in-fit residual targets; CF0 uses independently source-excluded anchor targets at each nesting level. Preprocessing, router, anchor regularization, codebook initialization and checkpoint selection all stay inside allowed references.

Every outer query includes its entire conservative source group. Query RNA and allowed reference RNA/ADT file identities are frozen in the prediction manifest; query ADT is neither opened nor hashed. Query scores are produced only by the separate `scripts/score_extended_baseline.py` interface, which freezes query ADT identity afterward. The prediction completion marker binds the nested fit artifacts and model as well as named predictions. A source/configuration/input change requires a new run directory.

Initial real diagnostic, fixed before training:

- References: Tonsil_AddOns, Tonsil, Breast; inner validation: already-seen Glioblastoma.
- Input: the matching reference-fitted 4000-gene transform and exact 31-channel centered-log target; same frozen bases and output axes as the previous B2 diagnostic.
- Eight fits: C0 and CF0 × four modes, default B2 Student architecture, seed 2025, CPU with two threads.
- 16 epochs, warm-up 8, ramp 8, earliest eligible residual checkpoint 16, identical optimizer/hyperparameters and no input augmentation. This produces 48 Student updates; Teacher modes perform 24 post-warm-up forwards/center updates. EMA modes also perform 24 parameter updates; fixed mode performs zero parameter updates.
- Save RNA-only predictions before final label scoring. Inner labels legitimately select/check the checkpoint; this is not an outer holdout or independent confirmation.
- Report matched MSE, per-protein and slice-mean/within-slice error, prototype use, Student/Teacher parameter counts, forward/EMA/center counters, actual optimizer steps and artifact identities.
- Reproduce EMA against the prior B2 real diagnostic, and compare no-loss with no-Teacher output/weight equality. Do not infer isolated Teacher value from unmatched MLP results.

This short diagnostic does not establish converged performance. After checking correctness and failure modes, use the same finite candidates in longer reference-only selection and complete outer-source evaluation. Three-seed shortlisting, independent paired protein confirmation and biology remain required. No external liver/dental data enters this development runner.
