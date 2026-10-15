# Balanced initialization integration, version 1

This is a new candidate namespace. Previously frozen training code and outcomes
are unchanged. The predictive Student, fixed spatial anchor, residual losses,
Teacher modes, source weighting and nested selection are inherited from
`stprotein2_teacher_ablation_v2`.

The only experimental factor for the first matched comparison is `initialization`:
`pooled` invokes the original source-pooled router/prototype initialization;
`hierarchical` invokes the audited source/slice/unique-spot initializer. The same
reference-only PCA/spherical router dimensions, codebook sizes, seed, model,
training budget, Teacher mode and execution policy apply to both arms. Router
and prototype sample limits are 4000 and 2000 per source. The old pooled
prototype initializer has a fixed 2000 cap and must not silently use another cap.

The hierarchical router and state/niche samples contain exactly the current
training names. State initializes after warm-up, niche is recomputed using the
initialized state, then Teacher copies the complete Student representation.
Validation/query RNA is never fitted into either initializer. Balanced sampling
changes initialization, not the already source/slice-balanced training objective.

Execution flags separately register weight-side modulation and representation
checkpointing. Configure the Student before optimizer/Teacher construction and
on artifact loading. Preserve parameters, buffers and state keys. Initializer
records and initialized-stage state survive interruptions; changed policies
cannot resume in the same output directory. Exported model records must carry
the execution policy; loading may not silently fall back to the old kernel.

The existing full/adaptive mode is inherited. This integration does not add
the independent METIS/AH2 hard-budget planner to the training CLI. Full mode is
used for the initial comparison to isolate initialization from partition owners.

Required checks: both policies, positive nonzero residual training, interruption
after initialization, exact CPU resumed weights/Teacher state, completed reuse,
policy mutation rejection, RNA-only restored prediction, all 31 fits in one
six-source synthetic outer fold, and exclusion of every gate-held source also
from epoch selection and both initializers. Real diagnostics are registered in
a separate finite protocol before training. They are development evidence, never
independent outer accuracy or proof of SOTA.
