# Reduced core and the clean research foundation

2026-10-26: the reduced core and compact three-seed confirmation are complete. The user authorized restarting the local A40, reducing unnecessary ablations, then simplifying the implementation before adding new mechanisms.

The completed core report is in `/fs04/scratch2/nd32/zhaorui/STProtein/STProtein2/runs/stprotein2_core_20261025_v1/README.md`: 198 structure, 36 graph and 72 target/lr cases passed, including the 16 previously missing target/lr cases. The original 339 deferred cases remain deferred. All six sources, 31 channels, original training/configuration and independent replay/scoring were retained.

The compact implementation was built and validated in the isolated worktree `/fs04/scratch2/nd32/zhaorui/STProtein2-clean` on branch `compact-refinement-20261025`. It retains preprocessing, grouped Ridge, the fixed spatial operator, inference, separate scoring, source identities and two optional compact residual controls. The B0 path requires no PyTorch. Its validated 11-file foundation is preserved at `baseline-b0-20261025`; the complete current development base is preserved at `compact-base-20261026`.

The complete six-source, three-seed comparison froze B0 under the predeclared parsimony rule; own_mlp had the lowest mean RMSE and remains the explicit comparison. See [the frozen decision](COMPACT_BASE_20261026.md) for the distinction, all negative sources and software evidence. This freezes a development starting point, not the final architecture including novelty or a SOTA claim.

New mechanisms are introduced one at a time against fixed B0, own-RNA and ordinary-neighborhood controls. The detailed research scope is recorded in `../STProtein/STProtein2/docs/STPROTEIN2_CORE_REDUCTION_20261025_v1.md`; [the refinement checklist](REFINEMENT_LOOP.md) retains formal benchmark, novelty and reserved external requirements.
