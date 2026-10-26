# First post-freeze experiment: source-conditional residual shrinkage

The registered primary mechanism failed for both residual heads. It is retained as a negative development result and is not added to the compact default. The existing B0 development freeze remains in force.

All 36 outer tasks completed: two frozen residual heads, six complete sources, three seeds, ten query slices and the fixed 31-protein panel. Four prespecified shrinkage controls and the B0/original binary controls produced 360 named point matrices. All predictions froze before a separate model-free process produced 216 complete score reports. Reference-only calibration did not use outer-query ADT; reserved external expression was not accessed.

Source-macro RMSE, lower is better:

| Control | Own RNA head | Neighborhood head |
|---|---:|---:|
| B0 | 0.794469009 | 0.794469009 |
| Original binary gate | 0.793147030 | 0.794450080 |
| Global pooled shrinkage | 0.793124220 | 0.793891437 |
| Global source consensus | 0.794850910 | 0.794300345 |
| Conditional pooled shrinkage | 0.793115156 | 0.793812500 |
| **Registered primary: conditional source consensus** | **0.794865375** | **0.794359808** |

The own-RNA primary variant has worse mean RMSE than B0 and all required controls, improving only 2/6 sources against B0 and 3/6 against its original binary gate. The neighborhood primary variant improves 5/6 sources against B0 and 4/6 against its binary gate, but loses to both the global-consensus and conditional-pooled controls. Both factors change predictions and worst-source harm is reduced relative to the binary controls; those successes do not satisfy the complete registered rule.

The conditional-pooled own-RNA control has a slightly lower mean than the original own-RNA gate, by approximately 0.000032 RMSE. It was a prespecified control, and this observed difference does not establish the primary mechanism, originality or independent predictive gain. No threshold, source list, stopping rule or compact-default criterion was changed after scoring.

An additional independent aggregation checked every score report, all 360 full-panel matrices and source/seed means within 1e-12. B0 reproduces the frozen compact comparison exactly; the replayed binary mean differs from that comparison by less than 1e-10 for either head.

Evidence in the preserved research checkout:

- [Registered protocol](/fs04/scratch2/nd32/zhaorui/STProtein/STProtein2/runs/source_risk_conditional_development_20261026_v1/protocol.json)
- [Complete results, per-source/seed/protein metrics and support](/fs04/scratch2/nd32/zhaorui/STProtein/STProtein2/runs/source_risk_conditional_development_20261026_v1/report.json)
- [Completion and scientific decisions](/fs04/scratch2/nd32/zhaorui/STProtein/STProtein2/runs/source_risk_conditional_development_20261026_v1/complete.json)
- [Independent aggregation audit](/fs04/scratch2/nd32/zhaorui/STProtein/STProtein2/runs/source_risk_conditional_development_20261026_v1/independent_result_audit.json)
- [Software admission, real reference replay and isolated installed-wheel prediction](/fs04/scratch2/nd32/zhaorui/STProtein/STProtein2/docs/source_risk_pipeline_validation_20261026_v1.json)

The final report SHA256 is `62a2b726781cee00705e15339d41ddfdddd37b9300bcd1ae99b562fc05ef7a8b`. The 17 compact package source files remain unchanged. Novelty, the complete eleven-method formal benchmark and reserved external confirmation remain open. A separately registered sciPENN development campaign is running; its current state is recorded in the [campaign directory](/fs04/scratch2/nd32/zhaorui/STProtein/STProtein2/runs/scipenn_registered_development_20261026_v2/pipeline_status.json).
