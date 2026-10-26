# sciPENN common-input development result

The registered common adapter completed all 18 source/seed tasks, 48 actual fits, 30 query prediction matrices, fresh CPU bundle replays and separate model-free scores. All six sources, ten slices, three seeds and 31 named proteins are included. No budget extension was triggered.

| Method | Source-macro RMSE |
|---|---:|
| Frozen compact default B0 | 0.794469009 |
| Original own-RNA residual comparison | 0.793147029 |
| sciPENN common adapter | 0.819911546 |

Lower is better. B0 has approximately 3.10% lower aggregate RMSE than this sciPENN profile, and lower RMSE in 3/6 sources. This is a completed development comparison; the complete eleven-method benchmark and reserved external confirmation remain open.

| Source | B0 | sciPENN |
|---|---:|---:|
| 10x_addon_source | 0.484141148 | 0.556539731 |
| 10x_breast_source | 0.726356491 | 0.724591630 |
| 10x_glioblastoma_source | 0.948748092 | 1.015080075 |
| 10x_tonsil_source | 0.510191880 | 0.556250012 |
| GSE263617_unresolved_donors | 1.008975434 | 1.005162403 |
| GSE300851_mesothelioma_unresolved_donors | 1.088401010 | 1.061845426 |

The author network and loss were preserved under the preregistered common-input profile. Selection used complete reference-source folds at seed2025; other seeds reused the frozen epoch count and learning-rate prefix. Outer query-source ADT paths remained absent during training and prediction. All predictions and independent replays froze before scoring. The maximum GPU/CPU point-prediction replay error was 5.2453e-6, within the unchanged tolerance.

[Complete report](/fs04/scratch2/nd32/zhaorui/STProtein/STProtein2/runs/scipenn_registered_development_20261026_v2/report.json), [completion record](/fs04/scratch2/nd32/zhaorui/STProtein/STProtein2/runs/scipenn_registered_development_20261026_v2/complete.json), [independent aggregation audit](/fs04/scratch2/nd32/zhaorui/STProtein/STProtein2/runs/scipenn_registered_development_20261026_v2/independent_result_audit.json).

This comparison does not validate the rejected source-conditional novelty mechanism or establish a final architecture.
