# Troubleshooting

Start with `stprotein2 --version` and the matching [API guide](API.md). Run `python -m evaluation` from the source checkout; evaluation/research are repository tools excluded from the core wheel.

| Message or behavior | Cause and action |
|---|---|
| Invalid integer, finite value or unknown configuration key | Check the named field against `configs/latest.json`. Counts/dimensions require integers; optimizer parameters, temperatures and rates require finite valid values. |
| Registry sample/group validation error | Supply unique sample names, RNA/ADT paths and a group ID for every sample. Relative paths are based on the registry file directory. Query ADT may remain absent until evaluation. |
| Raw nonnegative integer counts required | Supply raw RNA/ADT counts in `.X`; normalized/log expression is not accepted as raw input. |
| Ambiguous RNA/protein identity or panel/barcode mismatch | Use unique genes/barcodes and the fixed named 31-protein ADT panel. Match query barcodes rather than relying on file row order. |
| Insufficient common reference genes | Check gene identifiers and common coverage, or explicitly choose a smaller `n_genes` before starting a new experiment. |
| No measured input genes overlap the frozen model vocabulary | Check that query and reference gene identifiers use the same convention. Partial overlap is reported by prediction diagnostics. |
| Requested CUDA unavailable | Use a compatible CUDA PyTorch installation and visible GPU, or explicitly choose `--device cpu`. |
| CUDA allocation failure | The allocator limit caps use; it cannot make a dataset fit. Choose execution/model settings explicitly before a fresh run. |
| Prediction equals the spatial base | Check `gate_selection.json`. Fewer than four reference groups keep validation switches closed; more groups still require the prespecified gain/consistency rule. |
| Source identity changed or incompatible frozen anchor | Use the matching version/environment for existing artifacts, or start with fresh cache/output directories. |
| Artifact missing/changed or checksum mismatch | Restore the original artifact or rerun into a fresh destination. Completed artifacts are verified before reuse. |
| Evaluation RNA/sample/axes differ | Use the original prediction RNA file and a registry entry matching `predict --sample`; align measured ADT by barcode and named proteins. |
| Invalid domain cluster count | Choose distinct prespecified counts at least two and smaller than the number of query spots, or omit `--extended` for standard metrics. |
| Correlation/spatial metric is `null` | A constant channel/spot or unsupported spatial statistic is undefined. Inspect coverage diagnostics; complete-panel metrics preserve undefined cases. |

For reproducible issue reports, include the software version, command, configuration, traceback and input dimensions/identifier conventions. Keep private data and model files out of public issues.
