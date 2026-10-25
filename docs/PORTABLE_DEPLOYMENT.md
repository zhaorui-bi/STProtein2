# Standalone development-model export

The optional residual runtime supports the registered `own_mlp` and `neighbor_mlp` models. It retains their trained RNA ensemble, residual readout, QC gate and reference-selected binary switches. B0 stays external to the residual. Auxiliary training heads, prototypes, sheaves, Teachers and historical routers are absent from the exported network.

Install the optional runtime, then export an already independently verified **stable evaluation** folder:

```bash
python -m pip install -e '.[neural]'
python -m stprotein2 export \
  --run /path/to/stable/own_mlp_eps1e-08/seed_2025/Breast \
  --verification /path/to/independent_verification/0 \
  --output /path/to/fresh_bundle

python -m stprotein2 predict \
  --bundle /path/to/fresh_bundle --rna /path/to/query_RNA.h5ad \
  --sample query --output /path/to/fresh_prediction --threads 2
```

The export command binds the passed verification report to the source manifest, weights, preprocessing and gate artifacts. It reads those frozen artifacts only. It opens no reference or query expression data. Export accepts complete source holdouts and the fixed 31-channel, k6/radius2.5, alpha0.5 contract. It refuses changes to the verified artifacts and unsupported model tensors or normalization arithmetic.

The portable bundle has six files: its manifest and completion marker, numeric anchor arrays, numeric transform metadata, compact residual weights and conversion provenance. Normalization metadata records whether the source scaler casts statistics to input precision before each operation. This preserves its actual rounding without importing historical classes or depending on a future scikit-learn transform implementation. Bundle loading verifies the artifacts and matching package sources.

Prediction runs on CPU and accepts the whole query RNA slice with spatial coordinates. It handles unavailable genes through the fitted gene panel and availability mask. It returns named rows and proteins, B0, raw and centered residual components, binary gates and the final prediction. It requires no historical training files. A clean B0 training run can also be stripped with `export --run ... --output ...`, omitting `--verification`.

The reproducible real-checkpoint validation command is:

```bash
python scripts/validate_portable_deployment.py \
  --research-root /path/to/STProtein/STProtein2 \
  --output /path/to/fresh_validation
```

This builds a wheel, extracts it outside the checkouts, copies only the bundle and Breast RNA, and blocks historical package imports and research-file access. It compares both active residual heads against the independently verified 4,164 × 31 stored predictions using the existing `atol=1e-5, rtol=2e-5` limits. The synthetic tests separately delete the source runs, verification files and reference measurements before prediction; they also cover unavailable genes and damaged artifacts. These checks establish deployment compatibility. Clean neural training, final architecture selection, formal benchmark coverage and novelty evidence have separate completion requirements.
