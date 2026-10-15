# Development

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Optional partition-runtime checks need `.[test,partition]`; CUDA tests require a compatible PyTorch installation and GPU. Missing optional runtimes produce explicit skips. A focused boundary/deployment check is:

```bash
python -m pytest -q tests/test_package_layout.py tests/test_deploy.py tests/test_dpas_evaluation.py
```

Put reusable software in `stprotein2/`, measured-ADT metrics and comparison tools in `evaluation/`, and experimental variants/baselines in `research/`. Dependencies point from repository tools to the core. The core must not import either repository tool package. Keep model selection reference-only and evaluation separate from prediction.

Core algorithm changes should include relevant numerical, data-isolation or recovery checks. Update the API guide for public-interface changes. Keep large datasets, run directories and model checkpoints out of Git. Preserve historical validation records as evidence for their recorded version.

Build distributions with:

```bash
python -m pip install build
python -m build
```

The wheel includes only `stprotein2`. The source archive also includes evaluation, research, examples, configs and documentation. CI tests a built wheel from outside the checkout, including actual demo training, export and RNA-only prediction.

Runs, anchor caches and bundles bind source hashes. Use a fresh output/cache directory after changing source; preserve the matching environment for existing artifacts. Never weaken integrity checks to reuse incompatible results.
