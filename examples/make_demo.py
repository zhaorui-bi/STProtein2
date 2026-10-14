"""Generate a small synthetic cohort; no query protein file is created."""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from stprotein2.data import PANEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo/data"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists; use a fresh directory")
    output.mkdir(parents=True)
    rng = np.random.default_rng(2025)
    entries, groups = [], {}
    for i, name in enumerate(("reference_a", "reference_b", "reference_c", "query")):
        rna = ad.AnnData(sparse.csr_matrix(rng.poisson(3 + i / 3, (16, 8)).astype(np.float32)))
        rna.obs_names = [f"{name}_spot_{j}" for j in range(16)]
        rna.var_names = [f"gene_{j}" for j in range(8)]
        rna.obsm["spatial"] = np.asarray([(j // 4, j % 4) for j in range(16)], dtype=float)
        rna_path, adt_path = output / f"{name}_rna.h5ad", output / f"{name}_adt.h5ad"
        rna.write_h5ad(rna_path)
        if name != "query":
            adt = ad.AnnData(sparse.csr_matrix(rng.poisson(2 + i, (16, len(PANEL))).astype(np.float32)))
            adt.obs_names, adt.var_names = rna.obs_names, PANEL
            adt.write_h5ad(adt_path)
        entries.append({"name": name, "rna": str(rna_path), "adt": str(adt_path)})
        groups[name] = name
    (output / "registry.json").write_text(json.dumps(entries, indent=2) + "\n")
    (output / "groups.json").write_text(json.dumps({"groups": groups}, indent=2) + "\n")
    print(f"Created synthetic cohort in {output}; query ADT is absent.")


if __name__ == "__main__":
    main()
