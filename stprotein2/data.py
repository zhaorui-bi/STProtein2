"""RNA-only preprocessing and a separate, identity-checked protein reader."""
from dataclasses import dataclass, replace
import json
import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from stprotein2.graph import digest_array


PANEL = tuple(sorted("ACTA2 BCL2 CCR7 CD14 CD163 CD19 CD27 CD274 CD3E CD4 CD40 CD68 CD8A CEACAM8 CR2 CXCR5 EPCAM FCGR3A HLA_DRA ITGAM ITGAX KRT5 MS4A1 PAX5 PCNA PDCD1 PECAM1 PTPRC_1 PTPRC_2 SDC1 VIM".split()))


def counts(x):
    x = sparse.csr_matrix(x, dtype=np.float32)
    if not np.isfinite(x.data).all() or np.any(x.data < 0) or not np.allclose(x.data, np.rint(x.data), atol=1e-5):
        raise ValueError("Raw nonnegative integer counts required")
    return x


def normalized_log(x):
    x = counts(x)
    totals = np.asarray(x.sum(1)).ravel().astype(np.float64)
    scale = (10000 / np.maximum(totals, 1)).astype(np.float32)
    result = x.multiply(scale[:, None])
    result.data = np.log1p(result.data)
    return result.tocsr()


def dispersion_score(mean, variance):
    dispersion = np.zeros_like(mean, dtype=np.float64)
    valid = mean > 0
    dispersion[valid] = variance[valid] / (mean[valid] + 1e-12)
    log_dispersion, log_mean = np.log(dispersion + 1e-12), np.log1p(mean)
    edges = np.quantile(log_mean, np.linspace(0, 1, 21))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    bins = np.digitize(log_mean, edges[1:-1], right=False)
    score = np.zeros_like(mean)
    for group in range(20):
        take = bins == group
        if take.any():
            sd = log_dispersion[take].std()
            score[take] = (log_dispersion[take] - log_dispersion[take].mean()) / (sd if sd >= 1e-8 else 1)
    return score


def raw_qc(rna):
    x = counts(rna.X)
    library = np.asarray(x.sum(1)).ravel()
    expressed = np.asarray((x > 0).sum(1)).ravel()
    symbols = rna.var.get("gene_symbol_original", rna.var.get("gene_symbols", rna.var_names))
    mitochondrial = np.asarray([str(g).upper().startswith("MT-") for g in symbols])
    mt = np.asarray(x[:, mitochondrial].sum(1)).ravel() / np.maximum(library, 1)
    return np.column_stack([np.log1p(library), np.log1p(expressed), mt, np.ones(len(rna))]).astype(np.float32)


@dataclass
class RNAView:
    name: str
    group: str
    x: np.ndarray
    availability: np.ndarray
    qc: np.ndarray
    coordinates: np.ndarray
    barcodes: np.ndarray
    genes: tuple
    coverage_scale: float = 1.0

    def __post_init__(self):
        n, g = self.x.shape
        if self.availability.shape not in {(g,), (n, g)}:
            raise ValueError("Invalid availability shape")
        if self.qc.shape != (n, 4) or self.coordinates.shape != (n, 2) or len(self.barcodes) != n:
            raise ValueError("RNA row identity mismatch")
        if len(set(self.barcodes)) != n or len(self.genes) != g or len(set(self.genes)) != g:
            raise ValueError("Unique barcodes and genes required")
        for a in (self.x, self.availability, self.qc, self.coordinates):
            if not np.isfinite(a).all():
                raise ValueError("Non-finite RNA input")
        if not np.all((self.availability == 0) | (self.availability == 1)):
            raise ValueError("Availability must be binary")

    def masked(self, gene_keep):
        gene_keep = np.asarray(gene_keep, dtype=np.float32)
        if gene_keep.shape != (self.x.shape[1],) or not np.all((gene_keep == 0) | (gene_keep == 1)):
            raise ValueError("Invalid feature mask")
        available = self.availability * gene_keep
        old_mean = np.broadcast_to(self.availability, self.x.shape).mean(1)
        new_mean = np.broadcast_to(available, self.x.shape).mean(1)
        qc = self.qc.copy(); qc[:, 3] += (new_mean - old_mean) / self.coverage_scale
        return replace(self, x=np.where(available > 0, self.x, 0).astype(np.float32),
                       availability=available, qc=qc)

    def fingerprint(self):
        return {"x": digest_array(self.x), "availability": digest_array(self.availability),
                "coordinates": digest_array(self.coordinates), "genes": list(self.genes)}


@dataclass
class TrainingSlice:
    rna: RNAView
    target: np.ndarray
    count_views: tuple = ()

    def __post_init__(self):
        if self.target.shape != (len(self.rna.x), len(PANEL)) or not np.isfinite(self.target).all():
            raise ValueError("Invalid named protein target")
        for view in self.count_views:
            if view.name != self.rna.name or view.group != self.rna.group or view.genes != self.rna.genes or not np.array_equal(view.barcodes, self.rna.barcodes) or not np.array_equal(view.coordinates, self.rna.coordinates):
                raise ValueError("Count augmentation changed biological row/feature identity")


def count_thinning_views(rna, transform, name, group, rates, replicas, seed):
    """Fixed, reproducible raw-RNA views; same biological target, recomputed QC.

    Reference statistics are never refitted on a corrupted view. This finite
    augmentation bank is distinct from independently resampling every epoch.
    """
    if len(set(rates)) != len(rates) or any(not 0 < rate <= 1 for rate in rates) or replicas < 1:
        raise ValueError("Invalid raw-count augmentation schedule")
    if all(rate == 1 for rate in rates):
        return ()
    source = counts(rna.X)
    original_counts = source.data.astype(np.int64)
    salt = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
    rng = np.random.default_rng([seed, salt])
    result = []
    for rate in rates:
        if rate == 1:
            continue
        for _ in range(replicas):
            observed = rna.copy()
            matrix = source.copy()
            matrix.data = rng.binomial(original_counts, rate).astype(np.float32)
            matrix.eliminate_zeros(); observed.X = matrix
            result.append(transform.transform(observed, name, group))
    return tuple(result)


class ReferenceTransform:
    """Historical DPAS-controlled HVG/normalization semantics, independently implemented.

    HVGs use the reference intersection before normalization. Final feature
    normalization subsets selected genes first, matching the historical track.
    Standardization and QC statistics are then fitted only on references.
    """
    def __init__(self, n_genes=4000):
        self.n_genes = n_genes

    def fit(self, references, names):
        if len(references) != len(names) or not references or len(set(names)) != len(names):
            raise ValueError("Explicit unique training reference IDs required")
        common = set(references[0].var_names)
        for a in references:
            if not a.var_names.is_unique or not a.obs_names.is_unique:
                raise ValueError("Ambiguous RNA identity")
            common.intersection_update(a.var_names)
        genes = [g for g in references[0].var_names if g in common]
        if len(genes) < self.n_genes:
            raise ValueError("Insufficient common reference genes")
        total = np.zeros(len(genes), dtype=np.float64); squares = total.copy(); n = 0
        for a in references:
            z = normalized_log(a[:, genes].X)
            total += np.asarray(z.sum(0)).ravel()
            squares += np.asarray(z.power(2).sum(0)).ravel()
            n += len(a)
        if n < 2:
            raise ValueError("Insufficient reference spots")
        mean = total / n
        variance = np.maximum((squares - total ** 2 / n) / (n - 1), 0)
        rank = np.argsort(-dispersion_score(mean, variance), kind="mergesort")[:self.n_genes]
        self.genes_ = tuple(genes[i] for i in rank)
        self.scaler_ = StandardScaler()
        self.qc_scaler_ = StandardScaler()
        for a in references:
            self.scaler_.partial_fit(normalized_log(a[:, list(self.genes_)].X).toarray())
            self.qc_scaler_.partial_fit(raw_qc(a))
        self.fitted_names_ = list(names)
        return self

    def transform(self, rna, name, group):
        if not rna.var_names.is_unique or not rna.obs_names.is_unique:
            raise ValueError("Ambiguous query RNA identity")
        index = rna.var_names.get_indexer(self.genes_)
        available = (index >= 0).astype(np.float32)
        source = counts(rna.X)
        observed = source[:, index[index >= 0]].tocoo()
        columns = np.flatnonzero(index >= 0)[observed.col]
        panel = sparse.csr_matrix((observed.data, (observed.row, columns)), shape=(len(rna), len(index)))
        x = self.scaler_.transform(normalized_log(panel).toarray()).astype(np.float32)
        x[:, index < 0] = 0
        qc = raw_qc(rna); qc[:, 3] = available.mean()
        qc = self.qc_scaler_.transform(qc).astype(np.float32)
        return RNAView(name, group, x, available, qc, np.asarray(rna.obsm["spatial"], dtype=np.float64),
                       np.asarray(rna.obs_names, dtype=str), self.genes_, float(self.qc_scaler_.scale_[3]))


class Cohort:
    """Lazy RNA and measured-protein readers have separate entry points."""
    def __init__(self, registry, groups=None):
        self.registry_path = Path(registry).resolve()
        entries = json.loads(self.registry_path.read_text())
        if not isinstance(entries, list) or len({s["name"] for s in entries}) != len(entries):
            raise ValueError("Registry requires unique sample names")
        self.specs = {s["name"]: s for s in entries}
        self.groups = groups or {name: name for name in self.specs}
        if set(self.groups) != set(self.specs):
            raise ValueError("Every sample requires a declared group")
        self._rna = {}
        self.protein_reads = []

    def rna(self, name):
        if name not in self._rna:
            self._rna[name] = ad.read_h5ad(self.specs[name]["rna"])
        return self._rna[name]

    def input(self, name, transform):
        return transform.transform(self.rna(name), name, self.groups[name])

    def measured_protein(self, name):
        self.protein_reads.append(name)
        protein = ad.read_h5ad(self.specs[name]["adt"])
        rna = self.rna(name)
        if not protein.obs_names.is_unique or not protein.var_names.is_unique:
            raise ValueError("Ambiguous protein identity")
        if not set(PANEL) <= set(protein.var_names) or not set(rna.obs_names) <= set(protein.obs_names):
            raise ValueError("Protein panel/barcode mismatch")
        values = counts(protein[rna.obs_names, list(PANEL)].X).toarray().astype(np.float64)
        log = np.log1p(values)
        return (log - log.mean(1, keepdims=True)).astype(np.float32)

    def references(self, names, transform, count_keep_rates=(1.,), count_views_per_rate=1, seed=2025):
        return [TrainingSlice(self.input(name, transform), self.measured_protein(name),
                    count_thinning_views(self.rna(name), transform, name, self.groups[name],
                                         count_keep_rates, count_views_per_rate, seed)) for name in names]

