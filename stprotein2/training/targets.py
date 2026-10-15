"""An OOF spatial basis can use only the OOF anchor's RNA predictions."""
from dataclasses import dataclass

import numpy as np

from stprotein2.graph import physical_graph, digest_array
from stprotein2.training.residual_targets import BaselineTarget, get_anchor, build_targets as build_rna_targets, validate_targets as validate_rna_targets
from stprotein2.models.smoothing import spatial_residual, apply


@dataclass(frozen=True)
class SpatialBaselineTarget(BaselineTarget):
    rna_prediction: np.ndarray

    def __post_init__(self):
        super().__post_init__()
        raw = np.array(self.rna_prediction, dtype=np.float32, copy=True)
        if raw.shape != self.prediction.shape or not np.isfinite(raw).all():
            raise ValueError('Aligned raw RNA baseline required')
        raw.setflags(write=False)
        object.__setattr__(self, 'rna_prediction', raw)

    def signature(self):
        return {**super().signature(), 'rna_prediction_sha256': digest_array(self.rna_prediction)}


def build_targets(cohort, names, n_genes, config, full_anchor):
    raw = build_rna_targets(cohort, names, n_genes, config.residual_target, full_anchor, config.threads)
    result = {}
    before = list(cohort.protein_reads)
    for name, target in raw.items():
        rna = cohort.rna(name)
        if tuple(str(b) for b in rna.obs_names) != target.barcodes:
            raise ValueError('Spatial target coordinate row identity differs')
        graph = physical_graph(rna.obsm['spatial'], config.graph_k, config.graph_radius)
        delta = spatial_residual(target.prediction, graph)
        base = apply(target.prediction, delta, config.spatial_alpha)
        provenance = {**target.provenance, 'spatial_basis': {
            'alpha': config.spatial_alpha, 'graph_k': config.graph_k, 'graph_radius': config.graph_radius,
            'graph_sha256': graph.fingerprint, 'raw_prediction_sha256': digest_array(target.prediction),
            'RNA_only_spatial_operation': True, 'center': 'whole_slice_nonisolated_nodes'}}
        result[name] = SpatialBaselineTarget(name, target.group, target.barcodes, base, provenance, target.prediction)
    if cohort.protein_reads != before:
        raise RuntimeError('Spatial basis construction accessed protein measurements')
    return result


def validate_targets(references, baselines, config):
    # This validates every inherited anchor/group/row exclusion contract.
    validate_rna_targets(references, baselines, config.residual_target)
    for sample in references:
        target = baselines[sample.rna.name]
        if not isinstance(target, SpatialBaselineTarget):
            raise ValueError('Spatial baseline with raw RNA provenance required')
        graph = physical_graph(sample.rna.coordinates, config.graph_k, config.graph_radius)
        expected = {'alpha': config.spatial_alpha, 'graph_k': config.graph_k, 'graph_radius': config.graph_radius,
            'graph_sha256': graph.fingerprint, 'raw_prediction_sha256': digest_array(target.rna_prediction),
            'RNA_only_spatial_operation': True, 'center': 'whole_slice_nonisolated_nodes'}
        if target.provenance.get('spatial_basis') != expected:
            raise ValueError('Spatial target graph, alpha or raw prediction identity differs')
        rebuilt = apply(target.rna_prediction, spatial_residual(target.rna_prediction, graph), config.spatial_alpha)
        if not np.array_equal(rebuilt, target.prediction):
            raise ValueError('Spatial target does not match its frozen RNA base and graph')
    return {s.rna.name: baselines[s.rna.name].signature() for s in references}
