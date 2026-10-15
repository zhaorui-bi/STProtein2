"""Dependency specification for the spatial-anchor subclass only."""
from stprotein2.execution.planner import plan_partition
from stprotein2.execution.refinement import refine_partition
from stprotein2.models.spatial import SpatialAnchoredSTProtein2


def plan_for_model(model, graph, *, refinement=None, **kwargs):
    if type(model) is not SpatialAnchoredSTProtein2 or model.cfg.local_operator != 'csnn':
        raise ValueError('Only the audited spatial-anchor CSNN dependency is supported')
    if 'required_hops' in kwargs:
        raise ValueError('Model dependency depth cannot be overridden')
    # The fixed one-hop basis is computed on the full slice. The unchanged
    # partitioned CSNN still needs two hops and full-graph coefficients.
    initial = plan_partition(graph, required_hops=2, **kwargs)
    if refinement is None:
        return initial
    return refine_partition(graph, initial, node_keys=kwargs.get('node_keys'), embedding=kwargs.get('embedding'),
                            memory_cost=kwargs.get('memory_cost'), **refinement)
