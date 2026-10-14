"""Avoid N x members x genes expansion without changing learned parameters."""
from dataclasses import asdict, dataclass

import torch
from torch.utils.checkpoint import checkpoint

from stprotein2.prototypes import EnsembleLinear, Representations


@dataclass(frozen=True)
class ExecutionPolicy:
    weight_modulated_input: bool = True
    checkpoint_representations: bool = False

    def __post_init__(self):
        if not isinstance(self.weight_modulated_input, bool) or not isinstance(self.checkpoint_representations, bool):
            raise ValueError("Execution policy flags must be booleans")

    def to_dict(self):
        return asdict(self)


class WeightModulatedEnsembleLinear(EnsembleLinear):
    def forward(self, x):
        if x.ndim != 2:
            return super().forward(x)
        members, genes = self.r.shape
        if x.shape[1] != genes:
            raise ValueError("RNA feature count differs from ensemble weights")
        output = self.weight.shape[1]
        # G x K x O, instead of the original N x K x G input temporary.
        weights = (self.weight[:, None, :] * self.r.T[:, :, None]).reshape(genes, members*output)
        value = (x @ weights).reshape(len(x), members, output)
        return value * self.s + self.bias


class CheckpointRepresentations(Representations):
    def forward(self, x, availability, qc, graph):
        if not self.training or not torch.is_grad_enabled():
            return super().forward(x, availability, qc, graph)
        # The graph object and full-graph neighbor means remain in the differentiable computation.
        def evaluate(a, b, c):
            return super(CheckpointRepresentations, self).forward(a, b, c, graph)
        return checkpoint(evaluate, x, availability, qc, use_reentrant=False, preserve_rng_state=True)


def configure_execution(model, policy=ExecutionPolicy()):
    """Change only this model's runtime dispatch, never Parameter identities.

    Reapplying either policy is supported. Call before constructing a copied
    Teacher, and record this policy in a new experiment's metadata.
    """
    if not isinstance(policy, ExecutionPolicy):
        raise TypeError("An explicit ExecutionPolicy is required")
    parameters = {n: id(p) for n,p in model.named_parameters()}
    buffers = {n: id(p) for n,p in model.named_buffers()}
    changed = {"ensemble_layers": 0, "representation_modules": 0}
    for module in model.modules():
        if type(module) in (EnsembleLinear, WeightModulatedEnsembleLinear):
            module.__class__ = WeightModulatedEnsembleLinear if policy.weight_modulated_input else EnsembleLinear
            changed["ensemble_layers"] += 1
        elif type(module) in (Representations, CheckpointRepresentations):
            module.__class__ = CheckpointRepresentations if policy.checkpoint_representations else Representations
            changed["representation_modules"] += 1
    if not changed["ensemble_layers"] or not changed["representation_modules"]:
        raise ValueError("Expected a complete STProtein2 representation and ensemble model")
    assert parameters == {n:id(p) for n,p in model.named_parameters()}
    assert buffers == {n:id(p) for n,p in model.named_buffers()}
    return {"policy": policy.to_dict(), **changed, "parameter_identities_preserved": True,
            "state_dict_keys_preserved": True}
