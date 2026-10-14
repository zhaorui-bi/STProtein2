"""Register initialization and execution separately from model/Teacher choices."""
from dataclasses import dataclass

from stprotein2_execution.memory import ExecutionPolicy
from stprotein2_teacher_ablation_v2.config import TeacherAblationTrainConfig


@dataclass(frozen=True)
class BalancedTrainConfig(TeacherAblationTrainConfig):
    initialization: str = "hierarchical"
    router_components: int = 50
    router_clusters: int = 32
    router_per_source: int = 4000
    prototype_per_source: int = 2000
    weight_modulated_input: bool = True
    checkpoint_representations: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.initialization not in {"pooled", "hierarchical"}:
            raise ValueError("Unknown initialization policy")
        for name in ("router_components", "router_clusters", "router_per_source", "prototype_per_source"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (2 if name == "router_clusters" else 1):
                raise ValueError("Invalid initialization integer: " + name)
        if self.initialization == "pooled" and self.prototype_per_source != 2000:
            raise ValueError("Historical pooled prototype initializer uses a fixed cap of 2000")
        self.execution_policy

    @property
    def execution_policy(self):
        return ExecutionPolicy(self.weight_modulated_input, self.checkpoint_representations)

    @property
    def variant(self):
        target = "CF0" if self.residual_target == "cross_fitted" else "C0"
        return f"balanced_{self.initialization}_teacher_{self.teacher_mode}_{target}"
