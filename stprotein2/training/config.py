"""Training configuration hierarchy for anchored spatial residual prediction."""
from dataclasses import dataclass
import math
from stprotein2.config import TrainConfig
from stprotein2.execution.memory import ExecutionPolicy

@dataclass(frozen=True)
class AnchorTrainConfig(TrainConfig):
    checkpoint_stage: str = "any"

    def __post_init__(self):
        super().__post_init__()
        if self.checkpoint_stage not in {"any", "after_ramp"}:
            raise ValueError("Unknown anchor checkpoint stage")
        if self.epochs < self.first_checkpoint_epoch:
            raise ValueError("Training budget cannot reach the requested residual checkpoint stage")

    @property
    def first_checkpoint_epoch(self):
        if self.checkpoint_stage == "any":
            return 1
        return self.warmup_epochs + max(1, self.prototype_ramp_epochs)


@dataclass(frozen=True)
class ResidualTrainConfig(AnchorTrainConfig):
    residual_target: str = "cross_fitted"
    checkpoint_stage: str = "after_ramp"
    mask_rates: tuple = (0.0,)
    count_keep_rates: tuple = (1.0,)

    def __post_init__(self):
        super().__post_init__()
        if self.residual_target not in {"in_fit", "cross_fitted"}:
            raise ValueError("Unknown residual target protocol")
        if tuple(self.mask_rates) != (0.0,) or tuple(self.count_keep_rates) != (1.0,):
            raise ValueError("C0/CF0 require matched unaugmented inputs")
        if self.checkpoint_stage != "after_ramp" or self.base_weight <= 0:
            raise ValueError("C0/CF0 require after-ramp selection and positive true-Y auxiliary loss")

    @property
    def variant(self):
        return "residual_CF0" if self.residual_target == "cross_fitted" else "residual_C0"


@dataclass(frozen=True)
class SpatialAnchorTrainConfig(ResidualTrainConfig):
    spatial_alpha: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.spatial_alpha) or not 0 <= self.spatial_alpha <= 1:
            raise ValueError('Fixed spatial alpha must be finite and in [0,1]')
        if self.graph_k < 1 or not math.isfinite(self.graph_radius) or self.graph_radius <= 0:
            raise ValueError('Valid spatial graph settings required')

    @property
    def variant(self):
        return 'spatial_basis_CF0' if self.residual_target == 'cross_fitted' else 'spatial_basis_C0'


@dataclass(frozen=True)
class TeacherAblationTrainConfig(SpatialAnchorTrainConfig):
    teacher_mode: str = "ema"

    def __post_init__(self):
        super().__post_init__()
        if self.teacher_mode not in {"ema", "no_loss", "none", "fixed"}:
            raise ValueError("Unknown Teacher ablation mode")
        if self.state_weight < 0 or self.niche_weight < 0:
            raise ValueError("Prototype weights must be nonnegative")

    @property
    def variant(self):
        target = "CF0" if self.residual_target == "cross_fitted" else "C0"
        return f"spatial_teacher_{self.teacher_mode}_{target}"


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


