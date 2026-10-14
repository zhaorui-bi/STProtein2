"""Vary Teacher behavior while retaining an identical predictive Student."""
from dataclasses import dataclass

from stprotein2_spatial_anchor.config import SpatialAnchorTrainConfig


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
