"""A single-factor residual-target experiment with no input augmentation."""
from dataclasses import dataclass

from stprotein2_anchor.config import AnchorTrainConfig


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
