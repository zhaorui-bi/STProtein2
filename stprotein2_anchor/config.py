"""Anchor-specific checkpoint policy, separate from the frozen joint models."""
from dataclasses import dataclass

from stprotein2.config import TrainConfig


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
