"""Fixed spatial basis; unchanged nested residual and augmentation rules."""
from dataclasses import dataclass
import math

from stprotein2_crossfit.config import ResidualTrainConfig


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
