"""RNA representations, prototypes, sheaf experts and frozen spatial model."""
from .spatial import SpatialAnchoredSTProtein2 as STProtein2
from .anchor import FrozenRidge, AnchoredSTProtein2
__all__ = ["STProtein2", "FrozenRidge", "AnchoredSTProtein2"]
