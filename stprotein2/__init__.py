"""Unified STProtein2 API; imports do not select a GPU or load data."""
__version__ = "0.3.0.dev1"

__all__ = ["STProtein2", "ModelConfig", "export_run", "load_bundle", "predict_file"]

def __getattr__(name):
    if name == "STProtein2":
        from .models.spatial import SpatialAnchoredSTProtein2
        return SpatialAnchoredSTProtein2
    if name == "ModelConfig":
        from .config import ModelConfig
        return ModelConfig
    if name in {"export_run", "load_bundle", "predict_file"}:
        from .inference import bundle
        return getattr(bundle, name)
    raise AttributeError(name)
