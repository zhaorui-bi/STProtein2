"""Explicit, serializable experimental settings; unknown keys are rejected."""
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class ModelConfig:
    n_genes: int = 4000
    n_proteins: int = 31
    width: int = 256
    hidden: int = 128
    members: int = 8
    qc_dim: int = 4
    use_encoder_qc: bool = True
    dropout: float = 0.1
    state_prototypes: int = 32
    niche_prototypes: int = 16
    prototype_dim: int = 128
    prototype_temperature: float = 0.1
    use_state: bool = True
    use_niche: bool = True
    local_operator: str = "csnn"
    experts: int = 4
    stalk: int = 8
    channels: int = 8
    protein_embedding: int = 16
    protein_conditioning: bool = True
    center_local: bool = True
    checkpoint_blocks: bool = True

    def __post_init__(self):
        for key in ("n_genes", "n_proteins", "width", "hidden", "members", "qc_dim",
                    "state_prototypes", "niche_prototypes", "prototype_dim", "experts",
                    "stalk", "channels", "protein_embedding"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.local_operator not in {"csnn", "gat", "none"}:
            raise ValueError("Unknown local operator")
        if not 0 <= self.dropout < 1 or self.prototype_temperature <= 0:
            raise ValueError("Invalid dropout or temperature")
        if self.stalk < 2 and self.local_operator == "csnn":
            raise ValueError("CSNN requires stalk >= 2")
        if self.state_prototypes < 2 or self.niche_prototypes < 2:
            raise ValueError("Prototype banks require at least two codes")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 300
    warmup_epochs: int = 20
    prototype_ramp_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip_grad: float = 1.0
    base_weight: float = 0.2
    state_weight: float = 0.1
    niche_weight: float = 0.05
    ema_start: float = 0.99
    ema_end: float = 0.999
    teacher_temp_start: float = 0.07
    teacher_temp_end: float = 0.04
    center_momentum: float = 0.9
    mask_rates: tuple = (0.0, 0.1, 0.2)
    count_keep_rates: tuple = (1.0,)
    count_views_per_rate: int = 1
    validate_every: int = 5
    patience: int = 6
    core_size: int = 256
    min_core: int = 128
    max_core: int = 512
    graph_k: int = 6
    graph_radius: float = 2.5
    execution_mode: str = "adaptive"
    threads: int = 4
    seed: int = 2025

    def __post_init__(self):
        if min(self.epochs, self.validate_every, self.patience, self.core_size, self.threads) < 1:
            raise ValueError("Epochs, intervals, threads and core size must be positive")
        if not 0 <= self.warmup_epochs <= self.epochs:
            raise ValueError("Warm-up must be within the training budget")
        if not 0 < self.ema_start <= self.ema_end < 1:
            raise ValueError("Invalid EMA schedule")
        if not self.mask_rates or any(not 0 <= r < 1 for r in self.mask_rates):
            raise ValueError("Invalid masking rates")
        if not self.count_keep_rates or 1.0 not in self.count_keep_rates or any(not 0 < r <= 1 for r in self.count_keep_rates):
            raise ValueError("Count keep rates must include 1 and remain within (0, 1]")
        if len(set(self.count_keep_rates)) != len(self.count_keep_rates) or self.count_views_per_rate < 1:
            raise ValueError("Count views require distinct rates and a positive replica count")
        if self.min_core > self.max_core:
            raise ValueError("Invalid core size limits")
        if self.execution_mode not in {"adaptive", "full"}:
            raise ValueError("Execution mode must be adaptive or full")

    def to_dict(self):
        return asdict(self)


def from_dict(cls, values):
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**values)

