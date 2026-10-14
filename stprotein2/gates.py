"""Conservative reference-OOF selection of per-protein correction switches."""
import numpy as np


CHOICES = ((0, 0), (1, 0), (0, 1), (1, 1))
CHOICE_NAMES = ("none", "state", "local", "both")


def component_losses(target, arrays):
    """Four predeclared choices, without selecting against this target here."""
    target = np.asarray(target, dtype=np.float64)
    base, state, local = (np.asarray(arrays[k], dtype=np.float64)
                          for k in ("base", "state_delta", "local_delta"))
    if any(a.shape != target.shape or not np.isfinite(a).all() for a in (target, base, state, local)):
        raise ValueError("Finite, aligned OOF targets and components required")
    return np.stack([np.mean((base + s * state + l * local - target) ** 2, axis=0)
                     for s, l in CHOICES])


def select_gates(group_losses, proteins, min_groups=4, min_gain=.01, min_fraction=.6):
    """Input is group -> 4 x proteins, averaging slices equally within a group.

    Each group's predictions must come from a model whose training AND epoch
    selection excluded that group. The nested runner establishes this contract.
    This threshold is a development heuristic, not a coverage or FDR guarantee.
    """
    if min_groups < 2 or min_gain < 0 or not 0 < min_fraction <= 1:
        raise ValueError("Invalid predeclared gate thresholds")
    names = sorted(group_losses)
    p = len(proteins)
    losses = np.stack([np.asarray(group_losses[g], dtype=float) for g in names]) if names else np.empty((0, 4, p))
    if losses.shape != (len(names), 4, p) or not np.isfinite(losses).all() or np.any(losses < 0):
        raise ValueError("Invalid group OOF losses")
    state, local = np.zeros(p, dtype=int), np.zeros(p, dtype=int)
    rows = []
    for j, protein in enumerate(proteins):
        valid = losses[:, 0, j] > 1e-12
        n = int(valid.sum())
        gains = (losses[valid, 0:1, j] - losses[valid, :, j]) / losses[valid, 0:1, j] if n else np.empty((0, 4))
        average = gains.mean(0) if n else np.zeros(4)
        fraction = (gains > 0).mean(0) if n else np.zeros(4)
        eligible = [k for k in range(1, 4) if n >= min_groups and average[k] >= min_gain and fraction[k] >= min_fraction]
        chosen = 0
        if eligible:
            best = max(average[k] for k in eligible)
            tied = [k for k in eligible if np.isclose(average[k], best, rtol=0, atol=1e-12)]
            chosen = min(tied, key=lambda k: (sum(CHOICES[k]), k))
        state[j], local[j] = CHOICES[chosen]
        rows.append({"protein": protein, "choice": CHOICE_NAMES[chosen], "valid_groups": n,
                     "mean_relative_mse_gain": dict(zip(CHOICE_NAMES, average.tolist())),
                     "fraction_improved": dict(zip(CHOICE_NAMES, fraction.tolist())),
                     "reason": "insufficient_groups" if n < min_groups else "reference_OOF_rule"})
    return {"gamma_state": state.tolist(), "gamma_local": local.tolist(), "proteins": list(proteins),
            "groups": names, "thresholds": {"min_groups": min_groups, "min_relative_mse_gain": min_gain,
                                             "min_fraction_groups_improved": min_fraction},
            "per_protein": rows, "guarantee": None}
