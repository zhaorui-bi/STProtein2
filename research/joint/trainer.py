"""Historical joint model trainer for development controls."""
import copy
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from stprotein2.graph import physical_graph, partition_graph, boundary_score
from stprotein2.io import save_torch, write_json
from stprotein2.models.residual import STProtein2
from stprotein2.models.prototypes import EMATeacher, prototype_diagnostics

from stprotein2.training.basic import seed_all, tensor_input, partitions_for, predict, initialize_prototypes

def fit(references, model_config, train_config, router, output, device="cpu", validation=None):
    """Validation, if provided, is explicitly an INNER reference TrainingSlice.

    A complete query must never be passed as validation for a claimed outer
    evaluation. The runner records names/groups and enforces this separation.
    """
    if not references:
        raise ValueError("Training references required")
    names = [s.rna.name for s in references]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate training references")
    expected_count_views = (len(train_config.count_keep_rates) - 1) * train_config.count_views_per_rate
    if any(len(s.count_views) != expected_count_views for s in references):
        raise ValueError("Prepared count views do not match the declared training augmentation")
    validation_slices = [] if validation is None else (validation if isinstance(validation, (list, tuple)) else [validation])
    validation_names = [s.rna.name for s in validation_slices]
    if len(set(validation_names)) != len(validation_names):
        raise ValueError("Duplicate inner validation slices")
    if any(s.rna.name in names or s.rna.group in {r.rna.group for r in references} for s in validation_slices):
        raise ValueError("Inner validation group overlaps training")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(train_config.threads)
    seed_all(train_config.seed)
    rng = np.random.default_rng(train_config.seed)
    model = STProtein2(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    graphs = {s.rna.name: physical_graph(s.rna.coordinates, train_config.graph_k, train_config.graph_radius) for s in references}
    by_group = {}
    for sample in references:
        by_group.setdefault(sample.rna.group, []).append(sample)
    teacher = None
    history, start_epoch, best, best_epoch, stale = [], 1, float("inf"), None, 0
    resume = output / "resume.pt"
    if resume.exists():
        # This is our own manifest-protected local checkpoint, not arbitrary input.
        saved = torch.load(resume, map_location=device, weights_only=False)
        if saved["model_config"] != asdict(model_config) or saved["train_config"] != asdict(train_config) or saved["reference_names"] != names:
            raise RuntimeError("Checkpoint identity mismatch")
        if saved.get("validation_names", validation_names) != validation_names:
            raise RuntimeError("Checkpoint validation identity mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        if saved["teacher"] is not None:
            teacher = EMATeacher(model.representations, train_config.center_momentum).to(device)
            teacher.load_state_dict(saved["teacher"])
        history, best, best_epoch, stale = saved["history"], saved["best"], saved["best_epoch"], saved["stale"]
        start_epoch = saved["epoch"] + 1
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.startswith("cuda") and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
    elapsed_before_resume = history[-1]["elapsed_seconds"] if history else 0.
    start = time.monotonic()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, train_config.epochs + 1):
        if epoch == train_config.warmup_epochs + 1:
            initialize_prototypes(model, references, graphs, device, train_config.seed)
            if (model_config.use_state and train_config.state_weight > 0) or (model_config.use_niche and train_config.niche_weight > 0):
                teacher = EMATeacher(model.representations, train_config.center_momentum).to(device)
        ramp = min(1., max(0., (epoch - train_config.warmup_epochs) / max(train_config.prototype_ramp_epochs, 1)))
        progress = min(1., max(0., (epoch - 1) / max(train_config.epochs - 1, 1)))
        momentum = train_config.ema_start + (train_config.ema_end - train_config.ema_start) * progress
        temp = train_config.teacher_temp_start + (train_config.teacher_temp_end - train_config.teacher_temp_start) * ramp
        totals = {"loss": 0., "protein": 0., "state_proto": 0., "niche_proto": 0.}
        model.train()
        # Each group has equal expected weight; slices within a group are uniform.
        group_names = list(by_group)
        for index in rng.permutation(len(group_names)):
            group = group_names[index]
            sample = by_group[group][int(rng.integers(len(by_group[group])))]
            graph = graphs[sample.rna.name]
            views = (sample.rna,) + sample.count_views
            source_view = views[int(rng.integers(len(views)))] if sample.count_views else sample.rna
            rate = float(rng.choice(train_config.mask_rates))
            keep = np.ones(model_config.n_genes, dtype=np.float32)
            if rate > 0:
                keep[rng.choice(len(keep), int(round(rate * len(keep))), replace=False)] = 0
            strong = source_view.masked(keep)
            parts = partitions_for(strong, graph, router, train_config) if train_config.execution_mode == "adaptive" and model_config.local_operator == "csnn" and epoch > train_config.warmup_epochs else None
            args = tensor_input(strong, graph, device)
            target = torch.as_tensor(sample.target, device=device)
            optimizer.zero_grad(set_to_none=True)
            result = model(*args, partitions=[p.to(device) for p in parts] if parts is not None else None,
                           enable_corrections=epoch > train_config.warmup_epochs)
            protein = F.mse_loss(result["members"], target[:, None, :].expand_as(result["members"]))
            base = F.mse_loss(result["base_members"], target[:, None, :].expand_as(result["base_members"]))
            loss = protein + train_config.base_weight * base
            proto = {"state": protein.new_zeros(()), "niche": protein.new_zeros(())}
            teacher_result = None
            if teacher is not None:
                teacher_result = teacher(*tensor_input(sample.rna, graph, device))
                proto = teacher.loss(result["representations"], teacher_result, temp)
                loss = loss + ramp * (train_config.state_weight * proto["state"] + train_config.niche_weight * proto["niche"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.clip_grad, error_if_nonfinite=True)
            optimizer.step()
            if teacher is not None:
                teacher.update(model.representations, momentum)
                teacher.update_centers(teacher_result)
            totals["loss"] += float(loss.detach()); totals["protein"] += float(protein.detach())
            totals["state_proto"] += float(proto["state"].detach()); totals["niche_proto"] += float(proto["niche"].detach())
        row = {"epoch": epoch, **{k: v / len(by_group) for k, v in totals.items()},
               "elapsed_seconds": elapsed_before_resume + time.monotonic() - start, "last_gradient_norm": float(norm),
               "optimizer_updates": epoch * len(by_group),
               "teacher_updates": int(teacher.update_count) if teacher is not None else 0}
        if validation_slices and (epoch == 1 or epoch % train_config.validate_every == 0 or epoch == train_config.epochs):
            validation_scores = {}
            for sample in validation_slices:
                prediction, _ = predict(model, sample.rna, router, train_config, device)
                validation_scores[sample.rna.name] = float(np.mean((prediction["prediction"] - sample.target) ** 2, dtype=np.float64))
            score = float(np.mean([np.mean([validation_scores[s.rna.name] for s in validation_slices if s.rna.group == group])
                                   for group in sorted({s.rna.group for s in validation_slices})]))
            row["reference_validation_by_slice"] = validation_scores
            row["reference_validation_mse"] = score
            if score < best:
                best, best_epoch, stale = score, epoch, 0
                save_torch(output / "best.pt", {"state_dict": model.state_dict(), "epoch": epoch})
            else:
                stale += 1
        history.append(row)
        if epoch == 1 or epoch % train_config.validate_every == 0 or epoch == train_config.epochs:
            save_torch(resume, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "teacher": teacher.state_dict() if teacher is not None else None, "history": history,
                       "epoch": epoch, "best": best, "best_epoch": best_epoch, "stale": stale,
                       "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
                       "reference_names": names, "validation_names": validation_names,
                       "model_config": asdict(model_config), "train_config": asdict(train_config)})
            write_json(output / "history.json", history)
            print(f"epoch={epoch} protein_mse={row['protein']:.6f} loss={row['loss']:.6f} validation={row.get('reference_validation_mse')} seconds={row['elapsed_seconds']:.1f}", flush=True)
        if validation_slices and stale >= train_config.patience:
            break
    if validation_slices:
        selected = torch.load(output / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(selected["state_dict"])
    else:
        best_epoch = history[-1]["epoch"]
    model.eval()
    report = {"reference_names": names, "reference_groups": sorted(by_group),
              "inner_validation": validation_names,
              "inner_validation_groups": sorted({s.rna.group for s in validation_slices}),
              "selected_epoch": best_epoch, "reference_validation_mse": best if validation_slices else None,
              "architecture": model.architecture(), "teacher_parameters": sum(p.numel() for p in teacher.parameters()) if teacher else 0,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
              "history": history, "query_protein_used": False,
              "training_elapsed_seconds": history[-1]["elapsed_seconds"],
              "count_augmentation_views_per_slice": {s.rna.name: len(s.count_views) for s in references},
              "training_sampling": "one uniformly selected slice per group per epoch; equally weighted groups"}
    save_torch(output / "model.pt", {"model_config": asdict(model_config), "state_dict": model.state_dict()})
    write_json(output / "training.json", report)
    return model, report
