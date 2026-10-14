"""Pinned DGAT components with source-group reference-only model selection.

The author network, graph builders, five losses and single-device gradient
semantics are retained. Common preprocessing, group-balanced sampling and
reference validation replace native preprocessing and training-only EB stopping.
This adapter does not reproduce published scores or the dual-device variant.
"""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
import torch
from torch import nn

from stprotein2.data import PANEL, RNAView
from stprotein2.graph import digest_array
from stprotein2.io import save_torch, sha256, write_json
from stprotein2.trainer import seed_all

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent / "external/DGAT"
PIN = "b0128f342de70fa04546d27f9967eab1821f0933"


def upstream():
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if commit != PIN or subprocess.check_output(
            ["git", "-C", str(REPO), "diff", "HEAD", "--name-only"], text=True).strip():
        raise RuntimeError("DGAT tracked source differs from the pinned author checkout")
    for name in ("Model", "utils"):
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve().parent != REPO / name:
            raise RuntimeError(f"Conflicting upstream import namespace: {name}")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    model = importlib.import_module("Model.dgat")
    graph = importlib.import_module("utils.Graph_utils")
    return model, graph


class DGAT(nn.Module):
    """Original modules, in the author's initialization order, on one device."""

    def __init__(self, genes, hidden=1024, heads=16):
        super().__init__()
        source, _ = upstream()
        self.rna_encoder = source.GATEncoder(genes, hidden, num_heads=heads, dropout=.3)
        self.rna_decoder = source.Decoder_mRNA(hidden, genes, dropout=0)
        self.protein_encoder = source.GATEncoder(len(PANEL), hidden, num_heads=heads, dropout=.3)
        self.protein_decoder = source.Decoder_Protein(hidden, list(PANEL), dropout=0)
        self.architecture = {"genes": genes, "hidden": hidden, "heads": heads, "dropout": .3}

    def protein_prediction(self, batch):
        # No protein encoder, protein graph or protein measurements are read.
        z = self.rna_encoder(batch["x"], batch["rna_edge"], batch["rna_weight"])
        return self.protein_decoder(z)

    def objective(self, batch):
        x, y = batch["x"], batch["y"]
        zx = self.rna_encoder(x, batch["rna_edge"], batch["rna_weight"])
        zy = self.protein_encoder(y, batch["protein_edge"], batch["protein_weight"])
        xr = self.rna_decoder(zx)
        yr = self.protein_decoder(zy)
        # Preserve native single-device behavior: .to(same device) is a no-op
        # even inside no_grad. Explicit detach would change author gradients.
        with torch.no_grad():
            zy_on_rna = zy.to(x.device, non_blocking=True)
        rmse = lambda pred, truth: (nn.functional.mse_loss(pred, truth) + 1e-8).sqrt()
        terms = {
            "rna_reconstruction": rmse(xr, x),
            "protein_reconstruction": rmse(yr, y),
            "alignment": nn.functional.mse_loss(zx, zy_on_rna),
            "protein_prediction": rmse(self.protein_decoder(zx.to(y.device, non_blocking=True)), y),
            "rna_prediction": rmse(self.rna_decoder(zy_on_rna), x),
        }
        weights = {"rna_reconstruction": 5., "protein_reconstruction": 1.,
                   "alignment": 1., "protein_prediction": 3., "rna_prediction": 1.}
        for name in ("alignment", "protein_prediction", "protein_reconstruction"):
            if terms[name] < .015:
                weights[name] = 0.
        return sum(weights[name] * value for name, value in terms.items()), terms

    def optimizers(self):
        return [torch.optim.Adam(module.parameters(), lr=lr, weight_decay=decay)
                for module, lr, decay in (
                    (self.rna_encoder, 5e-4, 2e-5), (self.rna_decoder, 1e-4, 1e-5),
                    (self.protein_encoder, 5e-4, 2e-5), (self.protein_decoder, 1e-4, 1e-5))]

    def clip(self):
        for module in (self.rna_encoder, self.rna_decoder, self.protein_encoder, self.protein_decoder):
            nn.utils.clip_grad_norm_(module.parameters(), 1., error_if_nonfinite=True)


def graph_tensors(view, cache, target=None):
    """RNA-only and paired graph caches have distinct input-dependent identities."""
    if not isinstance(view, RNAView):
        raise TypeError("Graph input must be an RNAView")
    if len(view.x) < 10:
        raise ValueError("Native DGAT k=10 graph requires at least ten spots")
    if target is not None and (target.shape != (len(view.x), len(PANEL)) or not np.isfinite(target).all()):
        raise ValueError("Training protein target is invalid")
    _, native = upstream()
    identity = {"mode": "rna_only" if target is None else "paired_reference", "name": view.name,
                "rna": view.fingerprint(), "barcodes": view.barcodes.tolist(),
                "target": None if target is None else digest_array(target),
                "graph_source": sha256(REPO / "utils/Graph_utils.py")}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache) / f"{key}.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("DGAT graph cache identity changed")
        tensors = saved["tensors"]
    else:
        if target is None:
            data = native.create_pyg_data_no_protein(view.x, view.coordinates)
        else:
            data = native.create_pyg_data(view.x, target, view.coordinates)
        edge = data["mRNA", "mRNA_knn", "mRNA"]
        tensors = {"rna_edge": edge.edge_index, "rna_weight": edge.edge_attr}
        if target is not None:
            edge = data["protein", "protein_knn", "protein"]
            tensors.update(protein_edge=edge.edge_index, protein_weight=edge.edge_attr)
        save_torch(path, {"identity": identity, "tensors": tensors})
    expected = {"rna_edge", "rna_weight"} | ({"protein_edge", "protein_weight"} if target is not None else set())
    if set(tensors) != expected:
        raise ValueError("DGAT graph modalities differ from declared mode")
    return tensors


def batch(view, cache, target=None, device="cpu"):
    edges = graph_tensors(view, cache, target)
    values = {"x": torch.as_tensor(view.x), **edges}
    if target is not None:
        values["y"] = torch.as_tensor(target)
    return {key: value.to(device) for key, value in values.items()}


@torch.inference_mode()
def predict(model, view, cache, device="cpu"):
    model.eval()
    value = model.protein_prediction(batch(view, cache, device=device)).cpu().numpy()
    if value.shape != (len(view.x), len(PANEL)) or not np.isfinite(value).all():
        raise FloatingPointError("Invalid DGAT RNA-only prediction")
    return value


def fit_model(cohort, names, validation_names, prep_path, output, budget, args):
    """Matched reference-group validation; query labels have no selection role."""
    if budget < 1 or not names or set(names) & set(validation_names):
        raise ValueError("Invalid fit budget or sample split")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prep = joblib.load(prep_path)
    transform = prep["transform"]
    if transform.fitted_names_ != names or prep["reference_names"] != names:
        raise ValueError("Preprocessing training identity differs")
    identity = {"training": list(names), "validation": list(validation_names), "budget": budget,
                "seed": args.seed, "device": args.device, "preprocessing_sha256": sha256(prep_path),
                "adapter_sha256": sha256(__file__), "upstream_pin": PIN}
    seed_all(args.seed)
    model = DGAT(len(transform.genes_)).to(args.device)
    if (output / "training.json").exists():
        report = json.loads((output / "training.json").read_text())
        if report["identity"] != identity or report["model_sha256"] != sha256(output / "model.pt"):
            raise ValueError("Completed DGAT fit identity differs")
        model.load_state_dict(torch.load(output / "model.pt", map_location=args.device, weights_only=True))
        return model, report, transform
    samples = cohort.references(names, transform)
    validation = cohort.references(validation_names, transform)
    by_group = {}
    for sample in samples:
        by_group.setdefault(sample.rna.group, []).append(sample)
    if set(by_group) & {sample.rna.group for sample in validation}:
        raise ValueError("DGAT validation group overlaps training")
    graph_cache = output / "graphs"
    # Graphs/data stay on CPU until the selected slice is used.
    batches = {s.rna.name: batch(s.rna, graph_cache, s.target) for s in samples}
    optimizers = model.optimizers()
    rng = np.random.default_rng(args.seed)
    start_epoch, best, chosen, stale, history, best_state = 1, float("inf"), None, 0, [], None
    resume = output / "resume.pt"
    if resume.exists():
        saved = torch.load(resume, map_location=args.device, weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("DGAT resume identity differs")
        model.load_state_dict(saved["state"])
        for optimizer, values in zip(optimizers, saved["optimizers"]):
            optimizer.load_state_dict(values)
        history, best, chosen, stale, best_state = (saved[k] for k in ("history", "best", "chosen", "stale", "best_state"))
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if args.device.startswith("cuda") and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        start_epoch = saved["epoch"] + 1
    before = history[-1]["seconds"] if history else 0.
    started = time.monotonic()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, budget + 1):
        if validation and stale >= 6:
            break
        model.train()
        losses, term_values = [], []
        for group in rng.permutation(sorted(by_group)):
            sample = by_group[group][int(rng.integers(len(by_group[group])))]
            values = {k: v.to(args.device) for k, v in batches[sample.rna.name].items()}
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss, terms = model.objective(values)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite DGAT objective")
            loss.backward()
            model.clip()
            for optimizer in optimizers:
                optimizer.step()
            losses.append(float(loss.detach()))
            term_values.append({k: float(v.detach()) for k, v in terms.items()})
        row = {"epoch": epoch, "training_loss": float(np.mean(losses)),
               "training_terms": {k: float(np.mean([v[k] for v in term_values])) for k in term_values[0]},
               "seconds": before + time.monotonic() - started}
        if validation and (epoch == 1 or epoch % 4 == 0 or epoch == budget):
            by_slice = {s.rna.name: float(np.mean((predict(model, s.rna, graph_cache, args.device) - s.target).astype(np.float64)**2))
                        for s in validation}
            score = float(np.mean([np.mean([by_slice[s.rna.name] for s in validation if s.rna.group == group])
                                   for group in sorted({s.rna.group for s in validation})]))
            row.update(validation_mse=score, validation_by_slice=by_slice)
            if score < best:
                best, chosen, stale = score, epoch, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                stale += 1
        if epoch % 10 == 0:
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] *= .8
        history.append(row)
        if epoch == 1 or epoch % 4 == 0 or epoch == budget:
            save_torch(resume, {"identity": identity, "state": model.state_dict(),
                       "optimizers": [o.state_dict() for o in optimizers], "history": history,
                       "best": best, "chosen": chosen, "stale": stale, "best_state": best_state,
                       "epoch": epoch, "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else None})
            write_json(output / "history.json", history)
            print(f"DGAT epoch={epoch} loss={row['training_loss']:.5f} val={row.get('validation_mse')}", flush=True)
    if validation:
        model.load_state_dict(best_state)
    else:
        chosen = history[-1]["epoch"]
    model.eval()
    save_torch(output / "model.pt", model.state_dict())
    report = {"identity": identity, "training": names, "validation": validation_names,
              "budget": budget, "selected_epoch": chosen, "history": history,
              "parameters": sum(p.numel() for p in model.parameters()), "architecture": model.architecture,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
              "model_sha256": sha256(output / "model.pt"), "query_labels_for_selection": False,
              "single_device_protein_latent_detached": False}
    write_json(output / "training.json", report)
    return model, report, transform
