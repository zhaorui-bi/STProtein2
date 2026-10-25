"""Reference-only C0 training for the two compact ordinary residuals."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .graph import digest_array, physical_graph
from .io import freeze_run, sha256, verify_marker, write_json
from .neural import CompactResidual, EnsembleLinear, RNAEnsemble, center_local, neighbor_mean
from .smoothing import apply, spatial_residual


def execution_record(device):
    """Freeze arithmetic-affecting runtime settings, including CPU/CUDA choice."""
    requested = torch.device(device)
    if requested.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable')
        index = torch.cuda.current_device() if requested.index is None else requested.index
        props = torch.cuda.get_device_properties(index)
        actual = f'cuda:{index}'
        hardware = {'model': props.name, 'compute_capability': [props.major, props.minor],
                    'total_memory': props.total_memory}
    elif requested.type == 'cpu':
        actual = 'cpu'
        hardware = {'architecture': platform.machine(), 'processor': platform.processor()}
    else:
        raise ValueError('The validated compact trainer supports cpu and cuda')
    return {'device': actual, 'hardware': hardware, 'torch': str(torch.__version__),
            'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
            'default_dtype': str(torch.get_default_dtype()),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 80
    warmup_epochs: int = 8
    checkpoint_delay: int = 8
    validate_every: int = 4
    patience: int = 6
    seed: int = 2025
    threads: int = 2
    lr: float = .001
    optimizer_epsilon: float = 1e-8
    weight_decay: float = 1e-4
    clip_grad: float = 1.
    auxiliary_weight: float = .2
    checkpoint_representations: bool = True

    def __post_init__(self):
        for key in ('epochs', 'validate_every', 'patience', 'threads', 'seed', 'warmup_epochs', 'checkpoint_delay'):
            value = getattr(self, key)
            minimum = 0 if key in {'seed', 'warmup_epochs', 'checkpoint_delay'} else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError('Invalid training integer: ' + key)
        if self.seed >= 2**32 or self.epochs < self.first_checkpoint_epoch:
            raise ValueError('Seed or after-warmup checkpoint budget is invalid')
        for key in ('lr', 'optimizer_epsilon', 'weight_decay', 'clip_grad', 'auxiliary_weight'):
            value = getattr(self, key)
            if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
                    or (value < 0 if key == 'weight_decay' else value <= 0)):
                raise ValueError('Invalid training scalar: ' + key)
        if not isinstance(self.checkpoint_representations, bool):
            raise ValueError('Representation checkpointing must be boolean')

    @property
    def first_checkpoint_epoch(self):
        return self.warmup_epochs + max(1, self.checkpoint_delay)


class TrainingStudent(CompactResidual):
    """Auxiliary member supervision is used during fitting and removed on export."""
    initialization = 'ordinary_research_constructor_31x16_draw_v1'

    def __init__(self, cfg, checkpoint_representations=True):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.encoder = RNAEnsemble(cfg)
        self.direct = EnsembleLinear(cfg.hidden, cfg.n_proteins, cfg.members, cfg.weight_modulated_input)
        # The old ordinary model initialized, then deleted, a 31 x 16 embedding.
        # Consume that draw to preserve the tested initialization of the heads.
        # It is neither a parameter nor a deployed tensor in this implementation.
        torch.randn(cfg.n_proteins, 16)
        self._initialize_residual()
        self.checkpoint_representations = checkpoint_representations

    def forward(self, x, availability, qc, edge_index=None, degree=None, enabled=True):
        self._check_inputs(x, availability, qc)
        def encode(a, b, c):
            members = self.encoder(a, b, c)
            return members, members.mean(1)
        if self.training and self.checkpoint_representations and torch.is_grad_enabled():
            members, h = checkpoint(encode, x, availability, qc, use_reentrant=False, preserve_rng_state=True)
        else:
            members, h = encode(x, availability, qc)
        auxiliary = self.direct(members)
        result = self._components(h, qc, edge_index, degree, enabled)
        result['auxiliary_members'] = auxiliary
        return result

    def residual(self):
        result = CompactResidual(self.cfg)
        result.load_state_dict({n: v.detach().cpu() for n, v in self.state_dict().items() if not n.startswith('direct.')})
        return result.eval()


def tensor_input(view, graph, device):
    return (torch.as_tensor(view.x, device=device), torch.as_tensor(view.availability, device=device),
            torch.as_tensor(view.qc, device=device), torch.as_tensor(graph.edge_index, device=device, dtype=torch.long),
            torch.as_tensor(graph.degree, device=device, dtype=torch.float32))


def anchor_tensor(anchor, view, device='cpu'):
    """Historical FP32 training/validation arithmetic; deployment uses FP64."""
    x = torch.as_tensor(view.x, device=device)
    available = torch.as_tensor(view.availability, device=device)
    ridge = anchor['model']
    return F.linear(torch.where(available > 0, x, 0),
                    torch.as_tensor(ridge.coef_, dtype=torch.float32, device=device),
                    torch.as_tensor(ridge.intercept_, dtype=torch.float32, device=device))


def prediction_components(model, anchor, view, graph, device='cpu'):
    """FP32 anchor components used for nested epoch/gate selection only."""
    model.eval()
    with torch.inference_mode():
        args = tensor_input(view, graph, device)
        components = model(*args)
        raw = anchor_tensor(anchor, view, device)
        value = raw.double()
        delta = center_local(neighbor_mean(value, args[3], args[4]) - value, args[4])
        base = (value + .5 * delta).to(raw.dtype)
        prediction = (base + model.gamma_state * components['state_delta']
                      + model.gamma_local * components['local_delta'])
        return {**{k: components[k].cpu().numpy() for k in ('state_delta', 'local_delta', 'local_raw')},
                'prediction': prediction.cpu().numpy(), 'base': base.cpu().numpy(),
                'rna_base': raw.cpu().numpy(), 'spatial_basis_delta': delta.cpu().numpy()}


def validation_prediction(model, anchor, view, graph, device='cpu'):
    return prediction_components(model, anchor, view, graph, device)['prediction']


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary); temporary.replace(path)


def identity(sample):
    view = sample.rna
    return {'name': view.name, 'group': view.group, 'RNA': view.fingerprint(),
            'qc': digest_array(view.qc), 'barcodes': list(view.barcodes), 'target': digest_array(sample.target)}


def fit(references, validation, anchor, model_config, config, output, device='cpu'):
    """Fit one allowed subset; validation sources are excluded from all fitting."""
    names = [s.rna.name for s in references]
    held = [s.rna.name for s in validation]
    groups = {s.rna.group for s in references}
    if (not names or len(names) != len(set(names)) or len(groups) < 2 or len(held) != len(set(held))
            or set(names) & set(held) or groups & {s.rna.group for s in validation}):
        raise ValueError('Unique reference slices and disjoint validation sources are required')
    if (anchor['reference_names'] != names or anchor['transform'].fitted_names_ != names
            or any(s.rna.genes != tuple(anchor['transform'].genes_) for s in references + validation)
            or any(s.count_views for s in references + validation)
            or not anchor.get('independently_fitted') or anchor.get('residual_training_seen') is not False):
        raise ValueError('Anchor, gene identities or unaugmented C0 inputs differ')
    if torch.get_default_dtype() != torch.float32:
        raise ValueError('The registered training arithmetic requires default FP32')
    torch.set_num_threads(config.threads)
    output = Path(output)
    graphs = {s.rna.name: physical_graph(s.rna.coordinates) for s in references + validation}
    baselines = {}
    with torch.inference_mode():
        for s in references:
            raw = anchor_tensor(anchor, s.rna).numpy()
            baselines[s.rna.name] = apply(raw, spatial_residual(raw, graphs[s.rna.name]), .5)
    freeze_run(output, {'stage': 'compact_C0_reference_fit_v1', 'model': asdict(model_config),
               'train': asdict(config), 'initialization': TrainingStudent.initialization,
               'reference_identity': [identity(s) for s in references],
               'validation_identity': [identity(s) for s in validation],
               'anchor': {k: digest_array(np.asarray(getattr(anchor['model'], field), dtype=np.float32))
                          for k, field in [('weight', 'coef_'), ('bias', 'intercept_')]},
               'baseline_sha256': {n: digest_array(v) for n, v in baselines.items()},
               'training_baseline': 'FP32_CPU_anchor_then_FP64_numpy_spatial',
               'validation_baseline': 'FP32_device_anchor_then_FP64_torch_spatial',
               'query_protein_used': False, 'execution': execution_record(device)})
    marker = output / 'fit_complete.json'
    if marker.exists():
        verify_marker(output, marker.name)
        saved = torch.load(output / 'residual.pt', weights_only=True, map_location='cpu')
        model = CompactResidual(model_config); model.load_state_dict(saved['state_dict'])
        return model.eval().to(device), json.loads((output / 'training.json').read_text())
    torch.set_num_threads(config.threads)
    np.random.seed(config.seed); torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed)
    model = TrainingStudent(model_config, config.checkpoint_representations).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, eps=config.optimizer_epsilon,
                                 weight_decay=config.weight_decay)
    by_group = {}
    for s in references:
        by_group.setdefault(s.rna.group, []).append(s)
    history, start, best, best_epoch, stale = [], 1, float('inf'), None, 0
    resume = output / 'resume.pt'
    if resume.exists():
        saved = torch.load(resume, weights_only=True, map_location=device)
        if saved['manifest_sha256'] != sha256(output / 'manifest.json'):
            raise ValueError('Resume identity changed')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        expected = (config.lr, config.optimizer_epsilon, config.weight_decay, (.9, .999))
        if any((g['lr'], g['eps'], g['weight_decay'], tuple(g['betas'])) != expected for g in optimizer.param_groups):
            raise ValueError('Restored optimizer configuration differs')
        history, best, best_epoch, stale = saved['history'], saved['best'], saved['best_epoch'], saved['stale']
        start = saved['epoch'] + 1
        rng.bit_generator.state = saved['rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.startswith('cuda') and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    started = time.monotonic()
    elapsed = history[-1]['elapsed_seconds'] if history else 0.
    epochs = range(start, config.epochs + 1) if not validation or stale < config.patience else ()
    for epoch in epochs:
        model.train()
        totals = {'loss': 0., 'protein': 0., 'auxiliary': 0.}
        source_names = list(by_group)
        selected_names = []
        for index in rng.permutation(len(source_names)):
            samples = by_group[source_names[index]]
            sample = samples[int(rng.integers(len(samples)))]
            # Preserve the no-augmentation branch's historical RNG sequence.
            rng.choice((0.,))
            selected_names.append(sample.rna.name)
            target = torch.as_tensor(sample.target, device=device)
            baseline = torch.as_tensor(baselines[sample.rna.name], device=device)
            optimizer.zero_grad(set_to_none=True)
            result = model(*tensor_input(sample.rna, graphs[sample.rna.name], device), enabled=epoch > config.warmup_epochs)
            protein = F.mse_loss(baseline + result['correction'], target)
            auxiliary = F.mse_loss(result['auxiliary_members'], target[:, None, :].expand_as(result['auxiliary_members']))
            loss = protein + config.auxiliary_weight * auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite compact training loss')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad, error_if_nonfinite=True)
            optimizer.step()
            for k, value in [('loss', loss), ('protein', protein), ('auxiliary', auxiliary)]:
                totals[k] += float(value.detach())
        row = {'epoch': epoch, **{k: v / len(by_group) for k, v in totals.items()},
               'optimizer_updates': epoch * len(by_group), 'sampled_slices': selected_names,
               'last_gradient_norm': float(norm), 'elapsed_seconds': elapsed + time.monotonic() - started}
        checkpoint_epoch = epoch == 1 or epoch % config.validate_every == 0 or epoch == config.epochs
        if validation and checkpoint_epoch:
            scores = {s.rna.name: float(np.mean((validation_prediction(model, anchor, s.rna, graphs[s.rna.name], device)
                                               - s.target) ** 2, dtype=np.float64)) for s in validation}
            score = float(np.mean([np.mean([scores[s.rna.name] for s in validation if s.rna.group == group])
                                   for group in sorted({s.rna.group for s in validation})]))
            row.update(reference_validation_by_slice=scores, reference_validation_mse=score,
                       checkpoint_eligible=epoch >= config.first_checkpoint_epoch)
            if epoch >= config.first_checkpoint_epoch:
                if score < best:
                    best, best_epoch, stale = score, epoch, 0
                    save_torch(output / 'best.pt', {'state_dict': model.state_dict(), 'epoch': epoch})
                else:
                    stale += 1
        history.append(row)
        if checkpoint_epoch:
            save_torch(resume, {'manifest_sha256': sha256(output / 'manifest.json'), 'model': model.state_dict(),
                       'optimizer': optimizer.state_dict(), 'history': history, 'epoch': epoch,
                       'best': best, 'best_epoch': best_epoch, 'stale': stale, 'rng': rng.bit_generator.state,
                       'torch_rng': torch.get_rng_state(),
                       'cuda_rng': torch.cuda.get_rng_state_all() if device.startswith('cuda') else None})
            write_json(output / 'history.json', history)
        if validation and stale >= config.patience:
            break
    if validation:
        model.load_state_dict(torch.load(output / 'best.pt', weights_only=True, map_location=device)['state_dict'])
    else:
        best_epoch = history[-1]['epoch']
    report = {'selected_epoch': best_epoch, 'reference_validation_mse': best if validation else None,
              'reference_names': names, 'validation_names': held, 'reference_groups': sorted(groups),
              'auxiliary_target': 'true_Y', 'auxiliary_direct_head_deployed': False,
              'initialization': TrainingStudent.initialization, 'first_checkpoint_epoch': config.first_checkpoint_epoch,
              'optimizer': {'class': 'AdamW', 'lr': config.lr, 'eps': config.optimizer_epsilon,
                            'weight_decay': config.weight_decay, 'betas': [.9, .999]},
              'history': history, 'query_protein_used': False}
    residual = model.residual()
    save_torch(output / 'residual.pt', {'schema': 'compact_residual_weights_v1',
               'config': asdict(model_config), 'state_dict': residual.state_dict()})
    write_json(output / 'training.json', report)
    files = ['residual.pt', 'training.json', 'resume.pt'] + (['best.pt'] if validation else [])
    write_json(marker, {'manifest_sha256': sha256(output / 'manifest.json'),
               'artifacts': {n: sha256(output / n) for n in files}})
    return residual.to(device), report
