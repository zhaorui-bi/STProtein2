"""Residual-stage selection must not select warm-up or lose state on resume."""
from dataclasses import replace
import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge
import torch

from stprotein2.config import ModelConfig
from stprotein2.data import RNAView, TrainingSlice
from stprotein2.graph import FrozenRouter
from stprotein2.training.config import AnchorTrainConfig
import research.anchor.trainer as trainer


def test_checkpoint_policy_rejects_unreachable_residual_stage():
    cfg = AnchorTrainConfig(epochs=8, warmup_epochs=8, checkpoint_stage='any')
    assert cfg.first_checkpoint_epoch == 1
    with pytest.raises(ValueError, match='cannot reach'):
        replace(cfg, checkpoint_stage='after_ramp')
    short = AnchorTrainConfig(epochs=2, warmup_epochs=1, prototype_ramp_epochs=1, checkpoint_stage='after_ramp')
    assert short.first_checkpoint_epoch == 2
    assert replace(short, warmup_epochs=0, prototype_ramp_epochs=0).first_checkpoint_epoch == 1
    with pytest.raises(ValueError, match='Unknown'):
        replace(short, checkpoint_stage='unregistered')


def test_stage_selection_and_resume_even_when_warmup_scores_best(tmp_path, monkeypatch):
    rng = np.random.default_rng(71)
    paired = []
    for i in range(3):
        x = rng.normal(size=(12, 6)).astype(np.float32)
        view = RNAView(f's{i}', f'g{i}', x, np.ones(6, dtype=np.float32), np.zeros((12,4), dtype=np.float32),
                       np.array([(j//4,j%4) for j in range(12)], dtype=float),
                       np.array([f's{i}_{j}' for j in range(12)]), tuple(f'g{j}' for j in range(6)))
        paired.append(TrainingSlice(view, rng.normal(size=(12,31)).astype(np.float32)))
    anchor = Ridge(alpha=1.).fit(np.concatenate([s.rna.x for s in paired[:2]]),
                                np.concatenate([s.target for s in paired[:2]]))
    router = FrozenRouter(components=3, clusters=3).fit([s.rna.x for s in paired[:2]], ['g0','g1'])
    model_cfg = ModelConfig(n_genes=6, width=12, hidden=8, members=2, dropout=0,
                state_prototypes=3, niche_prototypes=3, prototype_dim=8, local_operator='none')
    train_cfg = AnchorTrainConfig(epochs=3, warmup_epochs=1, prototype_ramp_epochs=1,
                    checkpoint_stage='after_ramp', validate_every=1, threads=2, mask_rates=(0.,), execution_mode='full')
    real_predict = trainer.predict
    calls = []

    def ordered_scores(*args, **kwargs):
        arrays, diagnostics = real_predict(*args, **kwargs)
        arrays['prediction'] = paired[2].target + len(calls)
        calls.append(1)
        return arrays, diagnostics

    monkeypatch.setattr(trainer, 'predict', ordered_scores)
    uninterrupted, report = trainer.fit(paired[:2], model_cfg, train_cfg, router, tmp_path/'continuous',
                                       anchor, validation=paired[2])
    assert report['selected_epoch'] == 2
    assert report['history'][0]['reference_validation_mse'] == 0
    assert report['history'][0]['checkpoint_eligible'] is False
    assert report['history'][1]['checkpoint_eligible'] is True
    assert report['selected_corrections_trained'] is True
    assert report['selected_teacher_updates'] == 2
    real_save = trainer.save_torch
    calls.clear()

    def interrupt_after_warmup(path, value):
        real_save(path, value)
        if path.name == 'resume.pt' and value['epoch'] == 1:
            raise RuntimeError('injected interruption after atomic warm-up checkpoint')

    monkeypatch.setattr(trainer, 'save_torch', interrupt_after_warmup)
    with pytest.raises(RuntimeError, match='injected interruption'):
        trainer.fit(paired[:2], model_cfg, train_cfg, router, tmp_path/'resumed', anchor, validation=paired[2])
    assert not (tmp_path/'resumed/best.pt').exists()
    monkeypatch.setattr(trainer, 'save_torch', real_save)
    resumed, again = trainer.fit(paired[:2], model_cfg, train_cfg, router, tmp_path/'resumed', anchor, validation=paired[2])
    assert again['selected_epoch'] == 2
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    saved = json.loads((tmp_path/'resumed/training.json').read_text())
    assert saved['first_checkpoint_epoch'] == 2
