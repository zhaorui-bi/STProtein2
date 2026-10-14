"""Independent-anchor identity, genuine fallback and nested-label isolation."""
import argparse
import json

import anndata as ad
import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge
import torch

from stprotein2.config import ModelConfig
from stprotein2.data import Cohort, PANEL
from stprotein2.graph import physical_graph
from stprotein2.cli import score_run
from stprotein2_anchor.fitting import fit_anchor, load_frozen_anchor
from stprotein2_anchor.model import AnchoredSTProtein2
from stprotein2_anchor.nested import load_model, run_nested


def tiny_cohort(tmp_path, extra_group=False):
    rng = np.random.default_rng(31)
    entries, groups = [], {}
    for i, group in enumerate(['a', 'b', 'c', 'c', 'query'] + (['d'] if extra_group else [])):
        name = f's{i}'
        rna = ad.AnnData(sparse.csr_matrix(rng.poisson(3, (12, 8)).astype(np.float32)))
        rna.obs_names = [f'{name}_{j}' for j in range(12)]
        rna.var_names = [f'g{j}' for j in range(8)]
        rna.obsm['spatial'] = np.array([(j//4, j%4) for j in range(12)], dtype=float)
        adt = ad.AnnData(sparse.csr_matrix(rng.poisson(3, (12, 31)).astype(np.float32)))
        adt.obs_names = rna.obs_names; adt.var_names = PANEL
        rp, ap = tmp_path/f'{name}_rna.h5ad', tmp_path/f'{name}_adt.h5ad'
        rna.write_h5ad(rp); adt.write_h5ad(ap)
        entries.append({'name': name, 'rna': str(rp), 'adt': str(ap)})
        groups[name] = group
    registry = tmp_path/'registry.json'; registry.write_text(json.dumps(entries))
    return registry, groups


def test_anchor_exact_fallback_and_training_preserves_coefficients():
    torch.set_num_threads(2); torch.manual_seed(9)
    rng = np.random.default_rng(9)
    x = rng.normal(size=(20, 6)).astype(np.float32)
    y = rng.normal(size=(20, 31)).astype(np.float32)
    ridge = Ridge(alpha=1).fit(x, y)
    cfg = ModelConfig(n_genes=6, width=12, hidden=8, members=2,
                      state_prototypes=3, niche_prototypes=3, prototype_dim=8,
                      experts=2, stalk=2, channels=2, protein_embedding=3, dropout=0)
    model = AnchoredSTProtein2(cfg)
    model.anchor.set_coefficients(ridge.coef_, ridge.intercept_)
    for head in (model.state_readout[-1], model.niche_readout[-1], model.expert_readout):
        torch.nn.init.normal_(head.weight, std=.15)
    graph = physical_graph(np.array([(i//5, i%5) for i in range(20)], dtype=float)).torch()
    tx, available, qc = torch.tensor(x), torch.ones(6), torch.zeros(20, 4)
    expected = model.anchor(tx, available)
    np.testing.assert_allclose(expected.detach().numpy(), ridge.predict(x), atol=1e-6, rtol=1e-5)
    model.set_gates(np.zeros(31), np.zeros(31))
    result = model(tx, available, qc, graph)
    assert result['state_delta'].abs().max() > 0
    assert result['local_delta'].abs().max() > 0
    assert torch.equal(result['prediction'], expected)
    assert not torch.equal(result['auxiliary'], expected)
    available[0] = 0
    changed = tx.clone(); changed[:, 0] = 1e8
    assert torch.equal(model.anchor(tx, available), model.anchor(changed, available))
    assert not list(model.anchor.parameters())
    before = {k: v.clone() for k, v in model.anchor.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    model.set_gates(np.ones(31), np.ones(31))
    result = model(tx, torch.ones(6), qc, graph)
    loss = (result['prediction']-torch.tensor(y)).square().mean()
    loss += .2*(result['auxiliary_members']-torch.tensor(y)[:, None]).square().mean()
    loss.backward()
    assert model.direct.weight.grad is not None
    optimizer.step()
    assert all(torch.equal(before[k], v) for k, v in model.anchor.state_dict().items())
    model.set_gates(np.zeros(31), np.zeros(31))
    assert torch.equal(model(tx, torch.ones(6), qc, graph)['prediction'], expected)


def test_anchor_cv_refits_preprocessing_and_never_reads_disallowed_labels(tmp_path):
    registry, groups = tiny_cohort(tmp_path)
    cohort = Cohort(registry, groups)
    names = ['s0', 's1', 's2', 's3']
    original = cohort.measured_protein

    def guarded(name):
        assert name in names
        return original(name)

    cohort.measured_protein = guarded
    bundle, path = fit_anchor(cohort, names, 6, tmp_path/'cache')
    assert bundle['reference_names'] == names
    assert bundle['transform'].fitted_names_ == names
    for record in path.glob('inner_*.json'):
        fold = json.loads(record.read_text())
        assert fold['preprocessing_fitted_names'] == fold['training']
        assert not set(fold['training']) & set(fold['validation'])
        assert not {groups[n] for n in fold['training']} & {groups[n] for n in fold['validation']}
        assert 's4' not in fold['training'] + fold['validation']
    again, _ = fit_anchor(cohort, names, 6, tmp_path/'cache')
    np.testing.assert_array_equal(again['model'].coef_, bundle['model'].coef_)
    reused, _ = load_frozen_anchor(cohort, names, 6, tmp_path/'cache')
    np.testing.assert_array_equal(reused['model'].coef_, bundle['model'].coef_)
    changed = ad.read_h5ad(cohort.specs['s0']['adt'])
    changed.X.data[0] += 1; changed.write_h5ad(cohort.specs['s0']['adt'])
    with pytest.raises(RuntimeError, match='changed'):
        fit_anchor(cohort, names, 6, tmp_path/'cache')
    with pytest.raises(ValueError, match='input identity changed'):
        load_frozen_anchor(cohort, names, 6, tmp_path/'cache')


@pytest.mark.parametrize('extra_group', [False, True])
@pytest.mark.parametrize('checkpoint_stage', ['any', 'after_ramp'])
def test_nested_anchor_and_gate_exclusion_resume_and_score(tmp_path, extra_group, checkpoint_stage):
    registry, groups = tiny_cohort(tmp_path, extra_group)
    mapping = tmp_path/'groups.json'; mapping.write_text(json.dumps({'groups': groups}))
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'model': {'n_genes': 6, 'width': 12, 'hidden': 8, 'members': 2,
             'state_prototypes': 3, 'niche_prototypes': 3, 'prototype_dim': 8,
             'local_operator': 'none', 'dropout': 0},
             'train': {'epochs': 2, 'warmup_epochs': 1, 'prototype_ramp_epochs': 1,
                       'checkpoint_stage': checkpoint_stage,
                       'validate_every': 1, 'threads': 2, 'execution_mode': 'full'}}))
    output = tmp_path/'run'
    args = argparse.Namespace(registry=str(registry), groups=str(mapping), config=str(config),
              query='s4', output=str(output), anchor_cache=str(tmp_path/'anchors'), device='cpu')
    run_nested(args, prepare_only=True)
    prepared = json.loads((output/'anchors_prepared.json').read_text())
    assert prepared['query_protein_read'] is False
    run_nested(args)
    complete = json.loads((output/'prediction_complete.json').read_text())
    assert 's4' not in complete['protein_reads_before_prediction']
    assert not (output/'development_scores.json').exists()
    for record in output.rglob('training.json'):
        report = json.loads(record.read_text())
        assert report['anchor_unchanged'] is True
        if checkpoint_stage == 'after_ramp':
            assert report['selected_epoch'] == 2
            assert report['selected_corrections_trained'] is True
            assert report['selected_teacher_updates'] > 0
        assert 's4' not in report['reference_names'] + report['inner_validation']
        anchor = json.loads((record.parent/'anchor.json').read_text())
        assert anchor['reference_names'] == report['reference_names']
    if extra_group:
        evidence = list(output.glob('gate_oof_*/component_losses.json'))
        assert len(evidence) == 4
        for record in evidence:
            held = json.loads(record.read_text())['held_group']
            for anchor_record in record.parent.rglob('anchor.json'):
                anchor = json.loads(anchor_record.read_text())
                assert held not in {groups[n] for n in anchor['reference_names']}
    model = load_model(output/'model.pt', 'cpu')
    before = np.load(output/'predictions/s4/prediction.npz')['prediction'].copy()
    if not extra_group:
        arrays = np.load(output/'predictions/s4/prediction.npz')
        np.testing.assert_array_equal(before, arrays['base'])
        assert torch.count_nonzero(model.gamma_state) == torch.count_nonzero(model.gamma_local) == 0
    run_nested(args)
    np.testing.assert_array_equal(before, np.load(output/'predictions/s4/prediction.npz')['prediction'])
    score_run(args)
    assert (output/'development_scores.json').exists()
