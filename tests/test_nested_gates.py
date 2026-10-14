import argparse
import json

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from stprotein2.data import PANEL
from stprotein2.gates import select_gates
from stprotein2.nested import run_nested
from stprotein2.cli import score_run


def test_gate_requires_replication_and_rejects_majority_harm():
    # Four proteins: both wins, one-group windfall fails replication, exact
    # state/both tie selects simpler state, and zero baseline is undefined.
    losses = np.ones((4, 4, 4))
    losses[:, 1, 0] = .9; losses[:, 2, 0] = .95; losses[:, 3, 0] = .8
    losses[:, 1:, 1] = 1.1; losses[0, 1:, 1] = .1
    losses[:, 1, 2] = .8; losses[:, 3, 2] = .8
    losses[:, :, 3] = 0
    groups = {str(i): losses[i] for i in range(4)}
    selected = select_gates(groups, ['a', 'b', 'c', 'd'])
    assert selected['gamma_state'] == [1, 0, 1, 0]
    assert selected['gamma_local'] == [1, 0, 0, 0]
    insufficient = select_gates({k: v for k, v in groups.items() if k != '0'}, ['a', 'b', 'c', 'd'])
    assert insufficient['gamma_state'] == [0]*4
    assert insufficient['gamma_local'] == [0]*4


@pytest.mark.parametrize('extra_reference_group', [False, True])
def test_nested_multi_slice_group_and_separate_scoring(tmp_path, extra_reference_group):
    rng = np.random.default_rng(3)
    entries, groups = [], {}
    sample_groups = ['a', 'b', 'c', 'c', 'query'] + (['d'] if extra_reference_group else [])
    for i, group in enumerate(sample_groups):
        name = f'slice{i}'
        rna = ad.AnnData(sparse.csr_matrix(rng.poisson(3, (12, 8)).astype(np.float32)))
        rna.obs_names = [f'{name}_b{j}' for j in range(12)]
        rna.var_names = [f'g{j}' for j in range(8)]
        rna.obsm['spatial'] = np.array([(j//4, j%4) for j in range(12)], dtype=float)
        adt = ad.AnnData(sparse.csr_matrix(rng.poisson(2, (12, 31)).astype(np.float32)))
        adt.obs_names = rna.obs_names; adt.var_names = PANEL
        rp, ap = tmp_path/f'{name}_rna.h5ad', tmp_path/f'{name}_adt.h5ad'
        rna.write_h5ad(rp); adt.write_h5ad(ap)
        entries.append({'name': name, 'rna': str(rp), 'adt': str(ap)}); groups[name] = group
    registry, mapping, config = (tmp_path/s for s in ('registry.json', 'groups.json', 'config.json'))
    registry.write_text(json.dumps(entries)); mapping.write_text(json.dumps({'groups': groups}))
    config.write_text(json.dumps({'model': {'n_genes': 6, 'width': 12, 'hidden': 8, 'members': 2,
                      'use_state': False, 'use_niche': False, 'local_operator': 'none'},
                      'train': {'epochs': 2, 'warmup_epochs': 1, 'validate_every': 1, 'threads': 2,
                                'core_size': 6, 'min_core': 3, 'max_core': 8}}))
    args = argparse.Namespace(registry=str(registry), groups=str(mapping), config=str(config),
                              query='slice4', output=str(tmp_path/'run'), device='cpu')
    run_nested(args)
    output = tmp_path/'run'
    complete = json.loads((output/'prediction_complete.json').read_text())
    assert 'slice4' not in complete['protein_reads_before_prediction']
    assert complete['query_protein_read_for_prediction'] is False
    assert not (output/'development_scores.json').exists()
    gates = json.loads((output/'gate_selection.json').read_text())
    assert sum(gates['gamma_local']) + sum(gates['gamma_state']) == 0
    reports = [json.loads(p.read_text()) for p in output.glob('inner_*/training.json')]
    assert any(r['inner_validation'] == ['slice2', 'slice3'] for r in reports)
    for r in reports:
        assert not set(r['reference_groups']) & set(r['inner_validation_groups'])
        assert 'slice4' not in r['reference_names'] + r['inner_validation']
    if extra_reference_group:
        evidence = list(output.glob('gate_oof_*/component_losses.json'))
        assert len(evidence) == 4
        for path in evidence:
            record = json.loads(path.read_text())
            held_group = record['held_group']
            assert all(groups[n] != held_group and n != 'slice4' for n in record['allowed_names'])
            for training in path.parent.glob('inner_*/training.json'):
                fold = json.loads(training.read_text())
                assert held_group not in fold['reference_groups'] + fold['inner_validation_groups']
                assert 'slice4' not in fold['reference_names'] + fold['inner_validation']
    before = np.load(output/'predictions/slice4/prediction.npz')['prediction']
    run_nested(args)
    after = np.load(output/'predictions/slice4/prediction.npz')['prediction']
    np.testing.assert_array_equal(before, after)
    score_run(args)
    assert (output/'development_scores.json').exists()
    config.write_text(config.read_text().replace('"epochs": 2', '"epochs": 3'))
    with pytest.raises(RuntimeError, match='changed'):
        run_nested(args)
