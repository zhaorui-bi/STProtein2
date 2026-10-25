"""Selection cannot use partial coverage or pooled slices to choose a default."""
import copy
import importlib.util
from pathlib import Path
import statistics

import numpy as np
import pytest

from stprotein2.data import PANEL

PATH = Path(__file__).resolve().parents[1] / 'scripts/freeze_compact_base.py'
spec = importlib.util.spec_from_file_location('compact_base_freezer', PATH)
freezer = importlib.util.module_from_spec(spec); spec.loader.exec_module(freezer)


def fixture(baseline=1.2):
    mapping = {**{f'a{i}': 'a' for i in range(4)}, **{f'b{i}': 'b' for i in range(2)},
               'c': 'c', 'd': 'd', 'e': 'e', 'f': 'f'}
    def metric(value):
        return {'summary': {'rmse': value}, 'per_protein': [{'protein': p, 'rmse': value} for p in PANEL]}
    rows = {}
    for head, change in [('own_mlp', 0.), ('neighbor_mlp', -.0005)]:
        rows[head] = []
        for seed, jitter in [(2025, -.03), (2026, 0), (2027, .03)]:
            for query, source in mapping.items():
                value = .9 + .1 * 'abcdef'.index(source) + jitter + change
                rows[head].append({'seed': seed, 'query': query, 'source': source,
                                   'metrics': metric(value), 'base_metrics': metric(baseline)})
    summaries = {head: freezer.summarize(values, mapping) for head, values in rows.items()}
    summaries['B0'] = freezer.summarize(rows['own_mlp'], mapping, base=True)
    for summary in summaries.values():
        summary.update(scored_query_tissues=30, required_query_tissues=30)
    error = statistics.stdev([r['source_macro_RMSE'] for r in summaries['neighbor_mlp']['per_seed']]) / np.sqrt(3)
    cutoff = summaries['neighbor_mlp']['source_macro_RMSE'] + error
    eligible = [n for n in ('B0', 'own_mlp', 'neighbor_mlp') if summaries[n]['source_macro_RMSE'] <= cutoff + 1e-12]
    decision = {'strict_lowest_mean': 'neighbor_mlp', 'compact_default': eligible[0],
                'eligible_in_complexity_order': eligible, 'lowest_mean_optimizer_SE': error,
                'compact_cutoff_RMSE': cutoff, 'donor_confidence_interval': False}
    result = {'all_cases_verified': True, 'all_actual_pair_inputs_matched': True, 'registered_new_tasks': 24,
              'reused_seed2025_tasks': 12, 'sources': 6, 'query_slices': 10, 'development_only': True,
              'external_confirmation': False, 'rows': rows, 'summaries': summaries, 'decision': decision}
    return result, copy.deepcopy(decision), mapping


def test_equal_source_selection_keeps_strict_winner_and_negative_sources():
    result, decision, mapping = fixture()
    summaries, chosen = freezer.validate_result(result, decision, mapping)
    assert summaries['own_mlp']['source_macro_RMSE'] == pytest.approx(1.15)
    assert chosen['strict_lowest_mean'] == 'neighbor_mlp'
    assert chosen['compact_default'] == 'own_mlp'
    assert summaries['own_mlp']['per_source']['f'] > summaries['B0']['per_source']['f']


def test_b0_is_preferred_when_it_qualifies_for_the_registered_cutoff():
    result, decision, mapping = fixture(baseline=1.16)
    _, chosen = freezer.validate_result(result, decision, mapping)
    assert chosen['compact_default'] == 'B0'
    assert chosen['strict_lowest_mean'] == 'neighbor_mlp'


def test_missing_query_seed_prevents_any_selection():
    result, decision, mapping = fixture()
    result['rows']['neighbor_mlp'].pop()
    with pytest.raises(ValueError, match='All ten queries'):
        freezer.validate_result(result, decision, mapping)


def test_claimed_default_cannot_replace_the_registered_math():
    result, decision, mapping = fixture()
    decision['compact_default'] = 'B0'
    with pytest.raises(ValueError, match='registered compact decision differs'):
        freezer.validate_result(result, decision, mapping)
