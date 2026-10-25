"""Freeze the compact development default only after complete confirmation."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stprotein2 import __version__
from stprotein2.data import PANEL
from stprotein2.io import sha256, write_json
from stprotein2.provenance import sources

HEADS = ('own_mlp', 'neighbor_mlp')
SEEDS = (2025, 2026, 2027)


def read(path):
    return json.loads(Path(path).read_text())


def summarize(rows, mapping, *, base=False):
    expected = {(seed, query) for seed in SEEDS for query in mapping}
    indexed = {}
    for row in rows:
        key = row['seed'], row['query']
        if key not in expected or key in indexed or row['source'] != mapping[row['query']]:
            raise ValueError('Unexpected or duplicated confirmation source/seed/query')
        metric = row['base_metrics'] if base else row['metrics']
        value = metric['summary']['rmse']
        if (not math.isfinite(value) or value < 0
                or tuple(p['protein'] for p in metric['per_protein']) != PANEL):
            raise ValueError('Finite complete-panel confirmation scores are required')
        indexed[key] = value
    if set(indexed) != expected:
        raise ValueError('All ten queries, six sources and three seeds must be present')
    groups = sorted(set(mapping.values()))
    per_seed = []
    for seed in SEEDS:
        scores = {group: float(np.mean([indexed[(seed, query)] for query in mapping if mapping[query] == group]))
                  for group in groups}
        per_seed.append({'seed': seed, 'source_macro_RMSE': float(np.mean(list(scores.values()))), 'per_source': scores})
    return {'complete': True, 'source_macro_RMSE': float(np.mean([r['source_macro_RMSE'] for r in per_seed])),
            'per_seed': per_seed, 'per_source': {g: float(np.mean([r['per_source'][g] for r in per_seed])) for g in groups}}


def validate_result(result, registered_decision, mapping):
    if (len(mapping) != 10 or len(set(mapping.values())) != 6 or result['all_cases_verified'] is not True
            or result['all_actual_pair_inputs_matched'] is not True or result['registered_new_tasks'] != 24
            or result['reused_seed2025_tasks'] != 12 or result['sources'] != 6 or result['query_slices'] != 10
            or result['development_only'] is not True or result['external_confirmation'] is not False
            or set(result['rows']) != set(HEADS)):
        raise ValueError('Complete registered development confirmation is required')
    summaries = {head: summarize(result['rows'][head], mapping) for head in HEADS}
    summaries['B0'] = summarize(result['rows']['own_mlp'], mapping, base=True)
    other_base = summarize(result['rows']['neighbor_mlp'], mapping, base=True)
    np.testing.assert_allclose(summaries['B0']['source_macro_RMSE'], other_base['source_macro_RMSE'], atol=1e-5, rtol=2e-5)
    for name, row in summaries.items():
        declared = result['summaries'][name]
        if declared['complete'] is not True or declared['scored_query_tissues'] != 30 or declared['required_query_tissues'] != 30:
            raise ValueError('Reported confirmation coverage differs')
        np.testing.assert_allclose(row['source_macro_RMSE'], declared['source_macro_RMSE'], atol=1e-12, rtol=0)
        if [r['seed'] for r in declared['per_seed']] != list(SEEDS):
            raise ValueError('Reported optimizer seeds differ')
        for a, b in zip(row['per_seed'], declared['per_seed'], strict=True):
            np.testing.assert_allclose(a['source_macro_RMSE'], b['source_macro_RMSE'], atol=1e-12, rtol=0)
    strict = min(summaries, key=lambda name: summaries[name]['source_macro_RMSE'])
    error = statistics.stdev([r['source_macro_RMSE'] for r in summaries[strict]['per_seed']]) / math.sqrt(3)
    cutoff = summaries[strict]['source_macro_RMSE'] + error
    eligible = [name for name in ('B0', *HEADS) if summaries[name]['source_macro_RMSE'] <= cutoff + 1e-12]
    for decision in (result['decision'], registered_decision):
        if (decision['strict_lowest_mean'] != strict or decision['compact_default'] != eligible[0]
                or decision['eligible_in_complexity_order'] != eligible or decision['donor_confidence_interval'] is not False):
            raise ValueError('The registered compact decision differs from independently recomputed selection')
        np.testing.assert_allclose(decision['lowest_mean_optimizer_SE'], error, atol=1e-12, rtol=0)
        np.testing.assert_allclose(decision['compact_cutoff_RMSE'], cutoff, atol=1e-12, rtol=0)
    return summaries, {'strict_lowest_mean': strict, 'compact_default': eligible[0],
                       'lowest_mean_optimizer_SE': error, 'compact_cutoff_RMSE': cutoff,
                       'eligible_in_complexity_order': eligible, 'donor_confidence_interval': False}


def freeze(research, cuda_validation, output):
    research, output = Path(research).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a fresh compact base release directory')
    campaign = research / 'runs/stprotein2_compact_confirmation_20261025_v1'
    marker = read(campaign / 'confirmation_complete.json')
    result_path, decision_path = campaign / 'reports/result.json', campaign / 'compact_decision.json'
    if (marker['status'] != 'passed' or sha256(result_path) != marker['result_sha256']
            or sha256(decision_path) != marker['decision_sha256']):
        raise ValueError('Passed confirmation report identity differs')
    registered = read(decision_path)
    if registered['result_sha256'] != sha256(result_path):
        raise ValueError('The decision is not bound to its result')
    protocol = read(campaign / 'protocol.json')
    mapping = read(protocol['groups'])['groups']
    if sha256(protocol['groups']) != protocol['groups_sha256'] or sha256(protocol['registry']) != protocol['registry_sha256']:
        raise ValueError('Registered dataset metadata changed')
    result = read(result_path)
    summaries, decision = validate_result(result, registered, mapping)
    validation_path = ROOT / 'docs/training_validation.json'
    validation = read(validation_path)
    if (validation['status'] != 'passed' or validation['source_sha256'] != sources()
            or validation['whole_nested_comparison']['fits_per_head'] != {'own_mlp': 31, 'neighbor_mlp': 31}
            or validation['whole_nested_comparison']['retained_weights_and_histories_tolerance'] != 0):
        raise ValueError('Current compact software has not passed full CPU compatibility')
    full = validation['whole_nested_comparison']
    if (sha256(ROOT / full['report']) != full['report_sha256']
            or sha256(ROOT / validation['functional_tests']['log']) != validation['functional_tests']['log_sha256']):
        raise ValueError('Current CPU validation artifacts changed')
    selected = decision['compact_default']
    cuda_identity = None
    if selected != 'B0':
        if cuda_validation is None:
            raise ValueError('The selected neural model requires its controlled CUDA compatibility proof')
        cuda = Path(cuda_validation).resolve()
        done, cuda_protocol = read(cuda / 'complete.json'), read(cuda / 'protocol.json')
        if (done['status'] != 'passed' or sha256(cuda / 'protocol.json') != done['protocol_sha256']
                or cuda_protocol['core_source_sha256'] != sources()
                or cuda_protocol['tensor_atol'] != 0 or cuda_protocol['tensor_rtol'] != 0
                or {row['head'] for row in done['results']} != set(HEADS)
                or any(row['selected_and_last_parameters_maximum_error'] != 0 for row in done['results'])):
            raise ValueError('Current controlled CUDA compatibility proof differs')
        cuda_identity = {'complete_sha256': sha256(cuda / 'complete.json'),
                         'scope': done['validation_scope'], 'path': str(cuda / 'complete.json')}
        for row in done['results']:
            if (sha256(cuda / row['head'] / 'original/model.pt') != row['original_model_sha256']
                    or sha256(cuda / row['head'] / 'compact/residual.pt') != row['compact_model_sha256']):
                raise ValueError('Controlled CUDA trained artifacts changed')
    deployment_path = ROOT / 'docs/deployment_validation.json'
    deployed = read(deployment_path)
    if (deployed['status'] != 'passed' or deployed['source_sha256'] != sources()
            or sha256(ROOT / deployed['real_wheel_replay']) != deployed['real_wheel_replay_sha256']):
        raise ValueError('Current standalone deployment proof differs')
    chosen = summaries[selected]
    baseline = summaries['B0']['source_macro_RMSE']
    deltas = {s: chosen['per_source'][s] - summaries['B0']['per_source'][s] for s in sorted(set(mapping.values()))}
    profile = {'schema': 'stprotein2_compact_base_freeze_v1', 'status': 'frozen_development_base',
               'observed_unix': time.time(), 'package_version': __version__, 'package_source_sha256': sources(),
               'selection': decision, 'summaries': summaries, 'chosen_source_macro_RMSE': chosen['source_macro_RMSE'],
               'chosen_minus_B0_per_source': deltas, 'sources_improved_over_B0': sum(v < 0 for v in deltas.values()),
               'relative_improvement_over_B0_percent': 100 * (baseline - chosen['source_macro_RMSE']) / baseline if baseline > 0 else None,
               'confirmation_complete_sha256': sha256(campaign / 'confirmation_complete.json'),
               'confirmation_result_sha256': sha256(result_path), 'confirmation_decision_sha256': sha256(decision_path),
               'training_validation_sha256': sha256(validation_path), 'deployment_validation_sha256': sha256(deployment_path),
               'controlled_CUDA_validation': cuda_identity, 'query_ADT_read_or_hashed_during_freeze': False,
               'selection_scope': 'registered six-source three-optimizer-seed A40 development comparison',
               'novelty_implemented': False, 'final_architecture_including_novelty_frozen': False,
               'formal_benchmark_complete': False, 'external_confirmation_complete': False,
               'universal_or_SOTA_optimum_claimed': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.compact-base-freeze-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'release'; stage.mkdir()
        if selected == 'B0':
            profile['train_cli_model'] = 'b0'
            profile['neural_deployed_parameters'] = 0
        else:
            from stprotein2.neural import CompactResidual, ResidualConfig
            import torch
            recipe = read(ROOT / 'configs/compact_residual.json')
            recipe['model']['residual_head'] = selected
            cfg = ResidualConfig(**recipe['model'])
            # Bind the recipe to an actual verified model and its registered
            # pre-selection budget, not to potentially edited example defaults.
            reused = read(campaign / 'reused_seed2025.json')
            source = next(r for r in reused if r['head'] == selected and r['query'] == 'Breast')
            task = read(source['matrix'])['tasks'][source['index']]
            model_path = Path(task['output']) / 'model.pt'
            if result['hashes'].get(str(model_path.resolve())) != sha256(model_path):
                raise ValueError('Selected real checkpoint is not bound by the confirmation audit')
            trained = torch.load(model_path, weights_only=True, map_location='cpu')
            converted, _ = CompactResidual.from_research_checkpoint(trained)
            if asdict(cfg) != asdict(converted.cfg):
                raise ValueError('Selected recipe model differs from the tested real architecture')
            if sha256(task['config']) != task['config_sha256']:
                raise ValueError('Selected registered recipe changed')
            old_train = read(task['config'])['train']
            expected_train = {k: old_train[k] for k in ('epochs', 'warmup_epochs', 'validate_every', 'patience',
                'seed', 'threads', 'lr', 'optimizer_epsilon', 'weight_decay', 'clip_grad', 'checkpoint_representations')}
            expected_train.update(checkpoint_delay=old_train['prototype_ramp_epochs'], auxiliary_weight=old_train['base_weight'])
            if (recipe['train'] != expected_train or old_train['residual_target'] != 'in_fit'
                    or old_train['teacher_mode'] != 'none' or old_train['mask_rates'] != [0.]
                    or old_train['count_keep_rates'] != [1.]):
                raise ValueError('Selected recipe training differs from the registered C0 experiment')
            write_json(stage / 'selected_training.json', recipe)
            profile['train_cli_model'] = selected
            profile['neural_deployed_parameters'] = converted.specification()['deployed_parameters']
            profile['selected_real_checkpoint_sha256'] = sha256(model_path)
            profile['selected_registered_configuration_sha256'] = sha256(task['config'])
            profile['selected_training_sha256'] = sha256(stage / 'selected_training.json')
        write_json(stage / 'compact_base_freeze.json', profile)
        lines = ['# 当前开发证据支持的精简基线', '',
                 f"精简默认：`{selected}`；严格最低均值：`{decision['strict_lowest_mean']}`。", '',
                 '| 候选 | 来源宏RMSE |', '|---|---:|']
        lines += [f"| {name} | {summaries[name]['source_macro_RMSE']:.9f} |" for name in ('B0', *HEADS)]
        lines += ['', '选择使用预登记的一倍优化标准误精简规则。三种子反映优化波动，不增加独立供体数量。', '',
                  '完整负例和逐来源差值保存在compact_base_freeze.json。当前只冻结开发基线；novelty、完整正式benchmark和外部确认仍有独立验收要求。', '']
        (stage / 'README.md').write_text('\n'.join(lines))
        write_json(stage / 'freeze_complete.json', {'status': 'passed',
                   'artifacts': {p.name: sha256(p) for p in stage.iterdir() if p.is_file()},
                   'final_paper_or_novelty_complete': False})
        stage.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', required=True, type=Path)
    parser.add_argument('--cuda-validation', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(freeze(args.research_root, args.cuda_validation, args.output))


if __name__ == '__main__':
    main()
