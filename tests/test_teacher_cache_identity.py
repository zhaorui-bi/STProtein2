"""Serialization regression and unchanged model math for the Teacher v2 cache fix."""
import io
import json

import joblib
import numpy as np
import pytest

from stprotein2.graph import FrozenRouter
from stprotein2_teacher_ablation_v2.identity import content_identity
from stprotein2_teacher_ablation_v2 import nested, trainer
import test_teacher_ablation as original_tests


class NoncontiguousRouter(FrozenRouter):
    def __init__(self, seed=2025):
        super().__init__(components=3, clusters=3, seed=seed)

    def fit(self, arrays, groups):
        super().fit(arrays, groups)
        # Identical PCA values, valid negative-stride storage, changed opaque pickle hash.
        self.pca_.components_ = self.pca_.components_[:, ::-1].copy()[:, ::-1]
        return self


def test_noncontiguous_fitted_router_roundtrip_preserves_content_but_detects_mutations():
    rng = np.random.default_rng(73)
    x = rng.normal(size=(90, 12)).astype(np.float32)
    router = NoncontiguousRouter().fit([x[:30], x[30:]], ['a', 'b'])
    buffer = io.BytesIO(); joblib.dump(router, buffer); buffer.seek(0)
    restored = joblib.load(buffer)
    assert joblib.hash(router) != joblib.hash(restored)
    assert content_identity(router) == content_identity(restored)
    for a, b in zip(router.transform(x), restored.transform(x)):
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-7)
    restored.pca_.components_[0, 0] += .001
    assert content_identity(router) != content_identity(restored)
    with pytest.raises(TypeError, match='Object arrays'):
        content_identity(np.array([object()], dtype=object))
    with pytest.raises(ValueError):
        content_identity(float('nan'))


def test_v2_teacher_ema_matches_version1_training_weights(tmp_path):
    case = original_tests.setup_case(tmp_path)
    before, old = original_tests.run(case, tmp_path/'old', 'ema')
    after, new = original_tests.run(case, tmp_path/'new', 'ema', module=trainer)
    original_tests.assert_weights_equal(before, after)
    assert old['reference_validation_mse'] == new['reference_validation_mse']
    assert old['total_teacher_forwards'] == new['total_teacher_forwards']


def test_completed_fit_reuses_noncontiguous_router_without_retraining(tmp_path, monkeypatch):
    from test_crossfit import cohort_files, configs
    from stprotein2_teacher_ablation_v2.config import TeacherAblationTrainConfig
    cohort, _, _ = cohort_files(tmp_path)
    cfg, old = configs()
    train = TeacherAblationTrainConfig(**old.to_dict(), spatial_alpha=.5, teacher_mode='none')
    monkeypatch.setattr(nested, 'FrozenRouter', NoncontiguousRouter)
    first, report, _, router = nested.fit_block(cohort, ['s0','s1','s2'], ['s4'], cfg, train, tmp_path/'fit', 'cpu')
    disk = joblib.load(tmp_path/'fit/preprocessing.joblib')
    assert joblib.hash(router) != joblib.hash(disk['router'])
    monkeypatch.setattr(trainer, 'initialize_prototypes', lambda *a, **k: pytest.fail('Completed fit retrained'))
    second, restored, _, _ = nested.fit_block(cohort, ['s0','s1','s2'], ['s4'], cfg, train, tmp_path/'fit', 'cpu')
    original_tests.assert_weights_equal(first, second)
    assert report == restored
    marker = json.loads((tmp_path/'fit/preprocessing_complete.json').read_text())
    assert marker['content_identity'] == content_identity(disk)
    disk['router'].centers_[0,0] += .1
    joblib.dump(disk,tmp_path/'fit/preprocessing.joblib')
    with pytest.raises(RuntimeError,match='preprocessing changed'):
        nested.fit_block(cohort, ['s0','s1','s2'], ['s4'], cfg, train, tmp_path/'fit', 'cpu')


def test_v2_full_nested_query_isolation_and_preprocessing_markers(tmp_path, monkeypatch):
    monkeypatch.setattr(original_tests, 'nested', nested)
    original_tests.test_nested_31_fits_query_ADT_absent_recovery_and_separate_scoring(tmp_path, monkeypatch)
    output = tmp_path/'outer'
    markers = list(output.rglob('preprocessing_complete.json'))
    assert len(markers) == 31
    complete = json.loads((output/'prediction_complete.json').read_text())
    for path in markers:
        relative = str(path.relative_to(output))
        assert relative in complete['artifacts']
        assert str(path.with_name('preprocessing.joblib').relative_to(output)) in complete['artifacts']
