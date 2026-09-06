"""
Unit tests for the ML pipeline.
"""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml.feature_extractor import FeatureExtractor, FEATURE_NAMES, N_FEATURES
from ml.classifier import (
    StandardScaler, LogisticRegression, compute_metrics,
    stratified_kfold, build_dataset, heuristic_predict,
    cross_validate, ClassificationReport, sigmoid, dot, mean, std,
)


class TestMathHelpers:
    def test_sigmoid_zero(self):        assert sigmoid(0) == pytest.approx(0.5)
    def test_sigmoid_large_pos(self):   assert sigmoid(100) == pytest.approx(1.0, abs=1e-6)
    def test_sigmoid_large_neg(self):   assert sigmoid(-100) == pytest.approx(0.0, abs=1e-6)
    def test_dot_product(self):         assert dot([1,2,3],[4,5,6]) == pytest.approx(32.0)
    def test_mean(self):                assert mean([1,2,3,4,5]) == pytest.approx(3.0)
    def test_mean_empty(self):          assert mean([]) == 0.0
    def test_std(self):
        assert std([2,4,4,4,5,5,7,9]) == pytest.approx(2.0, rel=0.05)


class TestFeatureExtractor:

    @pytest.fixture
    def ex(self): return FeatureExtractor()

    def test_output_length(self, ex):
        fv = ex.extract({"name": "requests", "version": "2.31.0"})
        assert len(fv) == N_FEATURES

    def test_popular_lev_zero(self, ex):
        fv = ex.extract({"name": "requests"})
        assert fv[0] == 0.0

    def test_typosquat_lev_one(self, ex):
        fv = ex.extract({"name": "reqests"})
        assert fv[0] == 1.0

    def test_suspicious_suffix(self, ex):
        assert ex.extract({"name": "lib-internal"})[2] == 1.0

    def test_no_repo(self, ex):
        assert ex.extract({"name": "x", "repository_url": ""})[7] == 0.0

    def test_has_repo(self, ex):
        assert ex.extract({"name": "x", "repository_url": "https://github.com/a/b"})[7] == 1.0

    def test_high_version(self, ex):
        assert ex.extract({"name": "x", "version": "9.0.0"})[6] == 9.0

    def test_shell_exec(self, ex):
        fv = ex.extract({"name":"x"}, {"s.py":"import subprocess\nsubprocess.run(['id'])"})
        assert fv[11] == 1.0

    def test_env_access(self, ex):
        fv = ex.extract({"name":"x"}, {"s.py":"import os\nos.environ.get('KEY')"})
        assert fv[12] == 1.0

    def test_network_call(self, ex):
        fv = ex.extract({"name":"x"}, {"s.py":"import socket\ns=socket.socket()"})
        assert fv[13] == 1.0

    def test_clean_static_all_zero(self, ex):
        fv = ex.extract({"name":"x"}, {"s.py":"def add(a,b): return a+b"})
        assert fv[11] == fv[12] == fv[13] == fv[14] == 0.0

    def test_all_finite(self, ex):
        pkg = {"name":"reqests","version":"99.0.0","description":"",
               "repository_url":"","publisher_account_age_days":2,
               "publisher_pkg_count":1,"package_age_days":1,"weekly_downloads":0}
        assert all(math.isfinite(v) for v in ex.extract(pkg))


class TestStandardScaler:

    def test_zero_mean_after_fit(self):
        X  = [[1,2],[3,4],[5,6]]
        Xt = StandardScaler().fit_transform(X)
        assert mean([r[0] for r in Xt]) == pytest.approx(0.0, abs=1e-9)

    def test_unit_std_after_fit(self):
        X  = [[float(i)] for i in range(1, 11)]
        Xt = StandardScaler().fit_transform(X)
        assert std([r[0] for r in Xt]) == pytest.approx(1.0, rel=0.05)

    def test_serialization_roundtrip(self):
        X  = [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]
        sc = StandardScaler().fit(X)
        sc2 = StandardScaler.from_dict(sc.to_dict())
        for r1, r2 in zip(sc.transform(X), sc2.transform(X)):
            for v1, v2 in zip(r1, r2):
                assert v1 == pytest.approx(v2)


class TestLogisticRegression:

    def test_learns_separable(self):
        X = [[1.0],[1.0],[1.0],[-1.0],[-1.0],[-1.0]]
        y = [1,1,1,0,0,0]
        m = LogisticRegression(lr=0.5, n_iter=300).fit(X, y)
        assert m.predict(X) == y

    def test_loss_decreases(self):
        X = [[1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,1.0]]
        y = [1,1,0,0]
        m = LogisticRegression(lr=0.3, n_iter=100).fit(X, y)
        assert m.loss_history_[0] > m.loss_history_[-1]

    def test_probabilities_bounded(self):
        X = [[float(i)] for i in range(-5, 6)]
        y = [1 if i > 0 else 0 for i in range(-5, 6)]
        m = LogisticRegression(lr=0.3, n_iter=100).fit(X, y)
        assert all(0.0 <= p <= 1.0 for p in m.predict_proba(X))

    def test_serialization_roundtrip(self):
        X = [[1.0,2.0],[3.0,4.0],[-1.0,-2.0],[-3.0,-4.0]]
        y = [1,1,0,0]
        m1 = LogisticRegression(lr=0.3, n_iter=100).fit(X, y)
        m2 = LogisticRegression.from_dict(m1.to_dict())
        assert m1.predict(X) == m2.predict(X)

    def test_threshold_control(self):
        m = LogisticRegression(lr=0.1, n_iter=1)
        m.weights_ = [0.0]; m.bias_ = 0.0
        assert m.predict([[0.0]], threshold=0.3) == [1]
        assert m.predict([[0.0]], threshold=0.6) == [0]


class TestMetrics:

    def test_perfect(self):
        y = [1,1,0,0]
        r = compute_metrics(y, y)
        assert r.precision == r.recall == r.f1 == r.accuracy == 1.0
        assert r.fpr == 0.0

    def test_all_wrong(self):
        r = compute_metrics([1,1,0,0],[0,0,1,1])
        assert r.tp == 0; assert r.fn == 2; assert r.fp == 2

    def test_f1_harmonic(self):
        r = ClassificationReport(tp=2, fp=2, tn=0, fn=0, total=4)
        assert r.f1 == pytest.approx(2/3, rel=1e-4)


class TestCrossValidation:

    def test_kfold_covers_all(self):
        samples = build_dataset()
        seen = set()
        for _, test_idx in stratified_kfold(samples, k=5):
            seen.update(test_idx)
        assert len(seen) == len(samples)

    def test_kfold_no_leakage(self):
        samples = build_dataset()
        for train_idx, test_idx in stratified_kfold(samples, k=5):
            assert set(train_idx).isdisjoint(set(test_idx))

    def test_cv_metrics_valid_range(self):
        samples = build_dataset()
        cv = cross_validate(samples, k=3, n_iter=150)
        for metric, vals in cv.items():
            assert 0.0 <= vals["mean"] <= 1.0
            assert vals["std"] >= 0.0


class TestFullPipeline:

    def test_dataset_correct_size(self):
        from ml.classifier import RAW_DATASET
        samples = build_dataset()
        assert len(samples) == len(RAW_DATASET)
        assert all(s.label in (0,1) for s in samples)
        assert all(len(s.features) == N_FEATURES for s in samples)

    def test_heuristic_f1_above_chance(self):
        samples = build_dataset()
        r = compute_metrics([s.label for s in samples], heuristic_predict(samples))
        assert r.f1 > 0.5

    def test_logreg_f1_above_chance(self):
        samples = build_dataset()
        X = [s.features for s in samples]
        y = [s.label    for s in samples]
        sc = StandardScaler().fit(X)
        m  = LogisticRegression(lr=0.1, n_iter=300).fit(sc.transform(X), y)
        r  = compute_metrics(y, m.predict(sc.transform(X)))
        assert r.f1 > 0.6
        assert r.accuracy > 0.6