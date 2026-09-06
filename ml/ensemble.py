"""
Ensemble model combining logistic regression + heuristic scoring.

Three ensemble strategies implemented:
  1. Hard voting   — majority vote of N classifiers
  2. Soft voting   — weighted average of probabilities
  3. Stacking      — a meta-classifier trained on base model outputs

Run:
  python -m ml.ensemble --train
  python -m ml.ensemble --predict --name reqests
"""

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml.feature_extractor import FeatureExtractor, FEATURE_NAMES, N_FEATURES
from ml.classifier import (
    LogisticRegression, StandardScaler, compute_metrics,
    build_dataset, heuristic_predict, cross_validate,
    stratified_kfold, mean, std, sigmoid, dot,
    MALICIOUS_SCRIPT, BENIGN_SCRIPT, RAW_DATASET,
)

ENSEMBLE_MODEL_PATH = Path(__file__).parent / "saved_ensemble.json"


@dataclass
class BaseModelOutputs:
    lr_proba:        float
    heuristic_proba: float
    ast_proba:       float
    true_label:      int


def _heuristic_proba(features: list[float]) -> float:
    fv = features
    score = 0.0

    lev_d = fv[0]
    if lev_d <= 1:   score += 38
    elif lev_d <= 2: score += 28
    elif lev_d <= 3: score += 20
    if fv[1] > 0:    score += 32
    if fv[2] > 0:    score += 28

    if fv[3] < 30:   score += 20
    if fv[4] <= 1:   score += 15
    if fv[8] < 20:   score += 10
    if fv[6] >= 2:   score += 25
    if fv[7] == 0:   score += 10
    if fv[5] < 7:    score +=  5

    if fv[11] > 0:   score += 21
    if fv[12] > 0:   score += 14
    if fv[13] > 0:   score += 14
    if fv[14] > 0:   score += 14

    return min(score / 120.0, 1.0)


def _ast_proba(features: list[float]) -> float:
    has_shell  = features[11]
    has_env    = features[12]
    has_net    = features[13]
    has_b64    = features[14]
    has_file   = features[15]
    n_findings = features[16]

    score = (
        has_shell  * 0.90 +
        has_env    * 0.55 +
        has_net    * 0.55 +
        has_b64    * 0.55 +
        has_file   * 0.70 +
        min(n_findings * 0.05, 0.15)
    )
    return min(score, 1.0)


class SoftVotingEnsemble:
    def __init__(
        self,
        lr_weight:         float = 0.50,
        heuristic_weight:  float = 0.30,
        ast_weight:        float = 0.20,
        threshold:         float = 0.45,
    ):
        self.weights   = [lr_weight, heuristic_weight, ast_weight]
        self.threshold = threshold
        self.lr_model: LogisticRegression | None = None
        self.scaler:   StandardScaler | None     = None

    def fit(self, samples) -> "SoftVotingEnsemble":
        X = [s.features for s in samples]
        y = [s.label    for s in samples]
        self.scaler   = StandardScaler().fit(X)
        Xs            = self.scaler.transform(X)
        self.lr_model = LogisticRegression(lr=0.1, n_iter=600, lambda_reg=0.005)
        self.lr_model.fit(Xs, y)
        return self

    def predict_proba(self, features: list[float]) -> float:
        Xs         = self.scaler.transform([features])[0]
        lr_p       = sigmoid(dot(self.lr_model.weights_, Xs) + self.lr_model.bias_)
        heuristic_p= _heuristic_proba(features)
        ast_p      = _ast_proba(features)

        return sum(w * p for w, p in zip(
            self.weights, [lr_p, heuristic_p, ast_p]
        ))

    def predict(self, features: list[float]) -> int:
        return 1 if self.predict_proba(features) >= self.threshold else 0

    def predict_batch(self, samples) -> list[int]:
        return [self.predict(s.features) for s in samples]

    def to_dict(self) -> dict:
        return {
            "type":      "soft_voting",
            "weights":   self.weights,
            "threshold": self.threshold,
            "lr_model":  self.lr_model.to_dict() if self.lr_model else None,
            "scaler":    self.scaler.to_dict()   if self.scaler   else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SoftVotingEnsemble":
        ens = cls(
            lr_weight        = d["weights"][0],
            heuristic_weight = d["weights"][1],
            ast_weight       = d["weights"][2],
            threshold        = d["threshold"],
        )
        if d.get("lr_model"):
            ens.lr_model = LogisticRegression.from_dict(d["lr_model"])
        if d.get("scaler"):
            ens.scaler   = StandardScaler.from_dict(d["scaler"])
        return ens


class StackingEnsemble:
    def __init__(self, threshold: float = 0.5):
        self.threshold   = threshold
        self.lr_model:   LogisticRegression | None = None
        self.scaler:     StandardScaler | None     = None
        self.meta_model: LogisticRegression | None = None
        self._meta_scaler: StandardScaler | None  = None

    def fit(self, samples, k: int = 5) -> "StackingEnsemble":
        import random
        random.seed(42)

        X = [s.features for s in samples]
        y = [s.label    for s in samples]

        meta_X = [[0.0, 0.0, 0.0] for _ in samples]

        for train_idx, test_idx in stratified_kfold(samples, k=k):
            X_tr = [samples[i].features for i in train_idx]
            y_tr = [samples[i].label    for i in train_idx]

            fold_scaler   = StandardScaler().fit(X_tr)
            fold_lr       = LogisticRegression(lr=0.1, n_iter=400, lambda_reg=0.005)
            fold_lr.fit(fold_scaler.transform(X_tr), y_tr)

            for idx in test_idx:
                fv = samples[idx].features
                Xs = fold_scaler.transform([fv])[0]
                lr_p       = sigmoid(dot(fold_lr.weights_, Xs) + fold_lr.bias_)
                heuristic_p= _heuristic_proba(fv)
                ast_p      = _ast_proba(fv)
                meta_X[idx] = [lr_p, heuristic_p, ast_p]

        meta_scaler    = StandardScaler().fit(meta_X)
        meta_X_scaled  = meta_scaler.transform(meta_X)
        meta_lr        = LogisticRegression(lr=0.2, n_iter=300, lambda_reg=0.01)
        meta_lr.fit(meta_X_scaled, y)

        self.scaler   = StandardScaler().fit(X)
        self.lr_model = LogisticRegression(lr=0.1, n_iter=600, lambda_reg=0.005)
        self.lr_model.fit(self.scaler.transform(X), y)

        self._meta_scaler = meta_scaler
        self.meta_model   = meta_lr
        return self

    def predict_proba(self, features: list[float]) -> float:
        Xs  = self.scaler.transform([features])[0]
        lr_p       = sigmoid(dot(self.lr_model.weights_, Xs) + self.lr_model.bias_)
        heuristic_p= _heuristic_proba(features)
        ast_p      = _ast_proba(features)

        meta_x = self._meta_scaler.transform([[lr_p, heuristic_p, ast_p]])[0]
        return sigmoid(dot(self.meta_model.weights_, meta_x) + self.meta_model.bias_)

    def predict(self, features: list[float]) -> int:
        return 1 if self.predict_proba(features) >= self.threshold else 0

    def predict_batch(self, samples) -> list[int]:
        return [self.predict(s.features) for s in samples]

    def to_dict(self) -> dict:
        return {
            "type":         "stacking",
            "threshold":    self.threshold,
            "lr_model":     self.lr_model.to_dict() if self.lr_model else None,
            "scaler":       self.scaler.to_dict() if self.scaler else None,
            "meta_model":   self.meta_model.to_dict() if self.meta_model else None,
            "meta_scaler":  self._meta_scaler.to_dict() if self._meta_scaler else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StackingEnsemble":
        ens = cls(threshold=d["threshold"])
        if d.get("lr_model"):
            ens.lr_model      = LogisticRegression.from_dict(d["lr_model"])
        if d.get("scaler"):
            ens.scaler        = StandardScaler.from_dict(d["scaler"])
        if d.get("meta_model"):
            ens.meta_model    = LogisticRegression.from_dict(d["meta_model"])
        if d.get("meta_scaler"):
            ens._meta_scaler  = StandardScaler.from_dict(d["meta_scaler"])
        return ens


def cv_ensemble(model_cls, samples, k=5, **kwargs) -> dict:
    metrics = {"precision": [], "recall": [], "f1": [], "accuracy": [], "fpr": []}

    for train_idx, test_idx in stratified_kfold(samples, k=k):
        train = [samples[i] for i in train_idx]
        test  = [samples[i] for i in test_idx]
        y_te  = [samples[i].label for i in test_idx]

        model = model_cls(**kwargs)
        model.fit(train)
        preds = model.predict_batch(test)
        r     = compute_metrics(y_te, preds)

        metrics["precision"].append(r.precision)
        metrics["recall"].append(r.recall)
        metrics["f1"].append(r.f1)
        metrics["accuracy"].append(r.accuracy)
        metrics["fpr"].append(r.fpr)

    return {k: {"mean": mean(v), "std": std(v)} for k, v in metrics.items()}


def train_and_compare():
    print("\n" + "="*65)
    print("  ENSEMBLE MODEL — TRAINING & COMPARISON")
    print("="*65)

    samples = build_dataset()
    X = [s.features for s in samples]
    y = [s.label    for s in samples]

    print("\n  [1/3] Logistic Regression (baseline) ...")
    scaler  = StandardScaler().fit(X)
    lr_solo = LogisticRegression(lr=0.1, n_iter=600, lambda_reg=0.005)
    lr_solo.fit(scaler.transform(X), y)
    y_lr    = [1 if sigmoid(dot(lr_solo.weights_, scaler.transform([fv])[0]) + lr_solo.bias_) >= 0.5 else 0
               for fv in X]
    r_lr    = compute_metrics(y, y_lr, "Logistic Regression")

    print("  [2/3] Soft Voting Ensemble ...")
    sv      = SoftVotingEnsemble().fit(samples)
    y_sv    = sv.predict_batch(samples)
    r_sv    = compute_metrics(y, y_sv, "Soft Voting")

    print("  [3/3] Stacking Ensemble (5-fold OOF) ...")
    st      = StackingEnsemble().fit(samples)
    y_st    = st.predict_batch(samples)
    r_st    = compute_metrics(y, y_st, "Stacking")

    print("\n  " + "─"*60)
    print(f"  {'Metric':<14} {'LR Baseline':>14} {'Soft Voting':>14} {'Stacking':>14}")
    print("  " + "─"*60)

    for metric in ["precision", "recall", "f1", "accuracy", "fpr"]:
        lv = getattr(r_lr, metric)
        sv_v = getattr(r_sv, metric)
        st_v = getattr(r_st, metric)
        lower_better = metric == "fpr"

        all_v = [lv, sv_v, st_v]
        best  = min(all_v) if lower_better else max(all_v)

        def fmt(v):
            mark = " ◀" if v == best else "   "
            return f"{v:.3f}{mark}"

        print(f"  {metric:<14} {fmt(lv):>14} {fmt(sv_v):>14} {fmt(st_v):>14}")

    print("  " + "─"*60)

    data = st.to_dict()
    ENSEMBLE_MODEL_PATH.write_text(json.dumps(data, indent=2))
    print(f"\n  Ensemble saved → {ENSEMBLE_MODEL_PATH}")

    return sv, st


def ensemble_predict(name: str, registry: str = "pypi") -> dict:
    if not ENSEMBLE_MODEL_PATH.exists():
        raise FileNotFoundError("No saved ensemble. Run: python -m ml.ensemble --train")

    data = json.loads(ENSEMBLE_MODEL_PATH.read_text())
    if data["type"] == "stacking":
        model = StackingEnsemble.from_dict(data)
    else:
        model = SoftVotingEnsemble.from_dict(data)

    extractor = FeatureExtractor()
    pkg = {
        "name": name, "version": "1.0.0", "description": "",
        "repository_url": "", "publisher_account_age_days": 5,
        "publisher_pkg_count": 1, "package_age_days": 2, "weekly_downloads": 0,
    }
    fv    = extractor.extract(pkg)
    proba = model.predict_proba(fv)
    pred  = 1 if proba >= model.threshold else 0
    level = "HIGH" if proba >= 0.70 else "MEDIUM" if proba >= 0.40 else "LOW"

    return {
        "name":        name,
        "registry":    registry,
        "probability": round(proba, 4),
        "prediction":  "malicious" if pred == 1 else "benign",
        "risk_level":  level,
        "model_type":  data["type"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble classifier")
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--predict",  action="store_true")
    parser.add_argument("--name",     type=str, default="reqests")
    parser.add_argument("--registry", type=str, default="pypi")
    args = parser.parse_args()

    if args.train:
        train_and_compare()

    if args.predict:
        result = ensemble_predict(args.name, args.registry)
        print(f"\n  Package    : {result['registry']}/{result['name']}")
        print(f"  Model      : {result['model_type']}")
        print(f"  Probability: {result['probability']:.4f}  ({result['probability']*100:.1f}% malicious)")
        print(f"  Prediction : {result['prediction'].upper()}")
        print(f"  Risk level : {result['risk_level']}\n")