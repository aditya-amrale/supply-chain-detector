"""
Ablation study: compare detector combinations to understand
which signals contribute most to overall performance.

Runs four configurations and prints a comparison table:
  A) Name similarity only
  B) Name similarity + metadata
  C) Name similarity + static/AST
  D) All three combined (full detector)

Run:
  python -m evaluation.compare_detectors
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import logging
from datetime import datetime, timezone

from db.models import Package, Registry, RiskLevel
from detectors.name_similarity import NameSimilarityDetector, POPULAR_PYPI, POPULAR_NPM
from detectors.metadata_heuristics import MetadataHeuristicsDetector
from detectors.static_analysis import StaticAnalysisDetector
from detectors.ast_analyzer import ASTAnalyzer
from evaluation.ossf_evaluator import (
    OSSFEvaluator, SYNTHETIC_DATASET, MALICIOUS_SCRIPT,
    BENIGN_SCRIPT, EvalReport
)
from scoring.risk_scorer import WEIGHT_SIMILARITY, WEIGHT_METADATA, WEIGHT_STATIC


class AblationEvaluator(OSSFEvaluator):
    """Subclass that lets us disable individual detectors."""

    def __init__(self, use_similarity=True, use_metadata=True,
                 use_static=True, flag_on_medium=True):
        super().__init__(flag_on_medium=flag_on_medium)
        self.use_similarity = use_similarity
        self.use_metadata   = use_metadata
        self.use_static     = use_static

    def _run(self, samples):
        report = EvalReport(evaluated_at=datetime.now(timezone.utc).isoformat())

        for pkg, true_label, source_files in samples:
            sim_val  = self.similarity.check(pkg.name).signal_value if self.use_similarity else 0.0
            meta_val = self.metadata.check(pkg).signal_value        if self.use_metadata   else 0.0

            if self.use_static:
                stat_r = self.static_det.scan_files(source_files)
                ast_r  = self.ast_det.analyze_files(source_files)
                stat_val = max(stat_r.signal_value, ast_r.signal_value)
            else:
                stat_val = 0.0

            score = (
                sim_val  * WEIGHT_SIMILARITY +
                meta_val * WEIGHT_METADATA   +
                stat_val * WEIGHT_STATIC
            )

            if   score >= 70: risk_level = RiskLevel.HIGH
            elif score >= 30: risk_level = RiskLevel.MEDIUM
            else:             risk_level = RiskLevel.LOW

            is_pos    = (risk_level==RiskLevel.HIGH or
                         (self.flag_on_medium and risk_level==RiskLevel.MEDIUM))
            predicted = "malicious" if is_pos else "benign"

            if   true_label=="malicious" and predicted=="malicious": report.tp+=1
            elif true_label=="benign"    and predicted=="malicious": report.fp+=1
            elif true_label=="benign"    and predicted=="benign":    report.tn+=1
            else:                                                     report.fn+=1
            report.total += 1

        return report


def run_ablation():
    samples = []
    for reg, name, label, version, desc, has_script in SYNTHETIC_DATASET:
        script = MALICIOUS_SCRIPT if has_script else BENIGN_SCRIPT
        pkg = OSSFEvaluator()._make_package(reg, name, version, desc, label)
        samples.append((pkg, label, {"script.py": script}))

    configs = [
        ("Similarity only",              True,  False, False),
        ("Similarity + Metadata",        True,  True,  False),
        ("Similarity + Static/AST",      True,  False, True),
        ("All detectors (full)",         True,  True,  True),
    ]

    print("\n" + "=" * 78)
    print(f"  {'Configuration':<32} {'P':>6} {'R':>6} {'F1':>6} {'Acc':>6} {'FPR':>6}  TP/FP/TN/FN")
    print("=" * 78)

    for label, sim, meta, static in configs:
        ev     = AblationEvaluator(use_similarity=sim, use_metadata=meta,
                                   use_static=static, flag_on_medium=True)
        report = ev._run(samples)
        print(
            f"  {label:<32} "
            f"{report.precision:>6.3f} "
            f"{report.recall:>6.3f} "
            f"{report.f1:>6.3f} "
            f"{report.accuracy:>6.3f} "
            f"{report.fpr:>6.3f}  "
            f"{report.tp}/{report.fp}/{report.tn}/{report.fn}"
        )

    print("=" * 78)
    print("\n  Interpretation guide:")
    print("  P=Precision  R=Recall  F1=harmonic mean  Acc=Accuracy  FPR=False positive rate")
    print("  Higher P = fewer false alarms.  Higher R = fewer missed attacks.\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_ablation()