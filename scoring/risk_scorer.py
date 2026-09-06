"""
Combines signals from all detectors into a single 0–100 risk score
and a RiskLevel (low / medium / high).

Scoring weights:
  - Name similarity    : 40 pts max  (typosquatting is the #1 signal)
  - Metadata heuristics: 35 pts max  (version jump, new publisher, etc.)
  - Static analysis    : 25 pts max  (suspicious install scripts)

Thresholds:
  - 0–29   → LOW
  - 30–69  → MEDIUM
  - 70–100 → HIGH

Run standalone:
  python -m scoring.risk_scorer --registry pypi --name reqests
"""

import logging
from dataclasses import dataclass

from db.models import Package, ScanResult, DetectorSignal, RiskLevel, get_session

logger = logging.getLogger(__name__)


# ── Weights ────────────────────────────────────────────────────────────────

WEIGHT_SIMILARITY = 40.0
WEIGHT_METADATA   = 35.0
WEIGHT_STATIC     = 25.0


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class ScoreBreakdown:
    similarity_score: float   # 0–40
    metadata_score:   float   # 0–35
    static_score:     float   # 0–25
    total:            float   # 0–100
    risk_level:       RiskLevel
    verdict_reason:   str


# ── Scorer ─────────────────────────────────────────────────────────────────

class RiskScorer:
    """
    Orchestrates all three detectors and produces a final risk score.
    """

    THRESHOLDS = {
        RiskLevel.HIGH:   70,
        RiskLevel.MEDIUM: 30,
        RiskLevel.LOW:    0,
    }

    def __init__(
        self,
        similarity_detector,
        metadata_detector,
        static_detector,
        db_engine,
    ):
        self.similarity = similarity_detector
        self.metadata   = metadata_detector
        self.static     = static_detector
        self.engine     = db_engine

    def score(
        self,
        pkg: Package,
        source_files: dict[str, str] | None = None,
    ) -> ScoreBreakdown:
        """
        Run all detectors on `pkg` and save results to DB.
        """

        # ── Run detectors ──────────────────────────────────────────────────

        sim_result  = self.similarity.check(pkg.name)
        meta_result = self.metadata.check(pkg)
        stat_result = self.static.scan_files(source_files or {})

        # ── Compute weighted scores ────────────────────────────────────────

        similarity_score = round(sim_result.signal_value  * WEIGHT_SIMILARITY, 2)
        metadata_score   = round(meta_result.signal_value * WEIGHT_METADATA,   2)
        static_score     = round(stat_result.signal_value * WEIGHT_STATIC,     2)
        total            = round(similarity_score + metadata_score + static_score, 2)

        # ── Determine risk level ───────────────────────────────────────────

        if total >= self.THRESHOLDS[RiskLevel.HIGH]:
            risk_level = RiskLevel.HIGH
        elif total >= self.THRESHOLDS[RiskLevel.MEDIUM]:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # ── Build verdict reason ───────────────────────────────────────────

        reasons = []
        if sim_result.is_suspicious:
            reasons.append(
                f"name similar to '{sim_result.closest_match}' "
                f"(dist={sim_result.closest_distance}, method={sim_result.match_method})"
            )
        if meta_result.total_score > 0:
            reasons.append(f"metadata flags: [{meta_result.summary()}]")
        if stat_result.has_suspicious_scripts:
            reasons.append(f"suspicious scripts: [{stat_result.summary()}]")

        verdict_reason = "; ".join(reasons) if reasons else "no suspicious signals"

        breakdown = ScoreBreakdown(
            similarity_score = similarity_score,
            metadata_score   = metadata_score,
            static_score     = static_score,
            total            = total,
            risk_level       = risk_level,
            verdict_reason   = verdict_reason,
        )

        risk_str = risk_level.value if hasattr(risk_level, 'value') else str(risk_level)
        logger.info(
            f"Score for {pkg.name}: {total:.1f} [{risk_str.upper()}] — {verdict_reason}"
        )

        # ── Persist to DB ──────────────────────────────────────────────────

        self._save_scan_result(pkg, sim_result, meta_result, stat_result, breakdown)

        return breakdown

    def _save_scan_result(self, pkg, sim_result, meta_result, stat_result, breakdown):
        """Upsert a ScanResult and its DetectorSignal rows."""
        with get_session(self.engine) as session:
            result = (
                session.query(ScanResult)
                .filter_by(package_id=pkg.id)
                .first()
            ) or ScanResult(package_id=pkg.id)

            result.risk_score               = breakdown.total
            result.risk_level               = breakdown.risk_level
            result.is_typosquat             = sim_result.is_suspicious
            result.is_dependency_confusion  = any(
                s.name == "high_version_jump" and s.fired
                for s in meta_result.signals
            )
            result.has_suspicious_scripts   = stat_result.has_suspicious_scripts
            result.closest_match            = sim_result.closest_match
            result.closest_match_dist       = sim_result.closest_distance
            result.verdict_reason           = breakdown.verdict_reason

            session.add(result)
            session.flush()

            session.query(DetectorSignal).filter_by(scan_result_id=result.id).delete()

            signals = []

            # Similarity signal
            signals.append(DetectorSignal(
                scan_result_id = result.id,
                detector_name  = "name_similarity",
                signal_value   = sim_result.signal_value,
                weight         = WEIGHT_SIMILARITY,
                contribution   = breakdown.similarity_score,
                detail         = sim_result.detail,
            ))

            # Metadata signals
            for s in meta_result.signals:
                signals.append(DetectorSignal(
                    scan_result_id = result.id,
                    detector_name  = s.name,
                    signal_value   = 1.0 if s.fired else 0.0,
                    weight         = s.weight,
                    contribution   = s.contribution,
                    detail         = s.detail,
                ))

            # Static analysis signals
            for m in stat_result.matches:
                signals.append(DetectorSignal(
                    scan_result_id = result.id,
                    detector_name  = f"static_{m.pattern_name}",
                    signal_value   = 1.0,
                    weight         = WEIGHT_STATIC,
                    contribution   = m.severity * 3.0,
                    detail         = f"line {m.line_number}: {m.snippet[:80]}",
                ))

            session.bulk_save_objects(signals)
            session.commit()

            logger.debug(f"Saved ScanResult id={result.id} with {len(signals)} signals")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from db.models import Registry, init_db, get_session as gs
    from detectors.name_similarity import NameSimilarityDetector, POPULAR_PYPI, POPULAR_NPM
    from detectors.metadata_heuristics import MetadataHeuristicsDetector
    from detectors.static_analysis import StaticAnalysisDetector

    parser = argparse.ArgumentParser(description="Score a package from the DB")
    parser.add_argument("--registry", choices=["npm", "pypi"], default="pypi")
    parser.add_argument("--name",     required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    engine = init_db()
    with gs(engine) as session:
        reg = Registry(args.registry)
        pkg = session.query(Package).filter_by(registry=reg, name=args.name).first()

    if not pkg:
        print(f"Package not in DB. Run: python -m monitor.package_fetcher --registry {args.registry} --name {args.name}")
        sys.exit(1)

    ref_list = POPULAR_PYPI if args.registry == "pypi" else POPULAR_NPM
    scorer   = RiskScorer(
        similarity_detector = NameSimilarityDetector(ref_list),
        metadata_detector   = MetadataHeuristicsDetector(),
        static_detector     = StaticAnalysisDetector(),
        db_engine           = engine,
    )

    bd = scorer.score(pkg)

    risk_str = bd.risk_level.value if hasattr(bd.risk_level, 'value') else str(bd.risk_level)

    print(f"\n{'─'*50}")
    print(f"  Package    : {args.registry}/{args.name}")
    print(f"  Risk level : {risk_str.upper()}")
    print(f"  Score      : {bd.total:.1f} / 100")
    print(f"{'─'*50}")
    print(f"  Similarity : {bd.similarity_score:.1f} / {WEIGHT_SIMILARITY}")
    print(f"  Metadata   : {bd.metadata_score:.1f} / {WEIGHT_METADATA}")
    print(f"  Static     : {bd.static_score:.1f} / {WEIGHT_STATIC}")
    print(f"{'─'*50}")
    print(f"  Verdict    : {bd.verdict_reason}")
    print()