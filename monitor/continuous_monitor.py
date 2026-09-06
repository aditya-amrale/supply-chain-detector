"""
Runs the full detection pipeline continuously:
  1. Poll npm + PyPI for new packages every N seconds
  2. Fetch full metadata for each new package
  3. Score it (all three detectors)
  4. Dispatch alerts for HIGH (or MEDIUM) risk findings

Run:
  python -m monitor.continuous_monitor
  python -m monitor.continuous_monitor --interval 60 --limit 20
"""

import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db.models import Package, Registry, init_db, get_session
from monitor.registry_poller import RegistryPoller
from monitor.package_fetcher import PackageFetcher
from detectors.name_similarity import NameSimilarityDetector, POPULAR_PYPI, POPULAR_NPM
from detectors.metadata_heuristics import MetadataHeuristicsDetector
from detectors.static_analysis import StaticAnalysisDetector
from scoring.risk_scorer import RiskScorer
from integrations.alert_engine import AlertEngine

logger = logging.getLogger(__name__)


class ContinuousMonitor:
    def __init__(
        self,
        interval_seconds: int = 300,
        packages_per_poll: int = 50,
        alert_on_medium: bool = False,
        registries: list[str] = None,
    ):
        self.interval         = interval_seconds
        self.packages_per_poll= packages_per_poll
        self.engine           = init_db()
        self.poller           = RegistryPoller(registries=registries or ["npm", "pypi"])
        self.fetcher          = PackageFetcher(db_engine=self.engine)
        self.alert_engine     = AlertEngine(alert_on_medium=alert_on_medium)

        all_popular = list(set(POPULAR_PYPI + POPULAR_NPM))
        self.scorer = RiskScorer(
            similarity_detector=NameSimilarityDetector(all_popular),
            metadata_detector=MetadataHeuristicsDetector(),
            static_detector=StaticAnalysisDetector(),
            db_engine=self.engine,
        )

    async def run_once(self) -> dict:
        """Run one full poll-fetch-score cycle. Returns summary stats."""
        logger.info("=== Starting detection cycle ===")
        stats = {"polled": 0, "new": 0, "scored": 0, "alerted": 0, "errors": 0}

        # Step 1: Poll
        refs = await self.poller.poll_all(limit_per_registry=self.packages_per_poll)
        stats["polled"] = len(refs)

        # Step 2: Filter out already-known packages
        new_refs = []
        with get_session(self.engine) as session:
            for ref in refs:
                reg_enum = Registry(ref.registry.lower())
                exists = session.query(Package).filter_by(
                    registry=reg_enum, name=ref.name
                ).first()
                if not exists:
                    new_refs.append(ref)

        stats["new"] = len(new_refs)
        logger.info(f"New packages to scan: {len(new_refs)} / {len(refs)}")

        if not new_refs:
            return stats

        # Step 3: Fetch + score each new package
        for ref in new_refs:
            try:
                pkg = await self.fetcher.fetch_and_save(ref.registry, ref.name)
                if not pkg:
                    stats["errors"] += 1
                    continue

                bd = self.scorer.score(pkg)
                stats["scored"] += 1

                risk_str = bd.risk_level.value if hasattr(bd.risk_level, 'value') else str(bd.risk_level)
                logger.info(
                    f"  [{ref.registry:4s}] {ref.name:40s} "
                    f"→ {risk_str.upper():6s} ({bd.total:.0f})"
                )

                # Step 4: Alert
                with get_session(self.engine) as session:
                    from db.models import ScanResult
                    result_row = session.query(ScanResult).filter_by(package_id=pkg.id).first()
                    if result_row and self.alert_engine.dispatch(pkg, result_row):
                        stats["alerted"] += 1

            except Exception as exc:
                logger.error(f"Error processing {ref.registry}/{ref.name}: {exc}")
                stats["errors"] += 1

        return stats

    def run_forever(self):
        """Block and run detection cycles indefinitely."""
        logger.info(f"Continuous monitor started (interval={self.interval}s)")

        while True:
            try:
                stats = asyncio.run(self.run_once())
                logger.info(
                    f"Cycle complete — polled={stats['polled']} new={stats['new']} "
                    f"scored={stats['scored']} alerted={stats['alerted']} errors={stats['errors']}"
                )
            except KeyboardInterrupt:
                logger.info("Stopped by user.")
                break
            except Exception as exc:
                logger.error(f"Cycle failed: {exc}")

            logger.info(f"Next cycle in {self.interval}s ...")
            time.sleep(self.interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run supply chain detector continuously")
    parser.add_argument("--interval", type=int, default=300,   help="Seconds between cycles")
    parser.add_argument("--limit",    type=int, default=50,    help="Packages per registry per cycle")
    parser.add_argument("--medium",   action="store_true",     help="Alert on MEDIUM risk too")
    parser.add_argument("--once",     action="store_true",     help="Run one cycle and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    )

    monitor = ContinuousMonitor(
        interval_seconds=args.interval,
        packages_per_poll=args.limit,
        alert_on_medium=args.medium,
    )

    if args.once:
        stats = asyncio.run(monitor.run_once())
        print(f"\nDone: {stats}")
    else:
        monitor.run_forever()