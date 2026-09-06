"""
Scores a package based on metadata red flags.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from db.models import Package

logger = logging.getLogger(__name__)


@dataclass
class MetadataSignal:
    name:        str
    fired:       bool
    weight:      float
    contribution: float
    detail:      str


@dataclass
class MetadataResult:
    signals:      list[MetadataSignal] = field(default_factory=list)
    total_score:  float = 0.0

    def add(self, signal: MetadataSignal):
        self.signals.append(signal)
        if signal.fired:
            self.total_score += signal.contribution

    @property
    def signal_value(self) -> float:
        max_possible = sum(s.weight for s in self.signals)
        if max_possible == 0:
            return 0.0
        return min(self.total_score / max_possible, 1.0)

    def summary(self) -> str:
        fired = [s.name for s in self.signals if s.fired]
        return ", ".join(fired) if fired else "no flags"


class MetadataHeuristicsDetector:
    NEW_ACCOUNT_DAYS    = 30
    NEW_PACKAGE_DAYS    = 7
    SHORT_DESC_CHARS    = 20

    SUSPICIOUS_SUFFIXES = [
        "-internal", "-private", "-local", "-test", "-dev",
        "_internal", "_private", "_local", "_test", "_dev",
    ]

    def check(self, pkg: Package) -> MetadataResult:
        result = MetadataResult()
        now = datetime.now(timezone.utc)

        def _aware(dt):
            if dt is None:
                return None
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        # 1. New publisher account
        pub_created = _aware(pkg.publisher_created)
        if pub_created:
            age_days = (now - pub_created).days
            fired = age_days < self.NEW_ACCOUNT_DAYS
            result.add(MetadataSignal(
                name="new_publisher", fired=fired, weight=20,
                contribution=20 if fired else 0,
                detail=f"publisher account is {age_days} days old"
            ))
        else:
            result.add(MetadataSignal(
                name="new_publisher", fired=False, weight=20,
                contribution=0, detail="publisher creation date unknown"
            ))

        # 2. First-time author
        total_pkgs = pkg.total_packages_by_publisher or 0
        fired = total_pkgs <= 1
        result.add(MetadataSignal(
            name="first_time_author", fired=fired, weight=15,
            contribution=15 if fired else 0,
            detail=f"publisher has {total_pkgs} total package(s)"
        ))

        # 3. Empty or short description
        desc = (pkg.description or "").strip()
        fired = len(desc) < self.SHORT_DESC_CHARS
        result.add(MetadataSignal(
            name="no_readme", fired=fired, weight=10,
            contribution=10 if fired else 0,
            detail=f"description length: {len(desc)} chars"
        ))

        # 4. High version jump
        fired = False
        detail = "version not available"
        if pkg.version:
            try:
                major = int(pkg.version.split(".")[0])
                fired = major >= 2
                detail = f"first published version is {pkg.version}"
            except (ValueError, IndexError):
                detail = f"could not parse version: {pkg.version}"
        result.add(MetadataSignal(
            name="high_version_jump", fired=fired, weight=25,
            contribution=25 if fired else 0,
            detail=detail
        ))

        # 5. No linked repository
        fired = not bool(pkg.repository_url and pkg.repository_url.strip())
        result.add(MetadataSignal(
            name="no_repository", fired=fired, weight=10,
            contribution=10 if fired else 0,
            detail="no repository URL" if fired else f"repo: {pkg.repository_url}"
        ))

        # 6. Suspicious name suffix
        name_lower = (pkg.name or "").lower()
        matched_suffix = next((s for s in self.SUSPICIOUS_SUFFIXES if name_lower.endswith(s)), None)
        fired = matched_suffix is not None
        result.add(MetadataSignal(
            name="suspicious_name_suffix", fired=fired, weight=20,
            contribution=20 if fired else 0,
            detail=f"name ends with '{matched_suffix}'" if fired else "no suspicious suffix"
        ))

        # 7. Recently published
        first_pub = _aware(pkg.first_published)
        if first_pub:
            age_days = (now - first_pub).days
            fired = age_days < self.NEW_PACKAGE_DAYS
            result.add(MetadataSignal(
                name="recently_published", fired=fired, weight=5,
                contribution=5 if fired else 0,
                detail=f"package is {age_days} days old"
            ))
        else:
            result.add(MetadataSignal(
                name="recently_published", fired=False, weight=5,
                contribution=0, detail="publish date unknown"
            ))

        logger.debug(f"Metadata check for {pkg.name}: score={result.total_score:.1f} flags=[{result.summary()}]")
        return result


if __name__ == "__main__":
    import argparse
    from db.models import Registry, init_db, get_session

    parser = argparse.ArgumentParser(description="Run metadata heuristics on a DB package")
    parser.add_argument("--name",     required=True)
    parser.add_argument("--registry", choices=["npm", "pypi"], default="pypi")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    engine = init_db()
    with get_session(engine) as session:
        reg = Registry(args.registry)
        pkg = session.query(Package).filter_by(registry=reg, name=args.name).first()

    if not pkg:
        print(f"Package {args.registry}/{args.name} not in DB. Run package_fetcher first.")
    else:
        detector = MetadataHeuristicsDetector()
        result   = detector.check(pkg)
        print(f"\nMetadata signals for {pkg}:")
        for s in result.signals:
            status = "FIRED" if s.fired else "  ok "
            print(f"  [{status}] {s.name:<28} weight={s.weight:>3}  — {s.detail}")
        print(f"\nTotal score : {result.total_score:.1f}")
        print(f"Signal value: {result.signal_value:.2f}  ({result.summary()})")