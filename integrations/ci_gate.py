"""
CLI tool invoked from GitHub Actions (or any CI/CD system) to check
whether a package is safe to install.

Exit codes (standard for CI gates):
  0 → safe (LOW risk)
  1 → error / unknown
  2 → MEDIUM risk  (warns but doesn't fail by default)
  3 → HIGH risk    (always fails)
"""

import asyncio
import json
import logging
import sys
import os

import typer
from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db.models import Registry, RiskLevel, init_db, get_session
from db.models import Package, ScanResult
from monitor.package_fetcher import PackageFetcher
from detectors.name_similarity import NameSimilarityDetector, POPULAR_PYPI, POPULAR_NPM
from detectors.metadata_heuristics import MetadataHeuristicsDetector
from detectors.static_analysis import StaticAnalysisDetector
from scoring.risk_scorer import RiskScorer

app = typer.Typer(
    help="Supply Chain Attack Detector — CI/CD gate",
    no_args_is_help=True,
    invoke_without_command=False
)
console = Console()
logger  = logging.getLogger(__name__)


@app.command()
def check(
    name:      str = typer.Option(...,      help="Package name to check"),
    registry:  str = typer.Option("pypi",   help="Registry: pypi or npm"),
    fail_on:   str = typer.Option("high",   help="Fail CI on: medium or high"),
    output:    str = typer.Option("pretty", help="Output format: pretty or json"),
    db_url:    str = typer.Option("sqlite:///./supply_chain.db", help="DB URL"),
):
    """
    Fetch, scan, and score a package. Exit non-zero if risk >= fail_on threshold.
    """
    logging.basicConfig(level=logging.WARNING)

    reg_str = registry.strip().lower()
    pkg_name = name.strip()

    if reg_str not in ("npm", "pypi"):
        console.print(f"[red]Unknown registry: {registry}[/red]")
        raise typer.Exit(1)

    # ── Fetch metadata ─────────────────────────────────────────────────────
    engine  = init_db(db_url)
    fetcher = PackageFetcher(db_engine=engine)

    if output == "pretty":
        console.print(f"\n[bold]Checking[/bold] [cyan]{reg_str}/{pkg_name}[/cyan] ...")

    pkg = asyncio.run(fetcher.fetch_and_save(reg_str, pkg_name))
    if not pkg:
        _output_error(output, reg_str, pkg_name, "Package not found on registry")
        raise typer.Exit(1)

    # ── Score ──────────────────────────────────────────────────────────────
    ref_list = POPULAR_PYPI if reg_str == "pypi" else POPULAR_NPM
    scorer   = RiskScorer(
        similarity_detector = NameSimilarityDetector(ref_list),
        metadata_detector   = MetadataHeuristicsDetector(),
        static_detector     = StaticAnalysisDetector(),
        db_engine           = engine,
    )
    bd = scorer.score(pkg)

    # ── Output ─────────────────────────────────────────────────────────────
    risk_str = bd.risk_level.value if hasattr(bd.risk_level, 'value') else str(bd.risk_level)

    if output == "json":
        result = {
            "registry":    reg_str,
            "name":        pkg_name,
            "version":     pkg.version,
            "risk_level":  risk_str,
            "score":       bd.total,
            "breakdown": {
                "similarity": bd.similarity_score,
                "metadata":   bd.metadata_score,
                "static":     bd.static_score,
            },
            "verdict": bd.verdict_reason,
        }
        print(json.dumps(result, indent=2))
    else:
        _print_pretty(reg_str, pkg_name, pkg, bd)

    # ── Exit code ──────────────────────────────────────────────────────────
    fail_levels = {
        "medium": {RiskLevel.MEDIUM, RiskLevel.HIGH},
        "high":   {RiskLevel.HIGH},
    }
    should_fail = bd.risk_level in fail_levels.get(fail_on.lower(), {RiskLevel.HIGH})

    if should_fail:
        if output == "pretty":
            console.print(f"\n[bold red]BLOCKED[/bold red] — risk level [{risk_str.upper()}] exceeds threshold '{fail_on}'")
        raise typer.Exit(3 if bd.risk_level == RiskLevel.HIGH else 2)

    if output == "pretty":
        console.print(f"\n[bold green]PASSED[/bold green] — risk level [{risk_str.upper()}]")


def _print_pretty(registry, name, pkg, bd):
    risk_str = bd.risk_level.value if hasattr(bd.risk_level, 'value') else str(bd.risk_level)
    level_color = {"low": "green", "medium": "yellow", "high": "red"}.get(risk_str.lower(), "white")

    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    table.add_column("Field", style="dim", width=16)
    table.add_column("Value")

    table.add_row("Package",    f"{registry}/{name}")
    table.add_row("Version",    pkg.version or "unknown")
    table.add_row("Risk level", f"[{level_color}]{risk_str.upper()}[/{level_color}]")
    table.add_row("Score",      f"{bd.total:.1f} / 100")
    table.add_row("Similarity", f"{bd.similarity_score:.1f} / 40")
    table.add_row("Metadata",   f"{bd.metadata_score:.1f} / 35")
    table.add_row("Static",     f"{bd.static_score:.1f} / 25")
    table.add_row("Verdict",    bd.verdict_reason or "—")

    console.print()
    console.print(table)


def _output_error(output, registry, name, msg):
    if output == "json":
        print(json.dumps({"registry": registry, "name": name, "error": msg}))
    else:
        console.print(f"[red]Error:[/red] {msg}")


if __name__ == "__main__":
    app()
