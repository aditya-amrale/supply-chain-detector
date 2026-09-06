"""
Fetches full metadata for a package from npm or PyPI
and saves it as a Package row in the database.

Run standalone:
  python -m monitor.package_fetcher --registry pypi --name requests
  python -m monitor.package_fetcher --registry npm  --name express
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

import aiohttp

from db.models import Package, Registry, init_db, get_session

logger = logging.getLogger(__name__)


class PackageFetcher:
    """
    Fetches full metadata for a single package and upserts it into the DB.
    """

    PYPI_URL = "https://pypi.org/pypi/{name}/json"
    NPM_URL  = "https://registry.npmjs.org/{name}"

    def __init__(self, db_engine):
        self.engine = db_engine

    # ── Public API ─────────────────────────────────────────────────────────

    async def fetch_and_save(self, registry: str, name: str) -> Package | None:
        """Fetch metadata and upsert a Package row. Returns the Package."""
        async with aiohttp.ClientSession() as session:
            reg_lower = registry.lower().strip()
            if reg_lower == "pypi":
                pkg_data = await self._fetch_pypi(session, name)
            elif reg_lower == "npm":
                pkg_data = await self._fetch_npm(session, name)
            else:
                raise ValueError(f"Unknown registry: {registry}")

        if pkg_data is None:
            return None

        return self._upsert_package(pkg_data)

    # ── PyPI ───────────────────────────────────────────────────────────────

    async def _fetch_pypi(self, session: aiohttp.ClientSession, name: str) -> dict | None:
        url = self.PYPI_URL.format(name=name)
        logger.info(f"[pypi] Fetching {url}")

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 404:
                    logger.warning(f"[pypi] Package not found: {name}")
                    return None
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            logger.error(f"[pypi] Fetch failed for {name}: {exc}")
            return None

        info = data.get("info", {})

        first_published = None
        releases = data.get("releases", {})
        all_dates = []
        for version_files in releases.values():
            for f in version_files:
                upload_time = f.get("upload_time")
                if upload_time:
                    try:
                        dt = datetime.fromisoformat(upload_time)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        all_dates.append(dt)
                    except ValueError:
                        pass
        if all_dates:
            first_published = min(all_dates)

        return {
            "registry":        Registry.PYPI,
            "name":            info.get("name", name),
            "version":         info.get("version"),
            "author":          info.get("author"),
            "description":     info.get("summary"),
            "homepage":        info.get("home_page") or info.get("project_url"),
            "repository_url":  _find_repo(info.get("project_urls", {})),
            "publisher_email": info.get("author_email"),
            "first_published": first_published,
            "last_updated":    None,
            "raw_metadata":    json.dumps(info),
        }

    # ── npm ────────────────────────────────────────────────────────────────

    async def _fetch_npm(self, session: aiohttp.ClientSession, name: str) -> dict | None:
        url = self.NPM_URL.format(name=name)
        logger.info(f"[npm] Fetching {url}")

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 404:
                    logger.warning(f"[npm] Package not found: {name}")
                    return None
                resp.raise_for_status()
                data = await resp.json(content_type=None)
        except Exception as exc:
            logger.error(f"[npm] Fetch failed for {name}: {exc}")
            return None

        latest_version = (data.get("dist-tags") or {}).get("latest")
        latest_meta    = (data.get("versions") or {}).get(latest_version, {})

        time_data      = data.get("time", {})
        first_pub_str  = time_data.get("created")
        last_upd_str   = time_data.get("modified")

        def _parse_dt(s):
            if not s:
                return None
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                return None

        repo = latest_meta.get("repository", {})
        repo_url = repo.get("url", "") if isinstance(repo, dict) else ""

        return {
            "registry":        Registry.NPM,
            "name":            data.get("name", name),
            "version":         latest_version,
            "author":          _extract_npm_author(data.get("author") or latest_meta.get("author")),
            "description":     data.get("description") or latest_meta.get("description"),
            "homepage":        data.get("homepage") or latest_meta.get("homepage"),
            "repository_url":  repo_url,
            "publisher_email": None,
            "first_published": _parse_dt(first_pub_str),
            "last_updated":    _parse_dt(last_upd_str),
            "raw_metadata":    json.dumps({
                "name": data.get("name"),
                "dist-tags": data.get("dist-tags"),
                "keywords": data.get("keywords"),
                "license": latest_meta.get("license"),
            }),
        }

    # ── DB upsert ──────────────────────────────────────────────────────────

    def _upsert_package(self, data: dict) -> Package:
        with get_session(self.engine) as session:
            existing = (
                session.query(Package)
                .filter_by(registry=data["registry"], name=data["name"])
                .first()
            )

            if existing:
                for k, v in data.items():
                    if v is not None:
                        setattr(existing, k, v)
                pkg = existing
                logger.info(f"Updated {pkg}")
            else:
                pkg = Package(**data)
                session.add(pkg)
                logger.info(f"Inserted {pkg}")

            session.commit()
            session.refresh(pkg)
            return pkg


# ── Helpers ────────────────────────────────────────────────────────────────

def _find_repo(project_urls: dict) -> str | None:
    for key, url in (project_urls or {}).items():
        if "github" in url.lower() or "gitlab" in url.lower():
            return url
    return None


def _extract_npm_author(author) -> str | None:
    if isinstance(author, str):
        return author
    if isinstance(author, dict):
        return author.get("name")
    return None


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch metadata for a single package")
    parser.add_argument("--registry", choices=["npm", "pypi"], required=True)
    parser.add_argument("--name", required=True, help="Package name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    engine  = init_db()
    fetcher = PackageFetcher(db_engine=engine)
    pkg     = asyncio.run(fetcher.fetch_and_save(args.registry, args.name))

    if pkg:
        print(f"\nSaved: {pkg}")
        print(f"  version : {pkg.version}")
        print(f"  author  : {pkg.author}")
        print(f"  desc    : {pkg.description[:80] if pkg.description else 'n/a'}")
    else:
        print("Package not found or fetch failed.")