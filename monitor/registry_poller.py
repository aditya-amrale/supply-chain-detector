"""
Polls npm and PyPI for newly published packages.

How it works:
  - npm  : fetches /-/all (full package list) or uses the changes feed
  - PyPI : uses the RSS feed at https://pypi.org/rss/updates.xml
  - Results are saved to the database via PackageFetcher

Run standalone:
  python -m monitor.registry_poller --registry npm --limit 20
"""

import asyncio
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator

import aiohttp

logger = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class PackageRef:
    """Lightweight reference returned by the poller before full fetch."""
    registry: str
    name: str
    version: str | None = None
    discovered_at: datetime = None

    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = datetime.now(timezone.utc)


# ── npm poller ─────────────────────────────────────────────────────────────

class NpmPoller:
    """
    Streams new package names from the npm registry.
    """

    BASE_URL = "https://registry.npmjs.org"

    async def poll(
        self,
        session: aiohttp.ClientSession,
        limit: int = 100,
    ) -> AsyncIterator[PackageRef]:
        """Yield up to `limit` recently updated npm packages."""
        url = f"{self.BASE_URL}/-/all"
        logger.info(f"[npm] Polling {url} (limit={limit})")

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)

            count = 0
            for name, meta in data.items():
                if name.startswith("_"):
                    continue
                if count >= limit:
                    break

                version = None
                if isinstance(meta, dict):
                    dist_tags = meta.get("dist-tags", {})
                    version = dist_tags.get("latest")

                yield PackageRef(registry="npm", name=name, version=version)
                count += 1

            logger.info(f"[npm] Yielded {count} packages")

        except Exception as exc:
            logger.error(f"[npm] Poll failed: {exc}")


# ── PyPI poller ────────────────────────────────────────────────────────────

class PyPIPoller:
    """
    Streams new package names from the PyPI RSS feed.
    """

    RSS_URL = "https://pypi.org/rss/updates.xml"

    async def poll(
        self,
        session: aiohttp.ClientSession,
        limit: int = 40,
    ) -> AsyncIterator[PackageRef]:
        """Yield recently updated PyPI packages from the RSS feed."""
        logger.info(f"[pypi] Polling {self.RSS_URL}")

        try:
            async with session.get(self.RSS_URL, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                resp.raise_for_status()
                text = await resp.text()

            root = ET.fromstring(text)
            channel = root.find("channel")
            if channel is None:
                logger.warning("[pypi] No <channel> in RSS feed")
                return

            count = 0
            for item in channel.findall("item"):
                if count >= limit:
                    break

                title = item.findtext("title", "")
                parts = title.rsplit(" ", 1)
                name    = parts[0].strip() if parts else title
                version = parts[1].strip() if len(parts) == 2 else None

                if name:
                    yield PackageRef(registry="pypi", name=name, version=version)
                    count += 1

            logger.info(f"[pypi] Yielded {count} packages")

        except Exception as exc:
            logger.error(f"[pypi] Poll failed: {exc}")


# ── Unified poller ─────────────────────────────────────────────────────────

class RegistryPoller:
    def __init__(self, registries: list[str] = None):
        self.registries = registries or ["npm", "pypi"]
        self._npm  = NpmPoller()
        self._pypi = PyPIPoller()

    async def poll_all(self, limit_per_registry: int = 50) -> list[PackageRef]:
        """Poll all configured registries and return a flat list of PackageRefs."""
        refs: list[PackageRef] = []

        async with aiohttp.ClientSession() as session:
            tasks = []

            if "npm" in self.registries:
                tasks.append(self._collect(self._npm.poll(session, limit_per_registry)))

            if "pypi" in self.registries:
                tasks.append(self._collect(self._pypi.poll(session, limit_per_registry)))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Poller task failed: {result}")
                else:
                    refs.extend(result)

        logger.info(f"Total packages collected: {len(refs)}")
        return refs

    @staticmethod
    async def _collect(aiter: AsyncIterator[PackageRef]) -> list[PackageRef]:
        return [item async for item in aiter]


# ── CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Poll npm/PyPI for new packages")
    parser.add_argument("--registry", choices=["npm", "pypi", "both"], default="both")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    registries = ["npm", "pypi"] if args.registry == "both" else [args.registry]
    poller = RegistryPoller(registries=registries)
    refs   = asyncio.run(poller.poll_all(limit_per_registry=args.limit))

    print(f"\nFound {len(refs)} packages:\n")
    for r in refs[:50]:
        print(f"  [{r.registry:4s}] {r.name}  {r.version or ''}")