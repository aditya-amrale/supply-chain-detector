"""
Detects typosquatting by comparing a package name against a
reference list of popular/known-good packages.

Techniques used:
  1. Levenshtein distance  — catches single char typos (reqests vs requests)
  2. Soundex phonetic hash — catches phonetic substitutions
  3. Homoglyph detection   — catches unicode lookalikes (reqüests vs requests)
  4. Separator variants    — catches dash/underscore/dot swaps

Run standalone:
  python -m detectors.name_similarity --name reqests --registry pypi
"""

import logging
import unicodedata
from dataclasses import dataclass

import jellyfish
from rapidfuzz.distance.Levenshtein import distance as lev_distance

logger = logging.getLogger(__name__)

# ── Homoglyph map (common lookalikes) ─────────────────────────────────────

HOMOGLYPHS: dict[str, str] = {
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
    "6": "g", "7": "t", "8": "b", "ü": "u", "ö": "o",
    "ä": "a", "é": "e", "ñ": "n", "ç": "c", "í": "i",
}


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class SimilarityResult:
    is_suspicious:      bool
    signal_value:       float     # 0.0–1.0 (1.0 = very suspicious)
    closest_match:      str | None
    closest_distance:   int
    match_method:       str | None    # "levenshtein" | "soundex" | "homoglyph" | "separator"
    detail:             str


# ── Detector ───────────────────────────────────────────────────────────────

class NameSimilarityDetector:
    """
    Compares a candidate package name against a reference list.
    """

    MIN_NAME_LEN = 3

    def __init__(self, popular_packages: list[str]):
        self.popular = [p.lower().strip() for p in popular_packages if p.strip()]
        self._soundex_map = {p: jellyfish.soundex(p) for p in self.popular}
        logger.info(f"NameSimilarityDetector loaded {len(self.popular)} reference packages")

    def check(self, candidate: str) -> SimilarityResult:
        c = candidate.lower().strip()

        if len(c) < self.MIN_NAME_LEN:
            return SimilarityResult(
                is_suspicious=False, signal_value=0.0,
                closest_match=None, closest_distance=99,
                match_method=None, detail="name too short to check"
            )

        if c in self.popular:
            return SimilarityResult(
                is_suspicious=False, signal_value=0.0,
                closest_match=c, closest_distance=0,
                match_method=None, detail="name is in known-good list"
            )

        # 1. Homoglyph check
        homoglyph_res = self._check_homoglyph(c)
        if homoglyph_res.is_suspicious:
            return homoglyph_res

        # 2. Separator swap check
        separator_res = self._check_separator(c)
        if separator_res.is_suspicious:
            return separator_res

        # 3. Levenshtein check
        lev_res = self._check_levenshtein(c)

        # 4. Soundex check
        soundex_res = self._check_soundex(c)

        return max([lev_res, soundex_res], key=lambda r: r.signal_value)

    # ── Individual checks ──────────────────────────────────────────────────

    def _check_levenshtein(self, candidate: str) -> SimilarityResult:
        best_match = None
        best_dist  = 999

        for ref in self.popular:
            if len(ref) < self.MIN_NAME_LEN:
                continue
            d = lev_distance(candidate, ref)
            if d < best_dist:
                best_dist  = d
                best_match = ref
            if d == 1:
                break

        if best_dist == 0:
            signal = 0.0
        elif best_dist == 1:
            signal = 0.95
        elif best_dist == 2:
            signal = 0.75
        elif best_dist == 3:
            signal = 0.50
        else:
            signal = 0.0

        return SimilarityResult(
            is_suspicious   = signal >= 0.5,
            signal_value    = signal,
            closest_match   = best_match,
            closest_distance= best_dist,
            match_method    = "levenshtein" if signal > 0 else None,
            detail          = f"levenshtein({candidate}, {best_match}) = {best_dist}"
        )

    def _check_homoglyph(self, candidate: str) -> SimilarityResult:
        normalized = "".join(HOMOGLYPHS.get(ch, ch) for ch in candidate)
        nfkd = unicodedata.normalize("NFKD", candidate)
        ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))

        if normalized == candidate and ascii_only == candidate:
            return SimilarityResult(
                is_suspicious=False, signal_value=0.0, closest_match=None,
                closest_distance=99, match_method=None,
                detail="no homoglyphs detected"
            )

        for variant in {normalized, ascii_only}:
            if variant != candidate and variant in self.popular:
                return SimilarityResult(
                    is_suspicious   = True,
                    signal_value    = 0.90,
                    closest_match   = variant,
                    closest_distance= lev_distance(candidate, variant),
                    match_method    = "homoglyph",
                    detail          = f"homoglyph: '{candidate}' normalizes to '{variant}'"
                )

        return SimilarityResult(
            is_suspicious=False, signal_value=0.0, closest_match=None,
            closest_distance=99, match_method=None,
            detail="homoglyph normalization found no match"
        )

    def _check_separator(self, candidate: str) -> SimilarityResult:
        variants = set()
        for old, new in [("-", "_"), ("_", "-"), ("-", ""), ("_", ""), (".", "-"), (".", "_")]:
            variants.add(candidate.replace(old, new))
        variants.discard(candidate)

        for v in variants:
            if v in self.popular:
                return SimilarityResult(
                    is_suspicious   = True,
                    signal_value    = 0.80,
                    closest_match   = v,
                    closest_distance= lev_distance(candidate, v),
                    match_method    = "separator",
                    detail          = f"separator swap: '{candidate}' → '{v}' matches popular package"
                )

        return SimilarityResult(
            is_suspicious=False, signal_value=0.0, closest_match=None,
            closest_distance=99, match_method=None,
            detail="no separator variant matched"
        )

    def _check_soundex(self, candidate: str) -> SimilarityResult:
        candidate_soundex = jellyfish.soundex(candidate)

        for ref, ref_soundex in self._soundex_map.items():
            if candidate_soundex == ref_soundex and ref != candidate:
                len_ratio = min(len(candidate), len(ref)) / max(len(candidate), len(ref))
                if len_ratio > 0.6:
                    return SimilarityResult(
                        is_suspicious   = True,
                        signal_value    = 0.55,
                        closest_match   = ref,
                        closest_distance= lev_distance(candidate, ref),
                        match_method    = "soundex",
                        detail          = f"soundex match: '{candidate}' ~ '{ref}' ({candidate_soundex})"
                    )

        return SimilarityResult(
            is_suspicious=False, signal_value=0.0, closest_match=None,
            closest_distance=99, match_method=None,
            detail="no soundex match"
        )


def load_popular_packages(filepath: str) -> list[str]:
    try:
        with open(filepath) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        logger.warning(f"Popular packages file not found: {filepath}")
        return []


POPULAR_PYPI = [
    "requests", "numpy", "pandas", "flask", "django", "fastapi", "sqlalchemy",
    "pytest", "boto3", "pydantic", "click", "rich", "httpx", "aiohttp",
    "pillow", "scipy", "matplotlib", "tensorflow", "torch", "scikit-learn",
    "cryptography", "paramiko", "fabric", "celery", "redis", "psycopg2",
    "pymongo", "yaml", "toml", "dotenv", "uvicorn", "gunicorn", "twisted",
    "scrapy", "beautifulsoup4", "lxml", "selenium", "playwright", "pytest-asyncio",
]

POPULAR_NPM = [
    "express", "react", "lodash", "axios", "moment", "webpack", "babel",
    "typescript", "eslint", "prettier", "jest", "mocha", "chai", "sinon",
    "commander", "chalk", "dotenv", "cors", "helmet", "morgan", "mongoose",
    "sequelize", "socket.io", "nodemailer", "passport", "jsonwebtoken",
    "bcrypt", "multer", "sharp", "redis", "pg", "mysql2", "sqlite3",
    "next", "nuxt", "vue", "angular", "svelte", "vite", "rollup",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check a package name for typosquatting")
    parser.add_argument("--name",     required=True)
    parser.add_argument("--registry", choices=["npm", "pypi"], default="pypi")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ref_list = POPULAR_PYPI if args.registry == "pypi" else POPULAR_NPM
    detector = NameSimilarityDetector(popular_packages=ref_list)
    result   = detector.check(args.name)

    print(f"\nCandidate : {args.name}")
    print(f"Suspicious: {result.is_suspicious}")
    print(f"Signal    : {result.signal_value:.2f}")
    print(f"Closest   : {result.closest_match}  (dist={result.closest_distance})")
    print(f"Method    : {result.match_method}")
    print(f"Detail    : {result.detail}")