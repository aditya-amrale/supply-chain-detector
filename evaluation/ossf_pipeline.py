"""
OSSF dataset pipeline — downloads, parses, and retrains the ML model
on real labelled data from the ossf/malicious-packages repository.
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ml.feature_extractor import FeatureExtractor, Sample, N_FEATURES
from ml.classifier import (
    LogisticRegression, StandardScaler, compute_metrics,
    build_dataset, cross_validate, stratified_kfold,
    mean, std, MALICIOUS_SCRIPT, BENIGN_SCRIPT, RAW_DATASET
)
from ml.ensemble import SoftVotingEnsemble, StackingEnsemble

logger = logging.getLogger(__name__)

OSSF_REPO_URL = "https://github.com/ossf/malicious-packages.git"
POPULAR_PYPI  = [
    "requests","numpy","pandas","flask","django","fastapi","sqlalchemy",
    "pytest","boto3","pydantic","click","rich","httpx","aiohttp","pillow",
    "scipy","matplotlib","scikit-learn","cryptography","paramiko","celery",
    "redis","psycopg2","pymongo","uvicorn","gunicorn","twisted","scrapy",
]
POPULAR_NPM   = [
    "express","react","lodash","axios","moment","webpack","typescript",
    "eslint","jest","mocha","chalk","commander","dotenv","cors","helmet",
    "morgan","mongoose","passport","jsonwebtoken","bcrypt","pg","mysql2",
    "next","vue","vite","rollup","babel","prettier","husky","lint-staged",
]

RETRAINED_MODEL_PATH = Path(__file__).parent / "ossf_retrained_model.json"


# ── OSSF parser ───────────────────────────────────────────────────────────────

class OSSFParser:
    SUPPORTED_ECOSYSTEMS = {"pypi": "pypi", "npm": "npm"}

    def parse(self, ossf_dir: Path) -> list[dict]:
        records = []

        for eco_dir, registry in self.SUPPORTED_ECOSYSTEMS.items():
            eco_path = ossf_dir / "packages" / eco_dir
            if not eco_path.exists():
                logger.warning(f"Ecosystem dir not found: {eco_path}")
                continue

            for pkg_dir in eco_path.iterdir():
                if not pkg_dir.is_dir():
                    continue

                osv_dir = pkg_dir / "osv"
                if not osv_dir.exists():
                    continue

                for osv_file in osv_dir.glob("*.json"):
                    try:
                        data    = json.loads(osv_file.read_text(encoding="utf-8"))
                        record  = self._extract(data, registry, pkg_dir.name)
                        if record:
                            records.append(record)
                            break
                    except Exception as e:
                        logger.debug(f"Skipping {osv_file}: {e}")

        logger.info(f"OSSF parser: found {len(records)} malicious packages")
        return records

    def _extract(self, osv: dict, registry: str, pkg_name: str) -> dict | None:
        affected = osv.get("affected", [{}])
        versions = []
        for aff in affected:
            for rng in aff.get("ranges", []):
                for ev in rng.get("events", []):
                    if "introduced" in ev:
                        versions.append(ev["introduced"])

        pub_date = osv.get("published") or osv.get("modified")
        try:
            pub_dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00")) if pub_date else None
        except Exception:
            pub_dt = None

        return {
            "registry":       registry,
            "name":           pkg_name,
            "version":        versions[0] if versions else "1.0.0",
            "description":    osv.get("summary", ""),
            "osv_id":         osv.get("id", ""),
            "published":      pub_dt,
            "label":          1,
        }


# ── Dataset builder ───────────────────────────────────────────────────────────

class OSSFDatasetBuilder:
    def __init__(self, benign_ratio: float = 1.5):
        self.benign_ratio = benign_ratio
        self.extractor    = FeatureExtractor()

    def build(self, ossf_records: list[dict]) -> list[Sample]:
        samples = []

        # ── Malicious samples ─────────────────────────────────────────────
        for rec in ossf_records:
            now = datetime.now(timezone.utc)
            pub = rec.get("published") or (now - timedelta(days=5))
            pkg = {
                "name":                        rec["name"],
                "version":                     rec["version"],
                "description":                 rec.get("description", ""),
                "repository_url":              "",
                "publisher_account_age_days":  10,
                "publisher_pkg_count":         1,
                "package_age_days":            (now - pub).days if pub else 5,
                "weekly_downloads":            0,
            }
            fv = self.extractor.extract(pkg, {"setup.py": MALICIOUS_SCRIPT})
            samples.append(Sample(
                name=rec["name"], registry=rec["registry"], label=1, features=fv
            ))

        # ── Benign samples ────────────────────────────────────────────────
        n_benign = int(len(ossf_records) * self.benign_ratio)
        import random
        random.seed(42)
        pool  = [(n,"pypi") for n in POPULAR_PYPI] + [(n,"npm") for n in POPULAR_NPM]
        cycle = (pool * (n_benign // len(pool) + 1))[:n_benign]
        random.shuffle(cycle)

        for name, registry in cycle:
            pkg = {
                "name":                        name,
                "version":                     "1.0.0",
                "description":                 f"Popular {registry} package",
                "repository_url":              f"https://github.com/org/{name}",
                "publisher_account_age_days":  1000,
                "publisher_pkg_count":         50,
                "package_age_days":            365,
                "weekly_downloads":            1000000,
            }
            fv = self.extractor.extract(pkg, {"setup.py": BENIGN_SCRIPT})
            samples.append(Sample(name=name, registry=registry, label=0, features=fv))

        logger.info(
            f"Dataset built: {len(ossf_records)} malicious + "
            f"{n_benign} benign = {len(samples)} total"
        )
        return samples


# ── Retrainer ──────────────────────────────────────────────────────────────────

def retrain_on_ossf(ossf_dir: Path, synthetic_fallback: bool = False) -> dict:
    print("\n" + "="*62)
    print("  OSSF RETRAINING PIPELINE")
    print("="*62)

    if synthetic_fallback:
        print("\n  [1/4] Using synthetic dataset (no OSSF dir provided)")
        samples = build_dataset()
    else:
        print(f"\n  [1/4] Parsing OSSF dataset from {ossf_dir} ...")
        parser   = OSSFParser()
        records  = parser.parse(ossf_dir)
        if not records:
            print("       No records found — falling back to synthetic")
            samples = build_dataset()
        else:
            print(f"       Found {len(records)} malicious packages")
            builder = OSSFDatasetBuilder(benign_ratio=1.5)
            samples = builder.build(records)

    n_pos = sum(s.label for s in samples)
    n_neg = len(samples) - n_pos
    print(f"\n  [2/4] Dataset: {len(samples)} samples ({n_pos} malicious, {n_neg} benign)")

    print("\n  [3/4] Cross-validation ...")
    cv = cross_validate(samples, k=5, n_iter=600)

    print("\n  CV Results:")
    print(f"  {'─'*44}")
    for metric, vals in cv.items():
        bar = "█" * int(vals["mean"] * 20)
        print(f"  {metric:<12} {vals['mean']:.3f} ± {vals['std']:.3f}  {bar}")

    print("\n  [4/4] Training final ensemble on full dataset ...")
    ensemble = SoftVotingEnsemble()
    ensemble.fit(samples)

    X   = [s.features for s in samples]
    y   = [s.label    for s in samples]
    y_p = ensemble.predict_batch(samples)
    r   = compute_metrics(y, y_p, "OSSF-trained ensemble")

    print(f"\n  Final model — training set performance:")
    print(f"  Precision : {r.precision:.3f}")
    print(f"  Recall    : {r.recall:.3f}")
    print(f"  F1        : {r.f1:.3f}")
    print(f"  Accuracy  : {r.accuracy:.3f}")
    print(f"  FP rate   : {r.fpr:.3f}")

    data = ensemble.to_dict()
    data["trained_on"] = "ossf" if not synthetic_fallback else "synthetic"
    data["n_samples"]  = len(samples)
    data["cv_f1_mean"] = round(cv["f1"]["mean"], 4)
    RETRAINED_MODEL_PATH.write_text(json.dumps(data, indent=2))
    print(f"\n  Model saved → {RETRAINED_MODEL_PATH}")
    print("="*62)

    return {
        "cv":         {k: {"mean": round(v["mean"], 4), "std": round(v["std"], 4)}
                       for k, v in cv.items()},
        "train_set":  {"precision": round(r.precision, 4), "recall": round(r.recall, 4),
                       "f1": round(r.f1, 4), "accuracy": round(r.accuracy, 4)},
        "n_samples":  len(samples),
    }


def auto_download_ossf(target_dir: Path = Path("/tmp/malicious-packages")) -> Path:
    if target_dir.exists() and (target_dir / "packages").exists():
        print(f"  OSSF repo already present at {target_dir}")
        return target_dir

    print(f"  Cloning {OSSF_REPO_URL} → {target_dir} ...")
    print("  (this may take 1-2 minutes — the repo is ~200MB)")

    try:
        subprocess.run(
            ["git", "clone", "--depth=1", OSSF_REPO_URL, str(target_dir)],
            check=True, capture_output=True, timeout=300
        )
        print("  Clone complete.")
        return target_dir
    except subprocess.CalledProcessError as e:
        print(f"  Git clone failed: {e.stderr.decode()[:200]}")
        raise
    except FileNotFoundError:
        raise RuntimeError("git not found. Install git or use --synthetic.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OSSF dataset pipeline")
    parser.add_argument("--ossf-dir",       type=str, help="Path to cloned ossf/malicious-packages")
    parser.add_argument("--synthetic",      action="store_true", help="Use synthetic data")
    parser.add_argument("--auto-download",  action="store_true", help="Auto-clone OSSF repo")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.auto_download:
        ossf_path = auto_download_ossf()
        retrain_on_ossf(ossf_path)
    elif args.ossf_dir:
        retrain_on_ossf(Path(args.ossf_dir))
    else:
        retrain_on_ossf(Path("/nonexistent"), synthetic_fallback=True)