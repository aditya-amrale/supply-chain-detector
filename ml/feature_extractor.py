"""
Converts a (Package, source_files) pair into a fixed-length numeric
feature vector suitable for scikit-learn classifiers.

Feature groups (17 features total):
  Group A — Name similarity  (3 features)
  Group B — Metadata         (8 features)
  Group C — Static/AST       (6 features)

Run standalone:
  python -m ml.feature_extractor --name reqests --registry pypi
"""

import ast
import math
import os
import re
import sys
import unicodedata
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Pure-Python Levenshtein ───────────────────────────────────────────────

def _lev(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j-1] if a[i-1] == b[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]


# ── Reference package list ────────────────────────────────────────────────

POPULAR = [
    "requests","numpy","pandas","flask","django","fastapi","sqlalchemy",
    "pytest","boto3","pydantic","click","rich","httpx","aiohttp","pillow",
    "scipy","matplotlib","tensorflow","torch","scikit-learn","cryptography",
    "paramiko","celery","redis","psycopg2","pymongo","uvicorn","gunicorn",
    "express","react","lodash","axios","moment","webpack","typescript",
    "eslint","jest","mocha","chalk","commander","dotenv","cors","helmet",
    "morgan","mongoose","passport","jsonwebtoken","bcrypt","redis","pg",
]

SUSPICIOUS_SUFFIXES = [
    "-internal","-private","-local","-test","-dev",
    "_internal","_private","_local","_test","_dev",
]

HOMOGLYPHS = {
    "0":"o","1":"l","3":"e","4":"a","5":"s",
    "6":"g","7":"t","8":"b","ü":"u","ö":"o",
    "ä":"a","é":"e","ñ":"n","ç":"c","í":"i",
}

# AST patterns
EXEC_BUILTINS  = {"eval","exec","compile","__import__","execfile"}
SHELL_FUNCS    = {"subprocess.run","subprocess.call","subprocess.Popen",
                  "subprocess.check_output","os.system","os.popen","os.execv"}
NET_MODULES    = {"socket","requests","urllib","aiohttp","httpx","http"}
CRED_FUNCS     = {"os.environ.get","os.getenv"}
B64_FUNCS      = {"base64.b64decode","base64.decodebytes","base64.decodestring"}
SENSITIVE_PATHS= {".ssh",".aws",".npmrc",".pypirc","/etc/passwd","id_rsa","credentials"}


# ── Feature vector definition ─────────────────────────────────────────────

FEATURE_NAMES = [
    # Group A — Name similarity
    "min_lev_distance",          # 0  minimum Levenshtein distance to any popular pkg
    "has_separator_swap",        # 1  1 if a dash/underscore swap matches a popular pkg
    "has_suspicious_suffix",     # 2  1 if name ends with -internal, -private, etc.

    # Group B — Metadata
    "publisher_account_age_days",# 3  days since publisher account was created
    "publisher_pkg_count",       # 4  total packages published by this account
    "package_age_days",          # 5  days since package was first published
    "version_major",             # 6  major version number (≥2 is a dep-confusion flag)
    "has_repository",            # 7  1 if a repo URL is present
    "description_length",        # 8  character length of description
    "weekly_downloads_log",      # 9  log10(weekly_downloads + 1)
    "name_length",               # 10 character length of package name

    # Group C — Static / AST
    "has_shell_exec",            # 11 1 if shell execution call found
    "has_env_access",            # 12 1 if env variable access found
    "has_network_call",          # 13 1 if outbound network call found
    "has_b64_decode",            # 14 1 if runtime base64 decode found
    "has_sensitive_file_read",   # 15 1 if sensitive file path opened
    "ast_finding_count",         # 16 total number of AST findings
]

N_FEATURES = len(FEATURE_NAMES)


# ── Dataclass for a labelled sample ──────────────────────────────────────

@dataclass
class Sample:
    name:       str
    registry:   str
    label:      int         # 1 = malicious, 0 = benign
    features:   list[float]


# ── Extractor ─────────────────────────────────────────────────────────────

class FeatureExtractor:
    def extract(self, pkg: dict, source_files: dict[str, str] | None = None) -> list[float]:
        name        = (pkg.get("name") or "").lower().strip()
        version     = pkg.get("version") or "0.0.0"
        description = pkg.get("description") or ""
        repo_url    = pkg.get("repository_url") or ""
        source_files= source_files or {}

        fv = [0.0] * N_FEATURES

        # ── Group A: Name similarity ───────────────────────────────────────
        if name in POPULAR:
            fv[0] = 0.0
        else:
            fv[0] = float(min(_lev(name, p) for p in POPULAR))

        sep_variants = set()
        for old, new in [("-","_"),("_","-"),("-",""),("_","")]:
            sep_variants.add(name.replace(old, new))
        sep_variants.discard(name)
        fv[1] = 1.0 if any(v in POPULAR for v in sep_variants) else 0.0
        fv[2] = 1.0 if any(name.endswith(s) for s in SUSPICIOUS_SUFFIXES) else 0.0

        # ── Group B: Metadata ──────────────────────────────────────────────
        fv[3]  = float(pkg.get("publisher_account_age_days") or 999)
        fv[4]  = float(pkg.get("publisher_pkg_count") or 1)
        fv[5]  = float(pkg.get("package_age_days") or 0)

        try:
            fv[6] = float(version.split(".")[0])
        except (ValueError, IndexError):
            fv[6] = 0.0

        fv[7]  = 1.0 if repo_url.strip() else 0.0
        fv[8]  = float(len(description))

        dl_val = float(pkg.get("weekly_downloads") or 0)
        fv[9]  = math.log10(max(0.0, dl_val) + 1.0)
        fv[10] = float(len(name))

        # ── Group C: Static / AST ──────────────────────────────────────────
        ast_results = self._run_ast(source_files)
        fv[11] = 1.0 if ast_results["shell_exec"]   else 0.0
        fv[12] = 1.0 if ast_results["env_access"]   else 0.0
        fv[13] = 1.0 if ast_results["network_call"] else 0.0
        fv[14] = 1.0 if ast_results["b64_decode"]   else 0.0
        fv[15] = 1.0 if ast_results["sensitive_file"]else 0.0
        fv[16] = float(ast_results["finding_count"])

        return fv

    def _run_ast(self, source_files: dict[str, str]) -> dict:
        result = {
            "shell_exec": False, "env_access": False,
            "network_call": False, "b64_decode": False,
            "sensitive_file": False, "finding_count": 0,
        }

        for filename, source in source_files.items():
            if not filename.endswith(".py"):
                self._regex_scan(source, result)
                continue
            try:
                tree = ast.parse(source, filename=filename)
            except SyntaxError:
                self._regex_scan(source, result)
                continue

            imports: dict[str, str] = {}

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports[alias.asname or alias.name.split(".")[0]] = alias.name
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    for alias in node.names:
                        imports[alias.asname or alias.name] = f"{mod}.{alias.name}"
                elif isinstance(node, ast.Call):
                    fn = self._resolve_call(node.func, imports)

                    if fn in EXEC_BUILTINS:
                        result["shell_exec"] = True
                        result["finding_count"] += 1

                    elif any(fn == s or fn.startswith(s+".") for s in SHELL_FUNCS):
                        result["shell_exec"] = True
                        result["finding_count"] += 1

                    elif fn in CRED_FUNCS:
                        result["env_access"] = True
                        result["finding_count"] += 1

                    elif any(fn.startswith(m) for m in NET_MODULES):
                        result["network_call"] = True
                        result["finding_count"] += 1

                    elif fn in B64_FUNCS:
                        result["b64_decode"] = True
                        result["finding_count"] += 1

                    elif fn == "open":
                        if node.args and isinstance(node.args[0], ast.Constant):
                            path = str(node.args[0].value)
                            if any(s in path for s in SENSITIVE_PATHS):
                                result["sensitive_file"] = True
                                result["finding_count"] += 1

        return result

    def _resolve_call(self, func_node, imports: dict) -> str:
        try:
            if isinstance(func_node, ast.Name):
                return imports.get(func_node.id, func_node.id)
            if isinstance(func_node, ast.Attribute):
                obj = self._resolve_call(func_node.value, imports)
                return f"{obj}.{func_node.attr}"
        except Exception:
            pass
        return ""

    def _regex_scan(self, source: str, result: dict):
        if re.search(r"child_process|exec\s*\(|eval\s*\(|execSync", source):
            result["shell_exec"] = True
            result["finding_count"] += 1
        if re.search(r"process\.env|os\.environ", source):
            result["env_access"] = True
            result["finding_count"] += 1
        if re.search(r"fetch\s*\(|http\.get|https\.get|axios\.", source):
            result["network_call"] = True
            result["finding_count"] += 1
        if re.search(r"atob\s*\(|Buffer\.from.*base64", source):
            result["b64_decode"] = True
            result["finding_count"] += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features for a package")
    parser.add_argument("--name",     required=True)
    parser.add_argument("--registry", default="pypi")
    args = parser.parse_args()

    extractor = FeatureExtractor()
    pkg = {
        "name": args.name,
        "version": "1.0.0",
        "description": "",
        "repository_url": "",
        "publisher_account_age_days": 5,
        "publisher_pkg_count": 1,
        "package_age_days": 2,
        "weekly_downloads": 0,
    }
    fv = extractor.extract(pkg)

    print(f"\nFeature vector for '{args.name}':\n")
    for i, (feat_name, val) in enumerate(zip(FEATURE_NAMES, fv)):
        bar = "█" * int(min(val, 20))
        print(f"  {i:>2}  {feat_name:<32} {val:>8.2f}  {bar}")
    print()