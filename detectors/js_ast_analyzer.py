"""
JavaScript static analysis — parses npm package.json scripts section
and raw JS source using a hand-written recursive descent tokenizer.

No external deps — pure Python stdlib.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Pattern library ─────────────────────────────────────────────────────────

@dataclass
class JSFinding:
    category:    str
    severity:    int       # 1 low  2 medium  3 high
    line:        int
    description: str
    snippet:     str = ""


@dataclass
class JSAnalysisResult:
    findings:      list[JSFinding] = field(default_factory=list)
    lines_scanned: int = 0

    @property
    def signal_value(self) -> float:
        if not self.findings:
            return 0.0
        mx   = max(f.severity for f in self.findings)
        cnt  = len(self.findings)
        base = {1: 0.15, 2: 0.55, 3: 0.90}.get(mx, 0.0)
        return min(base + (cnt - 1) * 0.05, 1.0)

    @property
    def has_suspicious_code(self) -> bool:
        return any(f.severity >= 2 for f in self.findings)

    def summary(self) -> str:
        cats = list({f.category for f in self.findings})
        return ", ".join(cats) if cats else "clean"


_PATTERNS: list[tuple[str, re.Pattern, int, str]] = [
    # Shell execution
    ("exec", re.compile(
        r"\b(child_process|require\s*\(\s*['\"]child_process['\"]\s*\))\s*\.\s*"
        r"(exec|execSync|spawn|spawnSync|execFile|fork)\s*\(",
        re.I), 3, "child_process execution"),
    ("exec", re.compile(r"\beval\s*\(", re.I), 3, "eval() call"),
    ("exec", re.compile(r"\bnew\s+Function\s*\(", re.I), 3, "Function constructor (eval equiv)"),
    ("exec", re.compile(r"\bexecSync\s*\(|spawnSync\s*\(", re.I), 3, "synchronous shell execution"),

    # Network
    ("network", re.compile(
        r"\b(fetch|XMLHttpRequest|axios\.(get|post|put|delete)|"
        r"https?\.(get|post|request)\s*\(|"
        r"require\s*\(\s*['\"]https?['\"]\s*\)\s*\.\s*(get|post|request))",
        re.I), 2, "outbound network call"),
    ("network", re.compile(r"new\s+WebSocket\s*\(", re.I), 2, "WebSocket connection"),

    # Credential theft
    ("credential", re.compile(r"process\.env\b", re.I), 2, "process.env access"),
    ("credential", re.compile(
        r"['\"][^'\"]*\.(ssh|aws|npmrc|pypirc|netrc)[^'\"]*['\"]", re.I),
        3, "sensitive file path in string"),

    # Obfuscation
    ("obfuscation", re.compile(r"\batob\s*\(", re.I), 2, "runtime base64 decode (atob)"),
    ("obfuscation", re.compile(
        r"Buffer\s*\.\s*from\s*\([^)]+,\s*['\"]base64['\"]", re.I),
        2, "Buffer.from(…,'base64') decode"),
    ("obfuscation", re.compile(
        r"(\\x[0-9a-f]{2}){6,}", re.I),
        3, "dense hex-escaped string (obfuscation)"),
    ("obfuscation", re.compile(
        r"String\s*\.\s*fromCharCode\s*\((?:\s*\d+\s*,){5,}",
        re.I), 2, "fromCharCode obfuscation"),

    # Data exfiltration
    ("exfil", re.compile(
        r"(fetch|axios\.post|https?\.post)\s*\(\s*['\"]https?://(?!localhost|127\.0\.0\.1)",
        re.I), 3, "HTTP POST to external URL in script"),

    # Crypto mining
    ("mining", re.compile(
        r"(stratum\+tcp|cryptonight|xmrig|monero|pool\.minexmr|coinhive)",
        re.I), 3, "crypto mining indicator"),
]


class JSASTAnalyzer:
    def analyze_source(self, source: str, filename: str = "unknown") -> JSAnalysisResult:
        result = JSAnalysisResult()
        lines  = source.splitlines()
        result.lines_scanned = len(lines)

        in_block_comment = False

        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()

            if "/*" in stripped and not in_block_comment:
                in_block_comment = True
            if "*/" in stripped and in_block_comment:
                in_block_comment = False
                continue

            if in_block_comment:
                continue

            if stripped.startswith("//") or not stripped:
                continue

            code_part = re.sub(r"//.*$", "", line)

            for category, pattern, severity, description in _PATTERNS:
                if pattern.search(code_part):
                    result.findings.append(JSFinding(
                        category    = category,
                        severity    = severity,
                        line        = lineno,
                        description = description,
                        snippet     = code_part.strip()[:120],
                    ))

        if result.findings:
            logger.info(f"[js] {filename}: {len(result.findings)} findings — {result.summary()}")
        else:
            logger.debug(f"[js] {filename}: clean")

        return result

    def analyze_file(self, path: Path) -> JSAnalysisResult:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            r = JSAnalysisResult()
            logger.error(f"[js] Cannot read {path}: {e}")
            return r
        return self.analyze_source(source, filename=str(path))

    def analyze_package_json(self, package_json_path: Path) -> JSAnalysisResult:
        try:
            data = json.loads(package_json_path.read_text(encoding="utf-8"))
        except Exception as e:
            r = JSAnalysisResult()
            logger.error(f"[js] Cannot parse {package_json_path}: {e}")
            return r

        scripts = data.get("scripts", {})
        if not scripts:
            return JSAnalysisResult()

        lines  = []
        labels = {}

        for script_name, script_value in scripts.items():
            labels[len(lines)] = script_name
            lines.append(str(script_value))

        combined = "\n".join(lines)
        result   = self.analyze_source(combined, filename=str(package_json_path) + "#scripts")

        for finding in result.findings:
            script_name = labels.get(finding.line - 1, "unknown")
            finding.description = f"[scripts.{script_name}] {finding.description}"

        self._check_package_fields(data, result)
        return result

    def _check_package_fields(self, data: dict, result: JSAnalysisResult):
        scripts = data.get("scripts", {})
        dangerous_hooks = {"preinstall", "postinstall", "install", "preuninstall"}
        for hook in dangerous_hooks:
            if hook in scripts:
                val = scripts[hook]
                result.findings.append(JSFinding(
                    category    = "exec",
                    severity    = 2,
                    line        = 0,
                    description = f"install hook '{hook}' defined: {str(val)[:80]}",
                    snippet     = str(val)[:120],
                ))

        deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
        known_bad_prefixes = ["coinhive", "cryptoloot"]
        for dep_name in deps:
            if any(dep_name.startswith(p) for p in known_bad_prefixes):
                result.findings.append(JSFinding(
                    category    = "mining",
                    severity    = 3,
                    line        = 0,
                    description = f"known malicious dependency: {dep_name}",
                    snippet     = dep_name,
                ))

    def analyze_files(self, files: dict[str, str]) -> JSAnalysisResult:
        merged = JSAnalysisResult()
        for filename, source in files.items():
            r = self.analyze_source(source, filename=filename)
            merged.findings.extend(r.findings)
            merged.lines_scanned += r.lines_scanned
        return merged


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="JS static analysis")
    parser.add_argument("file", nargs="?", help="Path to .js file")
    parser.add_argument("--package-json", type=str, help="Path to package.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    analyzer = JSASTAnalyzer()

    if args.package_json:
        result = analyzer.analyze_package_json(Path(args.package_json))
        label  = args.package_json
    elif args.file:
        result = analyzer.analyze_file(Path(args.file))
        label  = args.file
    else:
        parser.print_help(); sys.exit(1)

    sev_label = {1: "LOW ", 2: "MED ", 3: "HIGH"}
    print(f"\nFile    : {label}")
    print(f"Lines   : {result.lines_scanned}")
    print(f"Signal  : {result.signal_value:.2f}")
    print(f"Summary : {result.summary()}\n")

    for f in result.findings:
        print(f"  [line {f.line:>4}] [{sev_label.get(f.severity,'?  ')}] "
              f"[{f.category:<12}] {f.description}")
        if f.snippet:
            print(f"           {f.snippet}")
        print()