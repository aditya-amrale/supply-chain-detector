"""
Analyses a package's install/post-install scripts and source files
for suspicious patterns WITHOUT executing any code.
"""

import base64
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Pattern library ────────────────────────────────────────────────────────

PATTERNS = [
    ("outbound_http",
     re.compile(
         r"(https?://(?!localhost|127\.0\.0\.1)[^\s'\"]+|"
         r"fetch\s*\(|axios\.|http\.get\s*\(|urllib\.request|requests\.get|aiohttp)",
         re.IGNORECASE
     ),
     2, "outbound network call detected"),

    ("env_access",
     re.compile(
         r"(process\.env|os\.environ|getenv\s*\(|"
         r"~\/\.ssh|\.aws/credentials|\.npmrc|\.pypirc)",
         re.IGNORECASE
     ),
     2, "environment variable or credential file access"),

    ("shell_exec",
     re.compile(
         r"(subprocess\.|child_process\.|exec\s*\(|eval\s*\(|"
         r"os\.system|os\.popen|spawn\s*\(|execSync|spawnSync)",
         re.IGNORECASE
     ),
     3, "shell command execution detected"),

    ("runtime_b64_decode",
     re.compile(
         r"(base64\.b64decode|atob\s*\(|Buffer\.from\s*\([^)]+,\s*['\"]base64['\"])",
         re.IGNORECASE
     ),
     2, "runtime base64 decoding (possible obfuscation)"),

    ("hex_obfuscation",
     re.compile(r"(\\x[0-9a-f]{2}){4,}", re.IGNORECASE),
     3, "dense hex-encoded string (obfuscation indicator)"),

    ("post_to_external",
     re.compile(
         r"(fetch\s*\(\s*['\"]https?://(?!localhost)|"
         r"requests\.post\s*\(\s*['\"]https?://(?!localhost)|"
         r"axios\.post\s*\(\s*['\"]https?://(?!localhost))",
         re.IGNORECASE
     ),
     3, "HTTP POST to external URL in install script"),

    ("crypto_mining",
     re.compile(
         r"(cryptonight|xmrig|monero|stratum\+tcp|pool\.minexmr)",
         re.IGNORECASE
     ),
     3, "crypto mining pattern detected"),
]


@dataclass
class PatternMatch:
    pattern_name: str
    severity:     int
    description:  str
    line_number:  int
    snippet:      str


@dataclass
class StaticAnalysisResult:
    matches:        list[PatternMatch] = field(default_factory=list)
    files_scanned:  int = 0
    lines_scanned:  int = 0

    @property
    def signal_value(self) -> float:
        if not self.matches:
            return 0.0
        max_sev = max(m.severity for m in self.matches)
        count   = len(self.matches)
        base    = {1: 0.2, 2: 0.55, 3: 0.85}.get(max_sev, 0.0)
        extra   = min((count - 1) * 0.05, 0.15)
        return min(base + extra, 1.0)

    @property
    def has_suspicious_scripts(self) -> bool:
        return any(m.severity >= 2 for m in self.matches)

    def summary(self) -> str:
        if not self.matches:
            return "clean"
        names = list({m.pattern_name for m in self.matches})
        return ", ".join(names)


class StaticAnalysisDetector:
    def scan_text(self, text: str, filename: str = "unknown") -> StaticAnalysisResult:
        result = StaticAnalysisResult(files_scanned=1)
        lines  = text.splitlines()
        result.lines_scanned = len(lines)

        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue

            for name, pattern, severity, description in PATTERNS:
                if pattern.search(line):
                    result.matches.append(PatternMatch(
                        pattern_name = name,
                        severity     = severity,
                        description  = description,
                        line_number  = lineno,
                        snippet      = line.strip()[:120],
                    ))

        if result.matches:
            logger.info(
                f"[static] {filename}: {len(result.matches)} patterns found — "
                f"{result.summary()}"
            )
        else:
            logger.debug(f"[static] {filename}: clean")

        return result

    def scan_files(self, files: dict[str, str]) -> StaticAnalysisResult:
        merged = StaticAnalysisResult()

        for filename, content in files.items():
            r = self.scan_text(content, filename=filename)
            merged.matches.extend(r.matches)
            merged.files_scanned += r.files_scanned
            merged.lines_scanned += r.lines_scanned

        return merged

    def scan_npm_package_json(self, package_json: dict) -> StaticAnalysisResult:
        scripts = package_json.get("scripts", {})
        if not scripts:
            return StaticAnalysisResult(files_scanned=1)

        combined = "\n".join(str(v) for v in scripts.values())
        return self.scan_text(combined, filename="package.json#scripts")


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Scan a file for suspicious patterns")
    parser.add_argument("file", help="Path to file to scan")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        with open(args.file) as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    detector = StaticAnalysisDetector()
    result   = detector.scan_text(content, filename=args.file)

    print(f"\nScanned : {args.file}")
    print(f"Lines   : {result.lines_scanned}")
    print(f"Signal  : {result.signal_value:.2f}")
    print(f"Summary : {result.summary()}\n")

    for m in result.matches:
        sev_label = {1: "LOW", 2: "MED", 3: "HIGH"}.get(m.severity, "?")
        print(f"  [line {m.line_number:>4}] [{sev_label}] {m.pattern_name}")
        print(f"           {m.snippet}")
        print()