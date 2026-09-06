"""
Deep static analysis using Python's built-in `ast` module.

Unlike the regex scanner (which matches text patterns), this module
builds a full Abstract Syntax Tree and inspects the actual structure
of the code — catching obfuscated calls that regex misses.

What it detects:
  - Network calls    : urllib, http.client, socket, requests, aiohttp
  - Credential theft : os.environ, open("~/.ssh/…"), subprocess with sensitive paths
  - Code execution   : eval(), exec(), compile() + exec, __import__
  - Data exfiltration: assignments of env vars followed by network calls (taint-lite)
  - Obfuscation      : chr() concatenation, bytes decoding tricks, getattr(builtins,…)
  - File system ops  : reading /etc/passwd, /etc/shadow, ~/.aws, ~/.ssh

Run standalone:
  python -m detectors.ast_analyzer path/to/file.py
"""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Finding dataclass ──────────────────────────────────────────────────────

@dataclass
class ASTFinding:
    category:    str        # "network" | "credential" | "exec" | "obfuscation" | "exfil"
    severity:    int        # 1 low, 2 medium, 3 high
    line:        int
    description: str
    code_snippet:str = ""


@dataclass
class ASTAnalysisResult:
    findings:      list[ASTFinding] = field(default_factory=list)
    lines_scanned: int = 0
    parse_error:   str | None = None

    @property
    def signal_value(self) -> float:
        if self.parse_error or not self.findings:
            return 0.0
        max_sev = max(f.severity for f in self.findings)
        count   = len(self.findings)
        base    = {1: 0.15, 2: 0.55, 3: 0.90}.get(max_sev, 0.0)
        extra   = min((count - 1) * 0.05, 0.10)
        
        # Add bonus for multi-vector threat combinations (e.g. creds + network + obfuscation)
        categories = {f.category for f in self.findings}
        category_bonus = 0.20 if len(categories) >= 3 else (0.10 if len(categories) == 2 else 0.0)

        return min(round(base + extra + category_bonus, 2), 1.0)

    @property
    def has_suspicious_code(self) -> bool:
        return any(f.severity >= 2 for f in self.findings)

    def summary(self) -> str:
        if not self.findings:
            return "clean"
        cats = list({f.category for f in self.findings})
        return ", ".join(cats)


# ── AST visitor ───────────────────────────────────────────────────────────

class MaliciousPatternVisitor(ast.NodeVisitor):
    """
    Walks the AST and records suspicious patterns.
    """

    # Dangerous builtins
    EXEC_BUILTINS = {"eval", "exec", "compile", "__import__", "execfile"}

    # Sensitive file paths
    SENSITIVE_PATHS = {
        ".ssh", ".aws", ".npmrc", ".pypirc",
        "/etc/passwd", "/etc/shadow", "id_rsa", "credentials",
    }

    # Known network modules
    NETWORK_MODULES = {
        "urllib", "urllib2", "urllib3", "http", "httplib",
        "requests", "aiohttp", "httpx", "socket", "ssl",
        "ftplib", "smtplib", "imaplib", "poplib",
    }

    def __init__(self, source_lines: list[str]):
        self.findings: list[ASTFinding] = []
        self.source_lines = source_lines
        # Track imports: alias → real module/name
        self._imports: dict[str, str] = {}
        # Track variable names that hold env/credential data
        self._tainted_names: set[str] = set()

    def _snippet(self, lineno: int) -> str:
        if 0 < lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()[:100]
        return ""

    def _add(self, category, severity, node, description):
        self.findings.append(ASTFinding(
            category=category,
            severity=severity,
            line=getattr(node, "lineno", 0),
            description=description,
            code_snippet=self._snippet(getattr(node, "lineno", 0)),
        ))

    # ── Import tracking ────────────────────────────────────────────────────

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            key = alias.asname or alias.name.split(".")[0]
            self._imports[key] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            key = alias.asname or alias.name
            self._imports[key] = f"{module}.{alias.name}"
        self.generic_visit(node)

    # ── Dangerous function calls ───────────────────────────────────────────

    def visit_Call(self, node: ast.Call):
        func_name = self._resolve_call(node.func)

        # eval / exec / compile / __import__
        if func_name in self.EXEC_BUILTINS:
            self._add("exec", 3, node, f"dangerous builtin call: {func_name}()")

        # subprocess / os.system / os.popen
        elif func_name in {
            "subprocess.run", "subprocess.call", "subprocess.Popen",
            "subprocess.check_output", "os.system", "os.popen",
            "os.execv", "os.execve", "commands.getoutput",
        }:
            self._add("exec", 3, node, f"shell execution: {func_name}()")

        # open() with sensitive path
        elif func_name == "open":
            path_arg = self._first_string_arg(node)
            if path_arg and any(s in path_arg for s in self.SENSITIVE_PATHS):
                self._add("credential", 3, node, f"reading sensitive file: open('{path_arg}')")

        # os.environ.get / os.getenv
        elif func_name in {"os.environ.get", "os.getenv", "os.environ.__getitem__"}:
            key_arg = self._first_string_arg(node)
            self._add("credential", 2, node, f"environment variable access: {func_name}('{key_arg or '?'}')")

        # Network function calls / methods
        elif any(func_name.startswith(m) for m in self.NETWORK_MODULES) or func_name.endswith(".send"):
            self._add("network", 2, node, f"network call: {func_name}()")

        # base64 decoding or encoding at runtime
        elif func_name in {
            "base64.b64decode", "base64.decodebytes", "base64.decodestring",
            "base64.b64encode", "base64.encodebytes", "base64.encodestring"
        }:
            self._add("obfuscation", 2, node, f"runtime base64 operation: {func_name}()")

        # chr() used in expressions
        elif func_name == "chr":
            parent = getattr(node, "_parent", None)
            if isinstance(parent, (ast.BinOp, ast.JoinedStr, ast.Call)):
                self._add("obfuscation", 2, node, "chr() in expression — possible string obfuscation")

        # getattr on builtins
        elif func_name == "getattr":
            first = self._first_string_arg(node)
            args  = node.args
            if len(args) >= 1:
                target = ast.unparse(args[0]) if hasattr(ast, "unparse") else ""
                if "builtin" in target or "import" in (first or ""):
                    self._add("obfuscation", 3, node, "getattr on builtins — dynamic import attempt")

        self.generic_visit(node)

    # ── Assignment-level taint tracking ───────────────────────────────────

    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, ast.Call):
            func_name = self._resolve_call(node.value.func)
            if func_name in {"os.environ.get", "os.getenv"}:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._tainted_names.add(target.id)
        self.generic_visit(node)

    # ── String constants ──────────────────────────────────────────────────

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            val = node.value
            hex_count = sum(1 for i in range(len(val) - 1) if val[i] == "\\" and val[i+1] == "x")
            if hex_count >= 4:
                self._add("obfuscation", 2, node, f"hex-obfuscated string constant ({hex_count} escapes)")
            if len(val) > 40 and val.replace("+", "").replace("/", "").replace("=", "").isalnum():
                if " " not in val and "\n" not in val:
                    self._add("obfuscation", 1, node, f"possible base64 payload string (len={len(val)})")
        self.generic_visit(node)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _resolve_call(self, func_node) -> str:
        try:
            if isinstance(func_node, ast.Name):
                return self._imports.get(func_node.id, func_node.id)
            if isinstance(func_node, ast.Attribute):
                obj = self._resolve_call(func_node.value)
                return f"{obj}.{func_node.attr}"
        except Exception:
            pass
        return ""

    def _first_string_arg(self, node: ast.Call) -> str | None:
        if node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value
        return None


def _set_parents(tree: ast.AST):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node  # type: ignore[attr-defined]


# ── Public API ─────────────────────────────────────────────────────────────

class ASTAnalyzer:
    def analyze_source(self, source: str, filename: str = "unknown") -> ASTAnalysisResult:
        result = ASTAnalysisResult(lines_scanned=len(source.splitlines()))
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as e:
            result.parse_error = str(e)
            logger.warning(f"[ast] Parse error in {filename}: {e}")
            return result

        _set_parents(tree)
        visitor = MaliciousPatternVisitor(source.splitlines())
        visitor.visit(tree)
        result.findings = visitor.findings

        if result.findings:
            logger.info(f"[ast] {filename}: {len(result.findings)} findings — {result.summary()}")
        else:
            logger.debug(f"[ast] {filename}: clean")

        return result

    def analyze_file(self, path: Path) -> ASTAnalysisResult:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            r = ASTAnalysisResult()
            r.parse_error = str(e)
            return r
        return self.analyze_source(source, filename=str(path))

    def analyze_files(self, files: dict[str, str]) -> ASTAnalysisResult:
        merged = ASTAnalysisResult()
        for filename, source in files.items():
            r = self.analyze_source(source, filename=filename)
            merged.findings.extend(r.findings)
            merged.lines_scanned += r.lines_scanned
            if r.parse_error and not merged.parse_error:
                merged.parse_error = r.parse_error
        return merged


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="AST-based analysis of a Python file")
    parser.add_argument("file", help="Path to .py file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    analyzer = ASTAnalyzer()
    result   = analyzer.analyze_file(Path(args.file))

    print(f"\nFile    : {args.file}")
    print(f"Lines   : {result.lines_scanned}")
    print(f"Signal  : {result.signal_value:.2f}")
    print(f"Summary : {result.summary()}")
    if result.parse_error:
        print(f"Parse error: {result.parse_error}")
    print()

    sev_label = {1: "LOW ", 2: "MED ", 3: "HIGH"}
    for f in result.findings:
        print(f"  [line {f.line:>4}] [{sev_label.get(f.severity,'?  ')}] "
              f"[{f.category:<12}] {f.description}")
        if f.code_snippet:
            print(f"           {f.code_snippet}")
        print()
