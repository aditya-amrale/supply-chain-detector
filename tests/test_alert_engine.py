"""
Tests for the alert engine and its channels.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from db.models import Package, ScanResult, Registry, RiskLevel
from integrations.alert_engine import (
    AlertEngine, AlertPayload, ConsoleChannel,
    SlackChannel, WebhookChannel
)


def make_alert(risk_level=RiskLevel.HIGH, score=85.0):
    return AlertPayload(
        registry="pypi",
        package_name="reqests",
        version="1.0.0",
        risk_level=risk_level,
        score=score,
        verdict="name similar to 'requests' (dist=1)",
        closest_match="requests",
        detected_at=datetime.now(timezone.utc),
    )


def make_pkg_and_result(risk_level=RiskLevel.HIGH, score=85.0):
    pkg = Package(
        id=1, registry=Registry.PYPI, name="reqests", version="1.0.0"
    )
    result = ScanResult(
        package_id=1,
        risk_level=risk_level,
        risk_score=score,
        verdict_reason="name similar to 'requests'",
        closest_match="requests",
        scanned_at=datetime.now(timezone.utc),
    )
    pkg.scan_result = result
    return pkg, result


class TestAlertPayload:

    def test_title_includes_level(self):
        alert = make_alert(RiskLevel.HIGH)
        assert "HIGH" in alert.title
        assert "reqests" in alert.title

    def test_body_text_has_all_fields(self):
        alert = make_alert()
        assert "pypi" in alert.body_text
        assert "85.0" in alert.body_text
        assert "requests" in alert.body_text

    def test_to_dict_serialisable(self):
        import json
        alert = make_alert()
        d = alert.to_dict()
        json.dumps(d)    # should not raise
        assert d["risk_level"] in ("high", "RiskLevel.HIGH")
        assert d["score"] == 85.0


class TestConsoleChannel:

    def test_console_always_returns_true(self, capsys):
        ch    = ConsoleChannel()
        alert = make_alert()
        ok    = ch.send(alert)
        assert ok is True
        captured = capsys.readouterr()
        assert "HIGH" in captured.out


class TestSlackChannel:

    def test_skipped_when_no_url(self):
        ch    = SlackChannel(webhook_url=None)
        alert = make_alert()
        ok = ch.send(alert)
        assert ok is False

    def test_sends_when_url_configured(self):
        ch = SlackChannel(webhook_url="https://hooks.slack.com/fake")
        alert = make_alert()
        with patch.object(ch, "_post_json", return_value=True) as mock_post:
            ok = ch.send(alert)
        assert ok is True
        assert mock_post.called
        payload = mock_post.call_args[0][1]
        assert "blocks" in payload


class TestWebhookChannel:

    def test_skipped_when_no_url(self):
        ch = WebhookChannel(webhook_url=None)
        ok = ch.send(make_alert())
        assert ok is False


class TestAlertEngine:

    def test_high_risk_triggers_alert(self):
        engine = AlertEngine()
        pkg, result = make_pkg_and_result(RiskLevel.HIGH, 85.0)
        with patch.object(ConsoleChannel, "send", return_value=True) as mock_send:
            sent = engine.dispatch(pkg, result)
        assert sent is True
        assert mock_send.called

    def test_low_risk_no_alert(self):
        engine = AlertEngine(alert_on_medium=False)
        pkg, result = make_pkg_and_result(RiskLevel.LOW, 10.0)
        with patch.object(ConsoleChannel, "send", return_value=True) as mock_send:
            sent = engine.dispatch(pkg, result)
        assert sent is False
        assert not mock_send.called

    def test_medium_alert_when_enabled(self):
        engine = AlertEngine(alert_on_medium=True)
        pkg, result = make_pkg_and_result(RiskLevel.MEDIUM, 45.0)
        with patch.object(ConsoleChannel, "send", return_value=True) as mock_send:
            sent = engine.dispatch(pkg, result)
        assert sent is True

    def test_medium_no_alert_when_disabled(self):
        engine = AlertEngine(alert_on_medium=False)
        pkg, result = make_pkg_and_result(RiskLevel.MEDIUM, 45.0)
        with patch.object(ConsoleChannel, "send", return_value=True) as mock_send:
            sent = engine.dispatch(pkg, result)
        assert sent is False