"""
Dispatches alerts when a HIGH or MEDIUM risk package is detected.

Supported channels:
  - Slack  : POST to an incoming webhook URL
  - Email  : SMTP (Gmail, Outlook, or any SMTP server)
  - Webhook: Generic HTTP POST (JSON payload) — works with PagerDuty, Discord, etc.
  - Console: Always-on fallback (rich formatted output)

Configure via environment variables or .env file:
  SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
  ALERT_EMAIL_TO=security@yourorg.com
  SMTP_HOST=smtp.gmail.com
  SMTP_PORT=587
  SMTP_USER=you@gmail.com
  SMTP_PASS=app-password
  GENERIC_WEBHOOK_URL=https://your-endpoint.com/alert

Run standalone:
  python -m integrations.alert_engine --registry pypi --name reqests
"""

import json
import logging
import os
import smtplib
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from db.models import Package, ScanResult, RiskLevel

logger = logging.getLogger(__name__)


# ── Alert payload ──────────────────────────────────────────────────────────

@dataclass
class AlertPayload:
    """Structured alert object passed to every channel."""
    registry:      str
    package_name:  str
    version:       str | None
    risk_level:    RiskLevel
    score:         float
    verdict:       str
    closest_match: str | None
    detected_at:   datetime

    @property
    def level_str(self) -> str:
        return self.risk_level.value.upper() if hasattr(self.risk_level, 'value') else str(self.risk_level).upper()

    @property
    def title(self) -> str:
        return f"[{self.level_str}] Supply chain risk: {self.registry}/{self.package_name}"

    @property
    def body_text(self) -> str:
        lines = [
            self.title,
            "",
            f"Registry    : {self.registry}",
            f"Package     : {self.package_name}",
            f"Version     : {self.version or 'unknown'}",
            f"Risk level  : {self.level_str}",
            f"Score       : {self.score:.1f} / 100",
            f"Verdict     : {self.verdict}",
        ]
        if self.closest_match:
            lines.append(f"Closest name: {self.closest_match}")
        lines += ["", f"Detected at : {self.detected_at.strftime('%Y-%m-%d %H:%M UTC')}"]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        level_val = self.risk_level.value if hasattr(self.risk_level, 'value') else str(self.risk_level)
        return {
            "registry":     self.registry,
            "package":      self.package_name,
            "version":      self.version,
            "risk_level":   level_val,
            "score":        self.score,
            "verdict":      self.verdict,
            "closest_match":self.closest_match,
            "detected_at":  self.detected_at.isoformat(),
        }


# ── Individual channels ────────────────────────────────────────────────────

class ConsoleChannel:
    """Always-on channel — prints a formatted alert to stdout."""

    COLORS = {
        RiskLevel.HIGH:   "\033[91m",   # bright red
        RiskLevel.MEDIUM: "\033[93m",   # yellow
        RiskLevel.LOW:    "\033[92m",   # green
    }
    RESET = "\033[0m"

    def send(self, alert: AlertPayload) -> bool:
        color = self.COLORS.get(alert.risk_level, "")
        border = "=" * 55
        print(f"\n{color}{border}")
        print(f"  SUPPLY CHAIN ALERT — {alert.level_str}")
        print(border)
        print(alert.body_text)
        print(f"{border}{self.RESET}\n")
        return True


class SlackChannel:
    """
    Sends a formatted message to a Slack incoming webhook.
    """

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")

    def send(self, alert: AlertPayload) -> bool:
        if not self.webhook_url:
            logger.debug("Slack channel skipped — SLACK_WEBHOOK_URL not set")
            return False

        level_val = alert.risk_level.value if hasattr(alert.risk_level, 'value') else str(alert.risk_level)
        emoji = {"high": ":rotating_light:", "medium": ":warning:", "low": ":white_check_mark:"}
        icon  = emoji.get(level_val.lower(), ":question:")

        payload = {
            "text": f"{icon} *{alert.title}*",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"{icon} {alert.title}"}
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Registry*\n{alert.registry}"},
                        {"type": "mrkdwn", "text": f"*Package*\n`{alert.package_name}`"},
                        {"type": "mrkdwn", "text": f"*Version*\n{alert.version or 'unknown'}"},
                        {"type": "mrkdwn", "text": f"*Score*\n{alert.score:.1f} / 100"},
                    ]
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Verdict*\n{alert.verdict}"}
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn",
                         "text": f"Detected at {alert.detected_at.strftime('%Y-%m-%d %H:%M UTC')}"}
                    ]
                }
            ]
        }

        return self._post_json(self.webhook_url, payload)

    def _post_json(self, url: str, payload: dict) -> bool:
        try:
            data = json.dumps(payload).encode("utf-8")
            req  = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                ok = resp.status == 200
                if not ok:
                    logger.warning(f"Slack returned {resp.status}")
                return ok
        except Exception as exc:
            logger.error(f"Slack send failed: {exc}")
            return False


class EmailChannel:
    """
    Sends an alert email via SMTP.
    """

    def __init__(self):
        self.to_addr   = os.getenv("ALERT_EMAIL_TO")
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("SMTP_PASS")

    def send(self, alert: AlertPayload) -> bool:
        if not all([self.to_addr, self.smtp_user, self.smtp_pass]):
            logger.debug("Email channel skipped — SMTP credentials not configured")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = alert.title
        msg["From"]    = self.smtp_user
        msg["To"]      = self.to_addr

        # Plain text part
        msg.attach(MIMEText(alert.body_text, "plain"))

        # Simple HTML part
        html = f"""
        <html><body style="font-family: monospace; padding: 20px;">
        <h2 style="color: {'#c0392b' if alert.risk_level == RiskLevel.HIGH else '#e67e22'}">
            {alert.title}
        </h2>
        <table style="border-collapse: collapse; width: 500px;">
        {"".join(
            f'<tr><td style="padding:6px 12px;border:1px solid #ddd;font-weight:bold">{k}</td>'
            f'<td style="padding:6px 12px;border:1px solid #ddd">{v}</td></tr>'
            for k, v in [
                ("Registry",    alert.registry),
                ("Package",     alert.package_name),
                ("Version",     alert.version or "unknown"),
                ("Risk level",  alert.level_str),
                ("Score",       f"{alert.score:.1f} / 100"),
                ("Verdict",     alert.verdict),
                ("Detected at", alert.detected_at.strftime("%Y-%m-%d %H:%M UTC")),
            ]
        )}
        </table>
        </body></html>
        """
        msg.attach(MIMEText(html, "html"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.sendmail(self.smtp_user, self.to_addr, msg.as_string())
            logger.info(f"Alert email sent to {self.to_addr}")
            return True
        except Exception as exc:
            logger.error(f"Email send failed: {exc}")
            return False


class WebhookChannel:
    """
    Generic HTTP POST webhook.
    """

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or os.getenv("GENERIC_WEBHOOK_URL")

    def send(self, alert: AlertPayload) -> bool:
        if not self.webhook_url:
            logger.debug("Webhook channel skipped — GENERIC_WEBHOOK_URL not set")
            return False

        try:
            data = json.dumps(alert.to_dict()).encode("utf-8")
            req  = urllib.request.Request(
                self.webhook_url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status < 300
        except Exception as exc:
            logger.error(f"Webhook send failed: {exc}")
            return False


# ── Alert engine ───────────────────────────────────────────────────────────

class AlertEngine:
    def __init__(self, alert_on_medium: bool = False):
        self.alert_on_medium = alert_on_medium
        self.channels = [
            ConsoleChannel(),
            SlackChannel(),
            EmailChannel(),
            WebhookChannel(),
        ]

    def dispatch(self, pkg: Package, result: ScanResult) -> bool:
        should_alert = (
            result.risk_level == RiskLevel.HIGH or
            (self.alert_on_medium and result.risk_level == RiskLevel.MEDIUM)
        )

        level_str = result.risk_level.value if hasattr(result.risk_level, 'value') else str(result.risk_level)
        if not should_alert:
            logger.info(f"No alert for {pkg.name} (level={level_str})")
            return False

        reg_str = pkg.registry.value if hasattr(pkg.registry, 'value') else str(pkg.registry)

        alert = AlertPayload(
            registry     = reg_str,
            package_name = pkg.name,
            version      = pkg.version,
            risk_level   = result.risk_level,
            score        = result.risk_score,
            verdict      = result.verdict_reason or "—",
            closest_match= result.closest_match,
            detected_at  = result.scanned_at or datetime.now(timezone.utc),
        )

        sent = False
        for channel in self.channels:
            try:
                ok = channel.send(alert)
                if ok:
                    sent = True
                    logger.info(f"Alert sent via {channel.__class__.__name__}")
            except Exception as exc:
                logger.error(f"Channel {channel.__class__.__name__} raised: {exc}")

        return sent


if __name__ == "__main__":
    import argparse, sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from db.models import Registry, init_db, get_session

    parser = argparse.ArgumentParser(description="Test alert dispatch for a scanned package")
    parser.add_argument("--name",     required=True)
    parser.add_argument("--registry", choices=["npm", "pypi"], default="pypi")
    parser.add_argument("--medium",   action="store_true", help="Also alert on MEDIUM")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    engine = init_db()
    with get_session(engine) as session:
        reg = Registry(args.registry)
        pkg = session.query(Package).filter_by(registry=reg, name=args.name).first()
        if not pkg or not pkg.scan_result:
            print("Package not found or not yet scanned. Run ci_gate first.")
            sys.exit(1)
        result = pkg.scan_result

    alert_engine = AlertEngine(alert_on_medium=args.medium)
    sent = alert_engine.dispatch(pkg, result)
    print(f"\nAlert dispatched: {sent}")