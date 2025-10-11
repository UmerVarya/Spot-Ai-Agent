import ast
import json
import smtplib
import os
import csv
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from html import escape
from typing import Any, Callable, Dict, Optional, Tuple

try:  # pragma: no cover - optional dependency in lean test envs
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

from log_utils import setup_logger, ensure_symlink

load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

__all__ = ["send_email", "log_rejection", "send_performance_email"]

logger = setup_logger(__name__)

REJECTED_TRADES_FILE = os.environ.get(
    "REJECTED_TRADES_FILE", "/home/ubuntu/spot_data/trades/rejected_trades.csv"
)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ensure_symlink(REJECTED_TRADES_FILE, os.path.join(_REPO_ROOT, "rejected_trades.csv"))

def _coerce_number(value: Any) -> Optional[float]:
    """Best-effort conversion of a value to ``float``.

    The emails occasionally receive stringified dictionaries or numbers stored as
    strings.  Converting here lets us apply consistent formatting while
    gracefully falling back when the value is not numeric.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None
    return None


def _format_number(value: Any) -> str:
    number = _coerce_number(value)
    if number is None:
        return _format_text(value)
    if abs(number) >= 100:
        formatted = f"{number:,.2f}"
    elif abs(number) >= 1:
        formatted = f"{number:,.3f}"
    else:
        formatted = f"{number:,.5f}"
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def _format_currency(value: Any) -> str:
    number = _coerce_number(value)
    if number is None:
        return _format_text(value)
    return f"${float(number):,.2f}"


def _format_percent(value: Any) -> str:
    number = _coerce_number(value)
    if number is None:
        return _format_text(value)
    return f"{number:.2f}%"


def _format_text(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, dict):
        return f"<pre style='margin:0'>{escape(json.dumps(value, indent=2, default=str))}</pre>"
    if isinstance(value, (list, tuple, set)):
        return escape(", ".join(str(item) for item in value))
    text = str(value).strip()
    if text.lower() in {"none", "n/a", "null", "nan"}:
        return "N/A"
    if not text:
        return "N/A"
    return escape(text).replace("\n", "<br>")


FieldFormatter = Optional[Callable[[Any], str]]
FieldSpec = Tuple[str, Tuple[str, ...], FieldFormatter]

FIELD_SPECS: Tuple[FieldSpec, ...] = (
    ("Symbol", ("symbol", "ticker"), None),
    ("Direction", ("direction", "side"), None),
    ("Score", ("confidence", "score", "confidence_score"), _format_number),
    ("Notional", ("notional", "notional_value", "size"), _format_currency),
    ("Quantity", ("position_size", "quantity", "qty"), _format_number),
    ("Entry", ("entry", "entry_price"), _format_number),
    ("Average Entry", ("avg_entry", "average_entry"), _format_number),
    ("Stop Loss", ("sl", "stop", "stop_loss", "stop_price"), _format_number),
    ("TP1", ("tp1", "target", "target_1"), _format_number),
    ("TP2", ("tp2", "target_2"), _format_number),
    ("TP3", ("tp3", "target_3"), _format_number),
    ("Exit", ("exit", "exit_price", "close_price"), _format_number),
    ("PnL", ("pnl", "pnl_usd", "profit"), _format_currency),
    ("PnL %", ("pnl_pct", "return_pct", "roi"), _format_percent),
    ("Time", ("entry_time", "timestamp", "created_at", "time"), _format_text),
)


def _choose_heading(subject: Optional[str]) -> str:
    if not subject:
        return "📈 Trade Notification"
    lowered = subject.lower()
    if "exit" in lowered:
        return "🚪 Trade Exit Alert"
    if "update" in lowered:
        return "🛠️ Trade Update"
    if "reject" in lowered:
        return "⚠️ Trade Rejected"
    return "📈 Trade Notification"


def _extract_details_from_message(message: str) -> Dict[str, Any]:
    """Attempt to recover structured information from a free-form message."""

    if not message:
        return {}
    text = message.strip()
    if not text:
        return {}
    candidate = text
    narrative = None
    if "Narrative:" in text:
        candidate, narrative = text.split("Narrative:", 1)
    candidate = candidate.strip()
    parsed: Dict[str, Any] = {}
    if candidate.startswith("{") and candidate.endswith("}"):
        try:
            parsed_obj = ast.literal_eval(candidate)
            if isinstance(parsed_obj, dict):
                parsed = parsed_obj
        except Exception:
            try:
                parsed_json = json.loads(candidate)
                if isinstance(parsed_json, dict):
                    parsed = parsed_json
            except Exception:
                parsed = {}
    if narrative:
        parsed.setdefault("narrative", narrative.strip())
    return parsed


def _prepare_details(trade_details: Any) -> Tuple[Dict[str, Any], Optional[str], Optional[str], Optional[str]]:
    """Normalise incoming trade payload into structured sections."""

    reasoning = None
    narrative = None
    notes = None

    if not trade_details:
        return {}, reasoning, narrative, notes

    if isinstance(trade_details, str):
        notes = trade_details.strip()
        inferred = _extract_details_from_message(notes)
        details: Dict[str, Any] = inferred or {}
        if inferred and notes:
            remaining = notes.split("Narrative:", 1)
            if len(remaining) == 2:
                notes = remaining[1].strip() or None
        if not details:
            return {}, None, None, notes
    else:
        details = {str(k): v for k, v in dict(trade_details).items()}

    reasoning = details.pop("reasoning", None) or details.pop("llm_reasoning", None)
    narrative = details.pop("narrative", None)
    notes = notes or details.pop("message", None)

    if isinstance(reasoning, dict):
        reasoning = json.dumps(reasoning, indent=2, default=str)
    if isinstance(narrative, dict):
        narrative = json.dumps(narrative, indent=2, default=str)
    if isinstance(notes, dict):
        notes = json.dumps(notes, indent=2, default=str)

    return details, reasoning, narrative, notes


def _render_rows(details: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    used_keys = set()
    rows = []
    for label, keys, formatter in FIELD_SPECS:
        value = None
        selected_key = None
        for key in keys:
            if key in details and details[key] not in (None, ""):
                value = details[key]
                selected_key = key
                break
        if selected_key is None:
            continue
        used_keys.add(selected_key)
        formatter_fn = formatter or _format_text
        rows.append(
            f"<tr><td style='padding:6px 12px;font-weight:600;background:#f8fafc;border:1px solid #e2e8f0;width:35%'>{escape(label)}</td>"
            f"<td style='padding:6px 12px;border:1px solid #e2e8f0;background:#ffffff'>{formatter_fn(value)}</td></tr>"
        )

    remaining = {k: v for k, v in details.items() if k not in used_keys}
    return "".join(rows), remaining


def _render_additional(remaining: Dict[str, Any]) -> str:
    if not remaining:
        return ""
    items = []
    for key in sorted(remaining.keys()):
        if remaining[key] in (None, ""):
            continue
        items.append(
            f"<tr><td style='padding:6px 12px;font-weight:600;background:#f8fafc;border:1px solid #e2e8f0;width:35%'>{escape(key.replace('_', ' ').title())}</td>"
            f"<td style='padding:6px 12px;border:1px solid #e2e8f0;background:#ffffff'>{_format_text(remaining[key])}</td></tr>"
        )
    if not items:
        return ""
    return (
        "<h3 style='margin:24px 0 12px;color:#1f2937;font-size:16px;'>Additional Details</h3>"
        "<table style='width:100%;border-collapse:collapse;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;'>"
        + "".join(items)
        + "</table>"
    )


def _render_section(title: str, content: Optional[str]) -> str:
    if not content:
        return ""
    return (
        f"<h3 style='margin:24px 0 12px;color:#1f2937;font-size:16px;'>{escape(title)}</h3>"
        f"<div style='padding:12px 16px;border:1px solid #e2e8f0;background:#f9fafb;border-radius:8px;white-space:pre-wrap;color:#111827;font-size:14px;line-height:1.5;'>"
        f"{_format_text(content)}</div>"
    )


def send_email(subject, trade_details):
    try:
        details, reasoning, narrative, notes = _prepare_details(trade_details)
        if not details and not any([reasoning, narrative, notes]):
            return

        summary_rows, remaining = _render_rows(details)
        additional = _render_additional(remaining)

        body = f"""
        <html>
          <body style="font-family: 'Segoe UI', Arial, sans-serif; background:#f3f4f6; padding:24px; color:#111827;">
            <div style="max-width:640px;margin:0 auto;background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;box-shadow:0 10px 30px rgba(15,23,42,0.08);">
              <div style="background:#0f172a;color:#f8fafc;padding:20px 24px;">
                <div style="font-size:22px;font-weight:600;">{_choose_heading(subject)}</div>
                <div style="margin-top:4px;font-size:14px;opacity:0.8;">{escape(subject or 'Trade Notification')}</div>
              </div>
              <div style="padding:24px;">
                {f"<table style='width:100%;border-collapse:collapse;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;'>{summary_rows}</table>" if summary_rows else ""}
                {additional}
                {_render_section('Narrative', narrative)}
                {_render_section('LLM Reasoning', reasoning)}
                {_render_section('Notes', notes)}
              </div>
            </div>
          </body>
        </html>
        """

        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject

        msg.attach(MIMEText(body, "html"))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        logger.info("LLM Explanation Email Sent!")

    except Exception as e:
        logger.error("Email sending failed: %s", e, exc_info=True)


def send_performance_email(subject: str, message: str) -> None:
    """Send a simple email containing a performance summary.

    Parameters
    ----------
    subject : str
        Email subject line.
    message : str
        HTML body to include in the email.
    """
    try:
        if not message:
            return
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "html"))
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        logger.info("Performance email sent")
    except Exception as exc:
        logger.error("Performance email failed: %s", exc, exc_info=True)


def log_rejection(symbol: str, reason: str) -> None:
    """Append a rejected trade and reason to ``rejected_trades.csv``.

    Parameters
    ----------
    symbol : str
        The trading symbol that was rejected.
    reason : str
        Explanation for why the trade was rejected.
    """

    log_file = REJECTED_TRADES_FILE
    headers = ["timestamp", "symbol", "reason"]
    row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "reason": reason,
    }

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
