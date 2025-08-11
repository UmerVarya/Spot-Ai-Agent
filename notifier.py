import smtplib
import os
import csv
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from log_utils import setup_logger

load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

__all__ = ["send_email", "log_rejection"]

logger = setup_logger(__name__)

def send_email(subject, trade_details):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject

        # ✅ Get reasoning explanation from trade details (provided by LLM earlier)
        reasoning = trade_details.get("reasoning", "No reasoning provided.")

        body = f"""
        <h2>🧠 New Trade Triggered</h2>
        <p><strong>Symbol:</strong> {trade_details.get('symbol')}</p>
        <p><strong>Direction:</strong> {trade_details.get('direction')}</p>
        <p><strong>Score:</strong> {trade_details.get('confidence')}</p>
        <p><strong>Position Size:</strong> ${trade_details.get('position_size')}</p>
        <p><strong>Entry:</strong> {trade_details.get('entry')}</p>
        <p><strong>SL:</strong> {trade_details.get('sl')}</p>
        <p><strong>TP1:</strong> {trade_details.get('tp1')}</p>
        <p><strong>TP2:</strong> {trade_details.get('tp2')}</p>
        <p><strong>TP3:</strong> {trade_details.get('tp3')}</p>
        <hr>
        <h3>🤖 LLM Reasoning:</h3>
        <p>{reasoning}</p>
        """

        msg.attach(MIMEText(body, "html"))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        logger.info("LLM Explanation Email Sent!")

    except Exception as e:
        logger.error("Email sending failed: %s", e, exc_info=True)


def log_rejection(symbol: str, reason: str) -> None:
    """Append a rejected trade and reason to ``rejected_trades.csv``.

    Parameters
    ----------
    symbol : str
        The trading symbol that was rejected.
    reason : str
        Explanation for why the trade was rejected.
    """

    log_file = "rejected_trades.csv"
    headers = ["timestamp", "symbol", "reason"]
    row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "reason": reason,
    }

    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
