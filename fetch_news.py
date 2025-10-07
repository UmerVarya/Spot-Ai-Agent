import os
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, List

import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
import config
from groq_safe import safe_chat_completion
from news_guardrails import (
    parse_llm_json,
    quantify_event_risk,
    reconcile_with_quant_filters,
)

from log_utils import setup_logger

load_dotenv()
logger = setup_logger(__name__)

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


async def _fetch_rss(session: aiohttp.ClientSession, url: str, impact: str) -> List[Dict[str, str]]:
    events: List[Dict[str, str]] = []
    try:
        async with session.get(url, timeout=10) as resp:
            text = await resp.text()
    except Exception as e:
        logger.warning("Failed to fetch RSS %s: %s", url, e, exc_info=True)
        return events
    try:
        soup = BeautifulSoup(text, features="xml")
        items = soup.find_all("item")[:10]
        for item in items:
            events.append({
                "event": item.title.text,
                "datetime": datetime.utcnow().isoformat() + "Z",
                "impact": impact,
            })
    except Exception as e:
        logger.warning("RSS parse error for %s: %s", url, e, exc_info=True)
    return events


async def fetch_crypto_news(session: aiohttp.ClientSession) -> List[Dict[str, str]]:
    return await _fetch_rss(session, "https://cryptopanic.com/news/rss/", "medium")


async def fetch_macro_news(session: aiohttp.ClientSession) -> List[Dict[str, str]]:
    return await _fetch_rss(session, "https://www.fxstreet.com/rss/news", "high")


async def _run_news_fetcher(path: str = "news_events.json") -> List[Dict[str, str]]:
    async with aiohttp.ClientSession() as session:
        crypto, macro = await asyncio.gather(fetch_crypto_news(session), fetch_macro_news(session))
    events = crypto + macro
    if events:
        save_events(events, path)
    return events


def run_news_fetcher(path: str = "news_events.json") -> List[Dict[str, str]]:
    """Synchronous wrapper for fetching news events."""
    return asyncio.run(_run_news_fetcher(path))


def save_events(events: List[Dict[str, str]], path: str = "news_events.json") -> None:
    with open(path, "w") as f:
        json.dump(events, f, indent=2)
    logger.info("Saved %d events to %s", len(events), path)


def _build_llm_payload(metrics: Dict[str, Any]) -> str:
    return json.dumps(metrics["events_in_window"], indent=2)


def analyze_news_with_llm(events: List[Dict[str, str]]) -> Dict[str, str]:
    if not GROQ_API_KEY:
        return {"safe": True, "sensitivity": 0, "reason": "No API key"}

    metrics = quantify_event_risk(events)
    if metrics["considered_events"] == 0:
        return {
            "safe": True,
            "sensitivity": 0.0,
            "reason": "No impactful events within the monitoring window.",
        }

    prompt = _build_llm_payload(metrics)
    client = Groq(api_key=GROQ_API_KEY)
    try:
        chat_completion = safe_chat_completion(
            client,
            model=config.get_groq_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a crypto macro risk analyst. Respond ONLY with a JSON object "
                        "containing the keys `safe_decision` (\"yes\" or \"no\") and `reason` (string). "
                        "Do not include any additional commentary or keys."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Assess the market impact of the following events occurring within the next "
                        f"{metrics['window_hours']} hours. There are {metrics['considered_events']} "
                        "events under review, including "
                        f"{metrics['high_impact_events']} high-impact entries. Respond with the "
                        "required JSON structure.\n"
                        f"Events:\n{prompt}"
                    ),
                },
            ],
        )
        raw_reply = chat_completion.choices[0].message.content
        safe_decision, reason = parse_llm_json(raw_reply, logger)
        if safe_decision is None:
            return {"safe": True, "sensitivity": 0, "reason": reason or "LLM error"}

        safe, sensitivity, reconciled_reason = reconcile_with_quant_filters(
            safe_decision, reason or "No reason provided.", metrics
        )
        return {"safe": safe, "sensitivity": sensitivity, "reason": reconciled_reason}
    except Exception as e:
        logger.error("Groq LLM analysis failed: %s", e, exc_info=True)
        return {"safe": True, "sensitivity": 0, "reason": "LLM error"}


async def fetch_news(symbol: str) -> List[Dict[str, str]]:
    """Fetch recent news using NewsAPI asynchronously."""
    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY not set; returning empty news list")
        return []
    url = (
        f"https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&language=en&pageSize=5&apiKey={NEWS_API_KEY}"
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as resp:
                data = await resp.json()
                return data.get("articles", [])
    except Exception as e:
        logger.warning("Failed to fetch news for %s: %s", symbol, e, exc_info=True)
        return []
