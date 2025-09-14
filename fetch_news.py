import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict

import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq
import config

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


def build_news_prompt(events: List[Dict[str, str]]) -> str:
    now = datetime.utcnow()
    filtered = []
    for event in events:
        try:
            event_time = datetime.fromisoformat(event["datetime"].replace("Z", ""))
            hours_until = (event_time - now).total_seconds() / 3600.0
            if hours_until >= -48:
                filtered.append(event)
        except Exception:
            continue
    return json.dumps(filtered, indent=2)


def analyze_news_with_llm(events: List[Dict[str, str]]) -> Dict[str, str]:
    if not GROQ_API_KEY:
        return {"safe": True, "sensitivity": 0, "reason": "No API key"}
    prompt = build_news_prompt(events)
    client = Groq(api_key=GROQ_API_KEY)
    try:
        chat_completion = client.chat.completions.create(
            model=config.get_groq_model(),
            messages=[
                {"role": "system", "content": "You are a crypto macro risk analyst."},
                {"role": "user", "content": prompt},
            ],
        )
        raw_reply = chat_completion.choices[0].message.content
        return json.loads(raw_reply)
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
