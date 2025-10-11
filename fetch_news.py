import os
import json
import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Callable, Coroutine, Tuple

import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import config
from groq_safe import describe_error
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
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

HEADERS = {"Content-Type": "application/json"}
if GROQ_API_KEY:
    HEADERS["Authorization"] = f"Bearer {GROQ_API_KEY}"


def _get_local_llm_adapter() -> Optional[Tuple[Callable[[], bool], Callable[..., str]]]:
    """Return callables for the local Ollama fallback when available."""

    try:
        from local_llm import is_enabled as local_is_enabled, generate as local_generate
    except Exception:
        return None
    return local_is_enabled, local_generate


def _run_coroutine(coro_factory: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    """Execute an async coroutine factory safely from synchronous code."""

    try:
        return asyncio.run(coro_factory())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro_factory())
        finally:
            loop.close()


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


@asynccontextmanager
async def _client_session(session: Optional[aiohttp.ClientSession]):
    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession()
    assert session is not None
    try:
        yield session
    finally:
        if own_session:
            await session.close()


async def run_news_fetcher_async(
    path: str = "news_events.json", *, session: Optional[aiohttp.ClientSession] = None
) -> List[Dict[str, str]]:
    """Asynchronously fetch and cache crypto + macro news events."""

    async with _client_session(session) as client:
        crypto, macro = await asyncio.gather(
            fetch_crypto_news(client), fetch_macro_news(client)
        )
    events = crypto + macro
    if events:
        save_events(events, path)
    return events


def run_news_fetcher(path: str = "news_events.json") -> List[Dict[str, str]]:
    """Synchronous wrapper for fetching news events."""
    return _run_coroutine(lambda: run_news_fetcher_async(path))


def save_events(events: List[Dict[str, str]], path: str = "news_events.json") -> None:
    with open(path, "w") as f:
        json.dump(events, f, indent=2)
    logger.info("Saved %d events to %s", len(events), path)


def _build_llm_payload(metrics: Dict[str, Any]) -> str:
    return json.dumps(metrics["events_in_window"], indent=2)


async def _post_groq_request(
    session: aiohttp.ClientSession,
    payload: Mapping[str, Any],
    *,
    model_used: str,
) -> Dict[str, Any]:
    """Send a POST request to the Groq chat completion endpoint."""

    start = time.perf_counter()
    async with session.post(GROQ_API_URL, headers=HEADERS, json=payload) as resp:
        latency = time.perf_counter() - start
        if resp.status != 200:
            error_payload = await resp.text()
            try:
                parsed = json.loads(error_payload)
            except Exception:
                parsed = error_payload
            logger.error(
                "Groq LLM request failed in %.2fs: %s, %s",
                latency,
                resp.status,
                describe_error(parsed) or error_payload,
            )
            raise RuntimeError("Groq LLM request failed")
        data = await resp.json()
        logger.info("Groq LLM call succeeded in %.2fs (model=%s)", latency, model_used)
        return data


async def analyze_news_with_llm_async(
    events: List[Dict[str, str]],
    *,
    session: Optional[aiohttp.ClientSession] = None,
) -> Dict[str, str]:
    """Asynchronously analyze news events with the Groq LLM."""

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
    system_prompt = (
        "You are a crypto macro risk analyst. Respond ONLY with a JSON object "
        "containing the keys `safe_decision` (\"yes\" or \"no\") and `reason` (string). "
        "Do not include any additional commentary or keys."
    )
    user_prompt = (
        "Assess the market impact of the following events occurring within the next "
        f"{metrics['window_hours']} hours. There are {metrics['considered_events']} "
        "events under review, including "
        f"{metrics['high_impact_events']} high-impact entries. Respond with the "
        "required JSON structure.\n"
        f"Events:\n{prompt}"
    )

    payload_template: Dict[str, Any] = {
        "model": config.get_groq_model(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    async def _run_local_news_fallback() -> Optional[Dict[str, Any]]:
        adapter = _get_local_llm_adapter()
        if adapter is None:
            return None
        is_enabled, generate = adapter
        if not is_enabled():
            return None

        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        loop = asyncio.get_running_loop()
        try:
            raw_reply = await loop.run_in_executor(
                None, lambda: generate(combined_prompt, temperature=0.1)
            )
        except Exception as exc:
            logger.warning("Local news fallback failed: %s", exc, exc_info=True)
            return None

        safe_decision, reason = parse_llm_json(raw_reply, logger)
        if safe_decision is None:
            return {"safe": True, "sensitivity": 0, "reason": reason or "Local LLM error"}

        safe, sensitivity, reconciled_reason = reconcile_with_quant_filters(
            safe_decision, reason or "No reason provided.", metrics
        )
        logger.info("Used local LLM fallback for news analysis")
        return {"safe": safe, "sensitivity": sensitivity, "reason": reconciled_reason}

    try:
        async with _client_session(session) as client:
            response: Optional[Dict[str, Any]] = None
            last_error: Optional[Exception] = None
            models_to_try = [payload_template["model"]]
            if payload_template["model"] != config.DEFAULT_GROQ_MODEL:
                models_to_try.append(config.DEFAULT_GROQ_MODEL)

            for index, model_name in enumerate(models_to_try):
                payload = {**payload_template, "model": model_name}
                try:
                    response = await _post_groq_request(
                        client, payload, model_used=model_name
                    )
                    break
                except Exception as err:
                    last_error = err
                    if (
                        index == 0
                        and len(models_to_try) > 1
                        and model_name != config.DEFAULT_GROQ_MODEL
                    ):
                        logger.warning(
                            "Groq model %s unavailable (%s). Retrying with fallback model %s.",
                            model_name,
                            err,
                            config.DEFAULT_GROQ_MODEL,
                        )

            if response is None:
                fallback_result = await _run_local_news_fallback()
                if fallback_result is not None:
                    return fallback_result
                if last_error is not None:
                    raise last_error
                raise RuntimeError("Groq LLM request failed")

            raw_reply = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            safe_decision, reason = parse_llm_json(raw_reply, logger)
            if safe_decision is None:
                return {"safe": True, "sensitivity": 0, "reason": reason or "LLM error"}

            safe, sensitivity, reconciled_reason = reconcile_with_quant_filters(
                safe_decision, reason or "No reason provided.", metrics
            )
            return {"safe": safe, "sensitivity": sensitivity, "reason": reconciled_reason}
    except Exception as exc:
        logger.error("Groq LLM analysis failed: %s", exc, exc_info=True)
        return {"safe": True, "sensitivity": 0, "reason": "LLM error"}


def analyze_news_with_llm(events: List[Dict[str, str]]) -> Dict[str, str]:
    """Synchronous wrapper around :func:`analyze_news_with_llm_async`."""

    return _run_coroutine(lambda: analyze_news_with_llm_async(events))


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
