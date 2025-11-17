from __future__ import annotations

import os
import json
import asyncio
import random
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, Iterable, List, Mapping, Optional, Tuple

import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import config
from groq_client import get_groq_client
from groq_safe import GroqAuthError, safe_chat_completion
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

_NEWS_CACHE_PATH_DEFAULT = "news_events.json"
_WARN_INTERVAL_SECONDS = 120.0
_NEWS_CACHE: Dict[str, Any] = {
    "ok": False,
    "items": [],
    "source": "neutral",
    "error": "not_loaded",
    "timestamp": 0.0,
}
_NEWS_CACHE_LOCK = threading.Lock()
_LAST_WARN_TS = 0.0
_BACKGROUND_LOOP: Optional[asyncio.AbstractEventLoop] = None
_BACKGROUND_THREAD: Optional[threading.Thread] = None
_BACKGROUND_LOCK = threading.Lock()
_INFLIGHT_REFRESH: Optional[asyncio.Future] = None
_NEWS_LLM_AUTH_DISABLED = False
_NEWS_LLM_AUTH_WARNED = False
_NEWS_LLM_AUTH_REASON = "Groq news analysis unavailable"


def _disable_news_llm(reason: str | None) -> None:
    """Record that Groq news analysis is disabled due to authentication."""

    global _NEWS_LLM_AUTH_DISABLED, _NEWS_LLM_AUTH_WARNED, _NEWS_LLM_AUTH_REASON
    _NEWS_LLM_AUTH_DISABLED = True
    text = (reason or "Groq authentication failed").strip()
    if not text:
        text = "Groq authentication failed"
    _NEWS_LLM_AUTH_REASON = text
    if not _NEWS_LLM_AUTH_WARNED:
        logger.warning(
            "Groq news analysis disabled due to authentication failure: %s",
            text,
        )
        _NEWS_LLM_AUTH_WARNED = True


def _update_cache(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge *payload* into the shared news cache."""

    with _NEWS_CACHE_LOCK:
        _NEWS_CACHE.update(payload)
        _NEWS_CACHE["timestamp"] = time.time()
        snapshot = {
            "ok": bool(_NEWS_CACHE.get("ok")),
            "items": list(_NEWS_CACHE.get("items", [])),
            "source": str(_NEWS_CACHE.get("source", "neutral")),
            "error": _NEWS_CACHE.get("error"),
        }
    return snapshot


def get_news_cache() -> Dict[str, Any]:
    """Return a snapshot of the last known news payload."""

    with _NEWS_CACHE_LOCK:
        return {
            "ok": bool(_NEWS_CACHE.get("ok")),
            "items": list(_NEWS_CACHE.get("items", [])),
            "source": str(_NEWS_CACHE.get("source", "neutral")),
            "error": _NEWS_CACHE.get("error"),
        }


def _throttled_warning(message: str, *args: Any, **kwargs: Any) -> None:
    """Emit a warning log no more than once every ``_WARN_INTERVAL_SECONDS``."""

    global _LAST_WARN_TS

    now = time.time()
    if now - _LAST_WARN_TS >= _WARN_INTERVAL_SECONDS:
        _LAST_WARN_TS = now
        logger.warning(message, *args, **kwargs)
    else:
        logger.debug(message, *args, **kwargs)


def _prime_cache_from_disk(path: str = _NEWS_CACHE_PATH_DEFAULT) -> None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            items = json.load(handle)
    except FileNotFoundError:
        return
    except Exception:
        logger.debug("Unable to prime news cache from %s", path, exc_info=True)
        return

    if isinstance(items, list) and items:
        _update_cache({
            "ok": True,
            "items": items,
            "source": "cache",
            "error": None,
        })


_prime_cache_from_disk()
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


@asynccontextmanager
async def _client_session(session: Optional[aiohttp.ClientSession]):
    own_session = session is None
    if own_session:
        timeout = aiohttp.ClientTimeout(total=6, connect=3, sock_read=3)
        session = aiohttp.ClientSession(timeout=timeout)
    assert session is not None
    try:
        yield session
    finally:
        if own_session:
            await session.close()


async def fetch_source(session: aiohttp.ClientSession, url: str, attempts: int = 2) -> Tuple[bool, str, Optional[str]]:
    """Fetch a URL with retries, returning ``(ok, text, error)``."""

    last_error: Optional[str] = None
    for attempt in range(max(1, int(attempts))):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return True, await response.text(), None
                last_error = f"HTTP {response.status}"
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < attempts - 1:
            await asyncio.sleep(0.4 * (2**attempt) + random.random() * 0.2)
    return False, "", last_error


def _parse_rss(text: str, impact: str) -> List[Dict[str, str]]:
    events: List[Dict[str, str]] = []
    if not text.strip():
        return events
    soup = BeautifulSoup(text, features="xml")
    items = soup.find_all("item")[:10]
    now = datetime.utcnow().isoformat() + "Z"
    for item in items:
        title = getattr(item, "title", None)
        if not title or not getattr(title, "text", "").strip():
            continue
        events.append({
            "event": title.text.strip(),
            "datetime": now,
            "impact": impact,
        })
    return events


async def run_news_fetcher_async(
    path: str = _NEWS_CACHE_PATH_DEFAULT, *, session: Optional[aiohttp.ClientSession] = None
) -> Dict[str, Any]:
    """Fetch news items with bounded retries, updating the shared cache."""

    if os.getenv("NEWS_DISABLED", "false").lower() in {"1", "true", "yes"}:
        cached = get_news_cache()
        result = {
            "ok": bool(cached["items"]),
            "items": cached["items"],
            "source": "disabled",
            "error": "disabled",
        }
        _update_cache(result)
        return result

    semaphore = asyncio.Semaphore(3)
    start_time = time.perf_counter()
    sources: Iterable[Tuple[str, str]] = (
        ("https://cryptopanic.com/news/rss/", "medium"),
        ("https://www.fxstreet.com/rss/news", "high"),
    )
    source_names = ["cryptopanic", "fxstreet"]

    async def _run_source(url: str, impact: str) -> Tuple[bool, List[Dict[str, str]]]:
        async with semaphore:
            ok, raw, err = await fetch_source(session_obj, url)
            if not ok:
                if err:
                    _throttled_warning("News source failed: %s (%s)", url, err)
                return False, []
            try:
                return True, _parse_rss(raw, impact)
            except Exception as parse_exc:  # pragma: no cover - defensive logging
                _throttled_warning("News parse failed for %s: %s", url, parse_exc)
                return False, []

    try:
        async with _client_session(session) as session_obj:
            tasks = [asyncio.create_task(_run_source(url, impact)) for url, impact in sources]
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
    except asyncio.CancelledError:
        cached = get_news_cache()
        result = {
            "ok": bool(cached["items"]),
            "items": cached["items"],
            "source": "cache" if cached["items"] else "neutral",
            "error": "cancelled",
        }
        _update_cache(result)
        return result
    except Exception as exc:
        _throttled_warning("News fetcher failed: %s", exc)
        cached = get_news_cache()
        result = {
            "ok": bool(cached["items"]),
            "items": cached["items"],
            "source": "cache" if cached["items"] else "neutral",
            "error": "fetch_failed",
        }
        _update_cache(result)
        return result

    events: List[Dict[str, str]] = []
    failures = 0
    successes = 0
    for outcome in results:
        if isinstance(outcome, Exception):
            failures += 1
            _throttled_warning("News task failed: %r", outcome)
            continue
        success, payload = outcome
        if success:
            successes += 1
            if payload:
                events.extend(payload)
        else:
            failures += 1

    duration = time.perf_counter() - start_time

    if events:
        save_events(events, path)
        result = {
            "ok": True,
            "items": events,
            "source": "fxstreet",
            "error": None,
        }
        _update_cache(result)
        logger.info(
            "news_refresh ok=1 fail=%s from=%s duration=%.2fs cached=%d",
            failures,
            ",".join(source_names),
            duration,
            len(events),
        )
        return result

    if successes and failures == 0:
        result = {
            "ok": True,
            "items": [],
            "source": ",".join(source_names),
            "error": None,
        }
        _update_cache(result)
        logger.info(
            "news_refresh ok=1 fail=%s from=%s duration=%.2fs cached=%d",
            failures,
            ",".join(source_names),
            duration,
            len(result["items"]),
        )
        return result

    cached = get_news_cache()
    source = "cache" if cached["items"] else "neutral"
    result = {
        "ok": bool(cached["items"]),
        "items": cached["items"],
        "source": source,
        "error": "fetch_failed",
    }
    _update_cache(result)
    logger.info(
        "news_refresh ok=0 fail=%s from=%s duration=%.2fs cached=%d",
        failures,
        ",".join(source_names),
        duration,
        len(result["items"]),
    )
    return result


def run_news_fetcher(path: str = "news_events.json") -> Dict[str, Any]:
    """Synchronous wrapper for fetching news events."""
    try:
        return _run_coroutine(lambda: run_news_fetcher_async(path))
    except Exception as exc:
        _throttled_warning("News fetcher execution failed: %s", exc)
        cached = get_news_cache()
        return {
            "ok": bool(cached["items"]),
            "items": cached["items"],
            "source": "cache" if cached["items"] else "neutral",
            "error": "fetch_failed",
        }


def save_events(events: List[Dict[str, str]], path: str = "news_events.json") -> None:
    with open(path, "w") as f:
        json.dump(events, f, indent=2)
    logger.info("Saved %d events to %s", len(events), path)


def _loop_runner(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    global _BACKGROUND_LOOP, _BACKGROUND_THREAD
    with _BACKGROUND_LOCK:
        loop = _BACKGROUND_LOOP
        if loop and loop.is_running():
            return loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=_loop_runner, args=(loop,), daemon=True)
        thread.start()
        _BACKGROUND_LOOP = loop
        _BACKGROUND_THREAD = thread
        return loop


def trigger_news_refresh(path: str = _NEWS_CACHE_PATH_DEFAULT) -> None:
    """Schedule a background refresh without blocking the caller."""

    loop = _ensure_background_loop()

    def _schedule() -> None:
        global _INFLIGHT_REFRESH
        if _INFLIGHT_REFRESH and not _INFLIGHT_REFRESH.done():
            return

        task = asyncio.create_task(run_news_fetcher_async(path))

        def _clear(_: asyncio.Future) -> None:
            global _INFLIGHT_REFRESH
            _INFLIGHT_REFRESH = None

        task.add_done_callback(_clear)
        _INFLIGHT_REFRESH = task

    loop.call_soon_threadsafe(_schedule)


def _build_llm_payload(metrics: Dict[str, Any]) -> str:
    return json.dumps(metrics["events_in_window"], indent=2)


def _extract_message_content(response: Any) -> str:
    """Return the first message content from a Groq chat completion."""

    try:
        choice = response.choices[0]
        return str(getattr(choice.message, "content", "") or "").strip()
    except Exception:
        return ""


async def _chat_completion_async(
    messages: List[Mapping[str, str]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
):
    """Execute a Groq chat completion using the shared SDK client."""

    client = get_groq_client()
    if client is None:
        raise RuntimeError("Groq client unavailable")

    return await asyncio.to_thread(
        safe_chat_completion,
        client,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def analyze_news_with_llm_async(
    events: List[Dict[str, str]],
    *,
    session: Optional[aiohttp.ClientSession] = None,
) -> Dict[str, str]:
    """Asynchronously analyze news events with the Groq LLM."""

    _ = session  # Retained for compatibility; Groq SDK manages HTTP connections internally.

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
        "model": config.get_news_model(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    if _NEWS_LLM_AUTH_DISABLED:
        return {
            "safe": True,
            "sensitivity": 0.0,
            "reason": _NEWS_LLM_AUTH_REASON,
        }

    try:
        response: Optional[Any] = None
        overflow_model = config.get_overflow_model()
        models_to_try = [payload_template["model"]]
        if payload_template["model"] != overflow_model:
            models_to_try.append(overflow_model)

        for index, model_name in enumerate(models_to_try):
            messages = [
                payload_template["messages"][0],
                {"role": "user", "content": user_prompt},
            ]
            try:
                response = await _chat_completion_async(
                    messages,
                    model=model_name,
                    temperature=0.1,
                    max_tokens=300,
                )
                break
            except GroqAuthError as err:
                _disable_news_llm(str(err))
                return {
                    "safe": True,
                    "sensitivity": 0.0,
                    "reason": _NEWS_LLM_AUTH_REASON,
                }
            except Exception as err:
                if (
                    index == 0
                    and len(models_to_try) > 1
                    and model_name != overflow_model
                ):
                    logger.warning(
                        "Groq model %s failed (%s). Retrying with fallback model %s.",
                        model_name,
                        err,
                        overflow_model,
                    )
                    continue
                logger.error(
                    "Groq news analysis failed with model %s: %s",
                    model_name,
                    err,
                    exc_info=True,
                )
                raise

        if response is None:
            raise RuntimeError("Groq LLM request failed")

        raw_reply = _extract_message_content(response)
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


def fetch_news_sync(symbol: str) -> List[Dict[str, str]]:
    """Synchronous wrapper around :func:`fetch_news`."""

    return _run_coroutine(lambda: fetch_news(symbol))
