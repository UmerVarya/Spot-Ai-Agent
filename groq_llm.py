"""
Safe wrapper around the Groq LLM API.

This module extends the basic LLM helper by sanitising user inputs to
mitigate prompt injection and requesting structured JSON output from
the model.  The ``get_llm_judgment`` function now instructs the model
to produce a JSON object with four fields:

* ``decision``: "Yes" or "No" indicating whether to take the trade.
* ``confidence``: a number between 0 and 10 representing the model's
  confidence in that decision.
* ``reason``: a brief explanation supporting the decision.
* ``thesis``: a 2–3 sentence trading thesis summarising the setup.

Downstream callers should attempt to parse the JSON; if parsing fails,
the raw response is returned for backward compatibility.
"""

import os
import requests
import re
import json
import aiohttp
import asyncio
import time
import config
from log_utils import setup_logger
from groq_safe import (
    describe_error,
    extract_error_payload,
    is_model_decommissioned_error,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

logger = setup_logger(__name__)


def _sanitize_prompt(prompt: str, max_len: int = 3000) -> str:
    """Sanitise prompt text to reduce the risk of injection attacks.

    The function removes markdown code fences, braces and other potentially
    malicious characters, strips excessive whitespace and truncates the
    prompt to a maximum length.
    """
    # Remove triple backtick code blocks
    prompt = re.sub(r"```.*?```", "", prompt, flags=re.DOTALL)
    # Remove any braces or JSON‑like delimiters
    prompt = prompt.replace("{", "").replace("}", "")
    # Collapse whitespace
    prompt = re.sub(r"\s+", " ", prompt).strip()
    # Truncate if too long
    if len(prompt) > max_len:
        prompt = prompt[-max_len:]
    return prompt


def get_llm_judgment(prompt: str, temperature: float = 0.4, max_tokens: int = 500) -> str:
    """Query Groq LLM with a prompt asking for trade advice in JSON format.

    The prompt is sanitised before being sent to mitigate injection risks.
    The model is instructed to respond with a JSON object containing the
    decision (Yes/No), confidence (0–10), reason and a 2–3 sentence thesis.
    """
    try:
        safe_prompt = _sanitize_prompt(prompt)
        user_prompt = (
            safe_prompt
            + "\n\nPlease respond in JSON format with the following keys:"
            + " decision (Yes or No), confidence (0 to 10 as a number), reason (a short explanation),"
            + " and thesis (2-3 sentence summary)."
        )
        model = config.get_groq_model()
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a highly experienced crypto trader assistant. Always respond in JSON."},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        start = time.perf_counter()
        response = requests.post(GROQ_API_URL, headers=HEADERS, json=data)
        latency = time.perf_counter() - start
        model_used = data["model"]

        if response.status_code != 200:
            error_detail = extract_error_payload(response)
            if (
                response.status_code == 400
                and model_used != config.DEFAULT_GROQ_MODEL
                and is_model_decommissioned_error(error_detail)
            ):
                fallback_model = config.DEFAULT_GROQ_MODEL
                logger.warning(
                    "Groq model %s unavailable (%s). Retrying with fallback model %s.",
                    model_used,
                    describe_error(error_detail),
                    fallback_model,
                )
                data = {**data, "model": fallback_model}
                start = time.perf_counter()
                response = requests.post(GROQ_API_URL, headers=HEADERS, json=data)
                latency = time.perf_counter() - start
                model_used = fallback_model
                if response.status_code != 200:
                    retry_detail = extract_error_payload(response)
                    logger.error(
                        "LLM request failed in %.2fs: %s, %s",
                        latency,
                        response.status_code,
                        describe_error(retry_detail) or response.text,
                    )
                    return "LLM error: Unable to generate response."
            else:
                logger.error(
                    "LLM request failed in %.2fs: %s, %s",
                    latency,
                    response.status_code,
                    describe_error(error_detail) or response.text,
                )
                return "LLM error: Unable to generate response."

        content = (
            response.json()
            .get("choices", [])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        logger.info("LLM call succeeded in %.2fs", latency)
        return content
    except Exception as e:
        latency = time.perf_counter() - start if 'start' in locals() else 0.0
        logger.error("LLM Exception after %.2fs: %s", latency, e, exc_info=True)
        return "LLM error: Exception occurred."


async def async_get_llm_judgment(prompt: str, temperature: float = 0.4, max_tokens: int = 500) -> str:
    """Asynchronous version of ``get_llm_judgment`` using aiohttp."""
    try:
        safe_prompt = _sanitize_prompt(prompt)
        user_prompt = (
            safe_prompt
            + "\n\nPlease respond in JSON format with the following keys:"
            + " decision (Yes or No), confidence (0 to 10 as a number), reason (a short explanation),"
            + " and thesis (2-3 sentence summary)."
        )
        model = config.get_groq_model()
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a highly experienced crypto trader assistant. Always respond in JSON."},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        start = time.perf_counter()
        async with aiohttp.ClientSession() as session:
            async with session.post(GROQ_API_URL, headers=HEADERS, json=data) as resp:
                latency = time.perf_counter() - start
                if resp.status != 200:
                    error_text = await resp.text()
                    try:
                        error_detail = json.loads(error_text)
                    except Exception:
                        error_detail = error_text
                    if (
                        resp.status == 400
                        and data["model"] != config.DEFAULT_GROQ_MODEL
                        and is_model_decommissioned_error(error_detail)
                    ):
                        fallback_model = config.DEFAULT_GROQ_MODEL
                        logger.warning(
                            "Groq model %s unavailable (%s). Retrying with fallback model %s.",
                            data["model"],
                            describe_error(error_detail),
                            fallback_model,
                        )
                        retry_payload = {**data, "model": fallback_model}
                        start = time.perf_counter()
                        async with session.post(
                            GROQ_API_URL, headers=HEADERS, json=retry_payload
                        ) as retry_resp:
                            latency = time.perf_counter() - start
                            if retry_resp.status == 200:
                                result = await retry_resp.json()
                                logger.info("Async LLM call succeeded in %.2fs", latency)
                                return (
                                    result.get("choices", [])[0]
                                    .get("message", {})
                                    .get("content", "")
                                    .strip()
                                )
                            retry_text = await retry_resp.text()
                            try:
                                retry_detail = json.loads(retry_text)
                            except Exception:
                                retry_detail = retry_text
                            logger.error(
                                "LLM request failed in %.2fs: %s, %s",
                                latency,
                                retry_resp.status,
                                describe_error(retry_detail) or retry_text,
                            )
                            return "LLM error: Unable to generate response."
                    logger.error(
                        "LLM request failed in %.2fs: %s, %s",
                        latency,
                        resp.status,
                        describe_error(error_detail) or error_text,
                    )
                    return "LLM error: Unable to generate response."

                result = await resp.json()
                logger.info("Async LLM call succeeded in %.2fs", latency)
                return (
                    result.get("choices", [])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
    except Exception as e:
        latency = time.perf_counter() - start if 'start' in locals() else 0.0
        logger.error("Async LLM Exception after %.2fs: %s", latency, e, exc_info=True)
        return "LLM error: Exception occurred."
