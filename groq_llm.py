"""
Wrapper for interacting with the Groq LLM.

This module centralises all interactions with the language model and enforces
a structured JSON output format to improve determinism and reduce prompt
injection risk.  The ``get_llm_judgment`` function constructs a prompt,
submits it to the Groq API using the provided API key and model, and
parses the response into a Python dict.  If the API call fails or the
response cannot be parsed, sensible defaults are returned.
"""

from __future__ import annotations
import json
import os
import re
from typing import Dict, Any

import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b")


def _call_llm(messages: list[dict[str, str]], temperature: float = 0.2, max_tokens: int = 200) -> str:
    """Low‑level helper to call the Groq chat API and return the raw content."""
    if not GROQ_API_KEY:
        return ""
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        return ""


def get_llm_judgment(prompt: str) -> Dict[str, Any]:
    """Query the LLM with a prompt and return a structured judgment.

    The LLM is instructed to respond with JSON containing keys
    ``decision`` (boolean), ``confidence`` (0–10) and ``reason`` (string).
    This helper sanitises the prompt by stripping triple backticks and
    limiting its length to mitigate prompt injection.
    """
    # Sanitize prompt: remove markdown fences and limit length
    clean_prompt = re.sub(r"```.*?```", "", prompt, flags=re.DOTALL)[:4000]
    system_msg = {
        "role": "system",
        "content": "You are a quantitative crypto trading assistant. "
                   "Given a trade setup, return a JSON with keys decision (true/false), "
                   "confidence (0-10) and reason (concise explanation). Do not include any extra text.",
    }
    user_msg = {
        "role": "user",
        "content": clean_prompt,
    }
    raw = _call_llm([system_msg, user_msg], temperature=0.2, max_tokens=150)
    raw = raw.strip() if raw else ""
    # Attempt to parse JSON directly
    if raw:
        # Remove code fences
        if raw.startswith("```"):
            raw = raw.strip('`')
            if raw.lower().startswith("json"):
                brace = raw.find('{')
                raw = raw[brace:] if brace != -1 else raw
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                return {
                    "decision": bool(result.get("decision", False)),
                    "confidence": float(result.get("confidence", 0.0)),
                    "reason": str(result.get("reason", "No reason provided.")),
                }
        except Exception:
            pass
    # Fallback: naive extraction from any numbers and yes/no
    dec = raw.lower().startswith("yes") if raw else False
    match = re.search(r"(\d+(?:\.\d+)?)", raw) if raw else None
    conf = float(match.group(1)) if match else 0.0
    return {"decision": dec, "confidence": conf, "reason": raw or "No response"}
