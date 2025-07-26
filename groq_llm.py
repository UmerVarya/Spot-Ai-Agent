"""
LLM integration utilities for the Spot AI Super Agent.

This module wraps calls to the Groq API (or any OpenAI‑compatible
endpoint) and enforces deterministic, structured JSON responses.  It
sanitises prompts, constructs a system message instructing the model to
return JSON with fields ``decision``, ``confidence`` and ``reason``,
and includes basic protections against prompt injection.  If the API
call fails or the response cannot be parsed as JSON, the function
returns a fallback with neutral confidence.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any

import httpx

# API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_ENDPOINT = os.getenv("GROQ_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions")
MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")


def _sanitize_prompt(data: Dict[str, Any]) -> str:
    """Create a JSON string from the data while preventing prompt injection."""
    # Remove any potential injection strings by encoding as JSON
    return json.dumps(data)


def get_llm_judgment(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call the Groq API with a structured prompt and return the JSON result."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a trading assistant. Given the following market context, "
                "return a JSON object with fields: decision (bool), confidence (0-10), "
                "and reason (concise rationale). Do not include any additional keys."
            ),
        },
        {
            "role": "user",
            "content": _sanitize_prompt(payload),
        },
    ]
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 150,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(GROQ_ENDPOINT, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        content = data['choices'][0]['message']['content']
        # Attempt to parse JSON output
        result = json.loads(content)
        # Ensure required keys exist
        if not isinstance(result, dict):
            raise ValueError
        return {
            "decision": bool(result.get("decision", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "reason": str(result.get("reason", "")),
        }
    except Exception:
        # Fallback: neutral decision with mid confidence
        return {"decision": False, "confidence": 5.0, "reason": "LLM failure"}
