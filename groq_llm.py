"""
Safe wrapper around the Groq LLM API.

This module extends the basic LLM helper by sanitising user inputs to
mitigate prompt injection and requesting structured JSON output from
the model.  The ``get_llm_judgment`` function now instructs the model
to produce a JSON object with three fields:

* ``decision``: "Yes" or "No" indicating whether to take the trade.
* ``confidence``: a number between 0 and 10 representing the model's
  confidence in that decision.
* ``reason``: a brief explanation supporting the decision.

Downstream callers should attempt to parse the JSON; if parsing fails,
the raw response is returned for backward compatibility.
"""

import os
import requests
import re
import json

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}


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
    decision (Yes/No), confidence (0–10) and reason.
    """
    try:
        safe_prompt = _sanitize_prompt(prompt)
        user_prompt = (
            safe_prompt
            + "\n\nPlease respond in JSON format with the following keys:"
            + " decision (Yes or No), confidence (0 to 10 as a number) and reason (a short explanation)."
        )
        data = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a highly experienced crypto trader assistant. Always respond in JSON."},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(GROQ_API_URL, headers=HEADERS, json=data)
        if response.status_code == 200:
            content = response.json().get("choices", [])[0].get("message", {}).get("content", "").strip()
            return content
        else:
            print(f"❌ LLM request failed: {response.status_code}, {response.text}")
            return "LLM error: Unable to generate response."
    except Exception as e:
        print(f"❌ LLM Exception: {e}")
        return "LLM error: Exception occurred."
