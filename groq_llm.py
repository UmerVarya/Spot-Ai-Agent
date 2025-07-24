"""
Helper module to interact with the Groq LLM API for trade decision support.

This module defines a single function, `get_llm_judgment`, which sends a prompt to
Groq's hosted LLaMA3 model and returns the text of its response.  The prompt is
augmented with instructions so that the language model returns a simple yes/no
decision, a numerical confidence rating (0–10) and a brief reason.  Downstream
logic can parse these elements to weight the AI's opinion when determining
whether to enter a trade.
"""

import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set this in your environment
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def get_llm_judgment(prompt: str, temperature: float = 0.4, max_tokens: int = 500) -> str:
    """
    Query Groq LLM with a prompt asking for trade advice.

    The language model is instructed to answer with `Yes` or `No` to indicate
    whether the trade should be taken, followed by a numeric confidence rating
    from 0 to 10 and a brief reason.  The rating allows the calling code to
    combine the model's perceived confidence with its own quantitative score.

    Parameters
    ----------
    prompt : str
        The trade-specific information to present to the model.
    temperature : float, optional
        Sampling temperature for the LLM.  Lower values make the output more
        deterministic.  Defaults to 0.4.
    max_tokens : int, optional
        The maximum number of tokens to return from the LLM.  Defaults to 500.

    Returns
    -------
    str
        The raw content returned by the model.  Caller is responsible for
        parsing the yes/no decision, confidence rating and reason.
    """
    try:
        # Append instructions to the user prompt to enforce structured replies.
        user_prompt = (
            prompt
            + "\n\nPlease respond with 'Yes' or 'No' to indicate whether to take the trade, "
            + "followed by a numerical confidence rating from 0 to 10 and a brief reason. "
            + "Example: Yes 8.0 - The technicals align and momentum is strong."
        )
        data = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a highly experienced crypto trader assistant."},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(GROQ_API_URL, headers=HEADERS, json=data)
        if response.status_code == 200:
            return response.json().get("choices", [])[0].get("message", {}).get("content", "").strip()
        else:
            print(f"❌ LLM request failed: {response.status_code}, {response.text}")
            return "LLM error: Unable to generate response."
    except Exception as e:
        print(f"❌ LLM Exception: {e}")
        return "LLM error: Exception occurred."
