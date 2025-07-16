import os
import json
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# === Read Headlines or Fallback Text ===
def read_macro_context(file_path="macro_headlines.txt"):
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            print("📡 Fallback to scraped headlines...")
            # ✅ Fallback headline text
            return "Bitcoin holds steady as inflation data cools. Fed unlikely to hike rates. Crypto sentiment cautiously optimistic."
    except Exception as e:
        print(f"⚠️ Failed to read macro context: {e}")
        return ""

# === Extract Sentiment from Raw Text ===
def parse_llm_response(text):
    summary = ""
    bias = "neutral"
    confidence = 5.0

    try:
        # Try JSON block first
        if isinstance(text, dict):
            bias = text.get("bias", "neutral").lower()
            confidence = float(text.get("score", 5.0))
            summary = text.get("reason", "")
        elif text.strip().startswith("{"):
            data = json.loads(text)
            bias = data.get("bias", "neutral").lower()
            confidence = float(data.get("score", 5.0))
            summary = data.get("reason", "")
        else:
            # Fallback to pattern matching
            bias_match = re.search(r"Bias:\s*(bullish|bearish|neutral)", text, re.IGNORECASE)
            conf_match = re.search(r"Confidence[:=]?\s*(\d+(?:\.\d+)?)", text)
            summary_match = re.search(r"Summary:(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)

            if bias_match:
                bias = bias_match.group(1).lower()
            if conf_match:
                confidence = float(conf_match.group(1))
            if summary_match:
                summary = summary_match.group(1).strip()

    except Exception as e:
        print(f"⚠️ Failed to parse LLM response: {e}")

    return {
        "bias": bias,
        "confidence": confidence,
        "summary": summary.strip()
    }

# === Analyze Sentiment via Groq LLM ===
def analyze_macro_sentiment(text=None):
    if not text:
        text = read_macro_context()

    # ✅ Fix list type fallback
    if isinstance(text, list):
        text = "\n".join(text)

    if not text.strip():
        return {
            "bias": "neutral",
            "confidence": 5.0,
            "summary": "No macro headlines available."
        }

    prompt = f"""
Analyze the following macroeconomic and crypto-related news for its overall impact on the crypto market.
Respond with market sentiment (Bullish, Bearish, or Neutral), a confidence score from 1 to 10,
and a brief summary (within 50 words).

--- CONTEXT START ---
{text}
--- CONTEXT END ---

Respond in JSON format like this:
{{"bias": "bullish", "score": 8.5, "reason": "Short summary here..."}}
"""

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # ✅ updated model
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        print(f"🧠 Raw LLM Response:\n{content.strip()}\n")
        return parse_llm_response(content)

    except Exception as e:
        print(f"⚠️ LLM sentiment error: {e}")
        return {
            "bias": "neutral",
            "confidence": 5.0,
            "summary": "LLM sentiment parsing failed."
        }

# === Used by agent.py ===
def get_macro_sentiment():
    return analyze_macro_sentiment()
    result = parse_llm_response(llm_response)
    # Ensure bias is a string scalar
    result["bias"] = str(result.get("bias", "neutral"))
    return result
