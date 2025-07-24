import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b")

# Cache for latest macro sentiment
latest_macro_sentiment = {"bias": "neutral", "confidence": 5.0, "summary": "No analysis yet."}
try:
    with open("macro_sentiment.json", "r") as f:
        latest_macro_sentiment = json.load(f)
except Exception:
    pass

def analyze_macro_news(news_text: str) -> dict:
    """Analyze combined macro news headlines using the Groq LLM API and return bias, confidence, summary."""
    if not GROQ_API_KEY:
        # No API key available, return default neutral sentiment
        return {"bias": "neutral", "confidence": 5.0, "summary": "LLM analysis not available"}
    # Prepare the prompt for the LLM
    prompt = (
        "You are a financial analyst. You will be given a series of news headlines about the markets and economy.\n"
        "Analyze the overall market sentiment reflected by these news (bullish, bearish, or neutral) and provide a JSON output with keys 'bias', 'confidence', and 'summary'.\n"
        "Respond ONLY with a JSON object in the format: "
        "{\"bias\": <bias>, \"confidence\": <confidence>, \"summary\": \"<short summary>\"} and nothing else (no markdown or extra text)."
    )
    messages = [
        {"role": "user", "content": prompt + "\nNews Headlines:\n" + news_text}
    ]
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 200
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)
    except Exception as e:
        print(f"❌ Error connecting to Groq API: {e}")
        return {"bias": "neutral", "confidence": 5.0, "summary": "No analysis (API error)"}
    if not response.ok:
        print(f"❌ Groq API returned error: {response.status_code}, {response.text}")
        return {"bias": "neutral", "confidence": 5.0, "summary": "No analysis (API error)"}
    # Extract the content from the API response
    result_text = ""
    try:
        result_json = response.json()
        result_text = result_json.get("choices", [])[0].get("message", {}).get("content", "")
    except Exception as e:
        print(f"⚠️ Failed to parse Groq API response: {e}")
        result_text = response.text or ""
    # ✅ Robust JSON parsing from LLM output
    result_text = result_text.strip()
    parsed = {}
    if result_text:
        # Remove any markdown code fences or extraneous text
        if result_text.startswith("```"):
            # Strip markdown formatting
            result_text = result_text.strip('`')
            if result_text.lower().startswith("json"):
                brace_index = result_text.find('{')
                if brace_index != -1:
                    result_text = result_text[brace_index:]
        # Try direct JSON parsing
        try:
            parsed = json.loads(result_text)
        except Exception:
            # Attempt regex extraction of JSON
            match = re.search(r'\{.*\}', result_text)
            if match:
                json_str = match.group(0)
                try:
                    parsed = json.loads(json_str)
                except Exception:
                    # Last resort: find first and last curly braces
                    first = result_text.find('{')
                    last = result_text.rfind('}')
                    if first != -1 and last != -1 and last > first:
                        json_str = result_text[first:last+1]
                        try:
                            parsed = json.loads(json_str)
                        except Exception:
                            parsed = {}
            else:
                parsed = {}
    # Ensure required fields with defaults
    if not parsed or "bias" not in parsed:
        parsed["bias"] = "neutral"
    if "confidence" not in parsed:
        parsed["confidence"] = 5.0
    if "summary" not in parsed:
        parsed["summary"] = "No summary available"
    # Normalize data types
    try:
        parsed["confidence"] = float(parsed["confidence"])
    except:
        parsed["confidence"] = 5.0
    parsed["bias"] = str(parsed["bias"]).lower()
    return parsed

def get_macro_sentiment() -> dict:
    """Retrieve the latest macro sentiment analysis results."""
    return latest_macro_sentiment
