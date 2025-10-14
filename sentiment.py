import os
import json
from dotenv import load_dotenv

from json_utils import parse_llm_json_response

from groq import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from log_utils import setup_logger
import config
from groq_client import get_groq_client
from groq_safe import safe_chat_completion

load_dotenv()

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Cache for latest macro sentiment
latest_macro_sentiment = {"bias": "neutral", "confidence": 5.0, "summary": "No analysis yet."}
try:
    with open("macro_sentiment.json", "r") as f:
        latest_macro_sentiment = json.load(f)
except Exception:
    pass

logger = setup_logger(__name__)

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
    model_name = config.get_macro_model()
    client = get_groq_client()
    if client is None:
        return {"bias": "neutral", "confidence": 5.0, "summary": "LLM analysis not available"}

    try:
        response = safe_chat_completion(
            client,
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=200,
        )
        result_text = response.choices[0].message.content if response.choices else ""
    except RateLimitError as err:
        logger.warning("Groq rate limited during macro analysis: %s", err)
        return {"bias": "neutral", "confidence": 5.0, "summary": "No analysis (rate limited)"}
    except (APIStatusError, APIConnectionError, APITimeoutError, APIError) as err:
        logger.error("Groq API error during macro analysis: %s", err)
        return {"bias": "neutral", "confidence": 5.0, "summary": "No analysis (API error)"}
    except Exception as err:
        logger.error("Error connecting to Groq API: %s", err, exc_info=True)
        return {"bias": "neutral", "confidence": 5.0, "summary": "No analysis (API error)"}

    parsed, _ = parse_llm_json_response(
        result_text,
        defaults={
            "bias": "neutral",
            "confidence": 5.0,
            "summary": "No summary available",
        },
        logger=logger,
    )
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
