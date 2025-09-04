import os
from news_scraper import get_combined_headlines
from groq import Groq
from log_utils import setup_logger

# Centralised configuration loader
import config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = config.get_groq_model()
client = Groq(api_key=GROQ_API_KEY)

logger = setup_logger(__name__)

def analyze_macro_sentiment():
    headlines = get_combined_headlines()

    if not headlines:
        return {
            "summary": "No news headlines available.",
            "bias": "neutral",
            "confidence": 0
        }

    prompt = f"""
You are a professional macroeconomic and crypto news analyst.
Given the following headlines, analyze and summarize the market sentiment.

Headlines:
{headlines}

Return your response exactly in this format:
Summary: <concise summary>
Bias: <bullish / bearish / neutral>
Confidence: <0-10 score>
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a crypto macro market analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        result = response.choices[0].message.content.strip()
        logger.debug("Raw LLM Response: %s", result)

        summary = "No summary extracted"
        bias = "neutral"
        confidence = 0

        for line in result.splitlines():
            lower = line.lower()
            if "summary:" in lower:
                summary = line.split(":", 1)[-1].strip()
            elif "bias:" in lower:
                b = line.split(":", 1)[-1].strip().lower()
                if b in ["bullish", "bearish", "neutral"]:
                    bias = b
            elif "confidence:" in lower:
                try:
                    confidence = float(line.split(":", 1)[-1].strip())
                except:
                    confidence = 0

        return {
            "summary": summary,
            "bias": bias,
            "confidence": confidence
        }

    except Exception as e:
        logger.error("News sentiment LLM error: %s", e, exc_info=True)
        return {
            "summary": "Error during LLM analysis.",
            "bias": "neutral",
            "confidence": 0
        }

# === Optional test ===
if __name__ == "__main__":
    result = analyze_macro_sentiment()
    logger.info("LLM Market Sentiment Analysis")
    logger.info("Summary: %s", result['summary'])
    logger.info("Bias: %s | Confidence: %s", result['bias'].upper(), result['confidence'])
