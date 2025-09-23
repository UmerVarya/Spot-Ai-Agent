import os
from groq import Groq
from dotenv import load_dotenv
import config
from groq_safe import safe_chat_completion

load_dotenv()

# === Load Groq API Key ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


def generate_narrative(trade):
    """
    Generate an LLM-based explanation for why this trade was taken.
    """
    try:
        symbol = trade.get("symbol")
        direction = trade.get("direction")
        score = trade.get("confidence")
        indicators = trade.get("indicators", {})
        macro_summary = trade.get("macro_summary", "Unknown")
        macro_bias = trade.get("macro_bias", "neutral")
        macro_conf = trade.get("macro_confidence", 0)
        headlines = trade.get("news_headlines", [])

        prompt = f"""
You are a professional quant trader assistant.
Analyze the following trade setup and generate a short, high-quality explanation like a trading journal entry.

Include:
- Reason for entry (based on indicators)
- Macro sentiment
- News influence (if any)
- Overall tone: objective, neutral, informative

Return your response in this format:

🧠 Trade Narrative ({symbol}, {direction.upper()})
Entry Decision: <indicator alignment>
Macro Sentiment: <bias> (Confidence: <score>)
News Influence: <summary of any news if present>
Final Thoughts: <objective commentary>

Trade Details:
- Score: {score}
- Indicators: {indicators}
- Macro Summary: {macro_summary}
- News Headlines: {headlines[:3]}  # Use top 3 headlines only
"""

        response = safe_chat_completion(
            client,
            model=config.get_groq_model(),
            messages=[
                {"role": "system", "content": "You are a professional crypto trading strategist."},
                {"role": "user", "content": prompt}
            ],
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Narrative generation failed: {e}"
