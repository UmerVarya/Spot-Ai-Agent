import os
from news_scraper import get_combined_headlines
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# === Load Groq API Key from Environment ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

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
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a crypto macro market analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        result = response.choices[0].message.content.strip()
        print("🧠 Raw LLM Response:\n", result)  # Optional debug

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
        print(f"❌ News sentiment LLM error: {e}")
        return {
            "summary": "Error during LLM analysis.",
            "bias": "neutral",
            "confidence": 0
        }

# === Optional test ===
if __name__ == "__main__":
    result = analyze_macro_sentiment()
    print("\n🧠 LLM Market Sentiment Analysis")
    print(f"Summary: {result['summary']}")
    print(f"Bias: {result['bias'].upper()} | Confidence: {result['confidence']}")
