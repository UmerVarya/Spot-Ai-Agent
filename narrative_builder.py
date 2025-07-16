# narrative_builder.py

from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

def generate_trade_narrative(trade_data: dict) -> str:
    prompt = f"""
You are a professional crypto trading assistant. Summarize the reasoning for entering a trade based on the following data:

Symbol: {trade_data.get('symbol')}
Direction: {trade_data.get('direction')}
Confidence Score: {trade_data.get('confidence')}
Pattern Detected: {trade_data.get('pattern')}
Macro Sentiment: {trade_data.get('macro_sentiment')}
BTC Dominance: {trade_data.get('btc_dominance')}
Fear & Greed Index: {trade_data.get('fear_greed')}
Session: {trade_data.get('session')}
Order Flow Pressure: {trade_data.get('orderflow')}
Support/Resistance Zone Match: {trade_data.get('zone_match')}
Trade Rationale:
- Indicator Score: {trade_data.get('score')}
- Reinforcement Weighting: {trade_data.get('reinforcement_weight')}
- Historical Pattern Memory Confidence: {trade_data.get('pattern_memory')}
- Volume Strength: {trade_data.get('volume_strength')}

Write a short, confident explanation justifying the trade in plain English. End with a recommendation (like "Good trade setup", "High-risk trade", etc.).
"""

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating narrative: {e}"
