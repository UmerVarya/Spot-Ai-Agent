"""
Updated narrative builder for Spot AI Super Agent.

This version of ``narrative_builder`` addresses a mismatch between how the
``brain`` module invokes the trade narrative function and the original
implementation.  The original ``generate_trade_narrative`` accepted only a
single dictionary argument called ``trade_data``.  In practice, the agent
passes individual keyword arguments (``symbol``, ``direction``, ``score``
etc.) directly to this function.  That mismatch caused errors like
``generate_trade_narrative() got an unexpected keyword argument 'symbol'``.

To resolve this, the updated implementation accepts arbitrary positional
and keyword arguments, coalesces them into a single ``trade_data``
dictionary, and then builds the LLM prompt.  No other changes to the
business logic have been made.
"""

from groq import Groq
import os

from dotenv import load_dotenv
import config
from groq_safe import safe_chat_completion

load_dotenv()

# Retrieve API key from environment and initialise the client once
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


def generate_trade_narrative(*args, **kwargs) -> str:
    """Generate a concise trade narrative for a given trading setup.

    The Spot AI agent calls this function with keyword arguments such as
    ``symbol``, ``direction``, ``score``, etc.  If the original dict-based
    signature is used, the first positional argument can be that dict.

    Args:
        *args: Optional positional arguments.  If the first argument is a
            dictionary, it will be treated as ``trade_data``.
        **kwargs: Individual trade attributes passed by the caller.  These
            override values in the first positional dictionary if provided.

    Returns:
        A narrative string describing the trade rationale, or a warning
        message if an exception occurs during LLM invocation.
    """
    # Normalise inputs: support both dict-based and keyword-based calls
    trade_data = {}
    if args and isinstance(args[0], dict):
        # Start with the provided dictionary
        trade_data.update(args[0])
    # Overlay any keyword arguments supplied
    trade_data.update(kwargs)

    # Compose the prompt.  Use ``dict.get`` to avoid KeyError if a field
    # happens to be missing.  Maintain the original prompt structure.
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
        response = safe_chat_completion(
            client,
            model=config.get_groq_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating narrative: {e}"
