import pandas as pd

def detect_aggression(df: pd.DataFrame) -> str:
    """
    Detect order flow aggression based on price and volume behavior.

    Returns:
        "bullish" if there's clear aggressive buying,
        "bearish" if there's clear aggressive selling,
        "neutral" otherwise.
    """
    if df is None or df.empty or len(df) < 5:
        return "neutral"

    # Take last 5 candles
    recent = df.tail(5)

    # Calculate average volume and price direction
    avg_volume = recent['volume'].mean()
    price_change = recent['close'].iloc[-1] - recent['open'].iloc[0]

    # Check for bullish aggression: rising price + rising volume
    bullish = price_change > 0 and recent['volume'].iloc[-1] > avg_volume * 1.2

    # Check for bearish aggression: falling price + rising volume
    bearish = price_change < 0 and recent['volume'].iloc[-1] > avg_volume * 1.2

    if bullish:
        return "bullish"
    elif bearish:
        return "bearish"
    else:
        return "neutral"
