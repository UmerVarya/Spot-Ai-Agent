import pandas as pd

def detect_aggression(df: pd.DataFrame) -> str:
    """
    Detect order flow aggression based on price and volume behavior.

    A bullish scenario occurs when price is rising and the most recent candle's volume is
    meaningfully above the recent average. Likewise, a bearish scenario is when price
    drops under the same volume conditions.  This function returns strings that are
    compatible with the downstream trading logic used by agent.py and trade_utils.evaluate_signal.

    Returns
    -------
    str
        "buyers in control" if there's clear aggressive buying,
        "sellers in control" if there's clear aggressive selling,
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
        return "buyers in control"
    elif bearish:
        return "sellers in control"
    else:
        return "neutral"
