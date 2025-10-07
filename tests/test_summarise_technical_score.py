from trade_utils import summarise_technical_score


def test_long_bias_rewards_alignment():
    indicators = {
        "rsi": 30.0,
        "macd": 0.06,
        "macd_signal": 0.015,
        "adx": 38.0,
        "di_plus": 32.0,
        "di_minus": 18.0,
    }

    score = summarise_technical_score(indicators, "long")

    assert 6.5 < score <= 10.0


def test_short_bias_rewards_overbought():
    indicators = {
        "rsi": 72.0,
        "macd": -0.04,
        "macd_signal": -0.01,
        "adx": 28.0,
        "di_plus": 15.0,
        "di_minus": 30.0,
    }

    score = summarise_technical_score(indicators, "short")

    assert score > 6.0


def test_counter_trend_high_adx_reduces_score():
    indicators = {
        "rsi": 48.0,
        "macd": -0.01,
        "macd_signal": 0.0,
        "adx": 45.0,
        "di_plus": 12.0,
        "di_minus": 34.0,
    }

    score = summarise_technical_score(indicators, "long")

    assert score < 5.0
    assert 0.0 <= score <= 10.0
