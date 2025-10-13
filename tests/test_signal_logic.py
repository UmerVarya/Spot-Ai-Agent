import pandas as pd

from brain import should_trade
from trade_utils import evaluate_signal


def test_evaluate_signal_missing_columns():
    df = pd.DataFrame({
        "open": [1, 2, 3, 4],
        "high": [1, 2, 3, 4],
        "low": [1, 2, 3, 4],
        "close": [1, 2, 3, 4],
    })
    score, direction, size, pattern = evaluate_signal(df, symbol="TEST")
    assert score == 0
    assert direction is None
    assert size == 0
    assert pattern is None


def test_should_trade_reject_non_long():
    result = should_trade(
        symbol="BTCUSDT",
        score=6.0,
        direction="short",
        indicators={},
        session="US",
        pattern_name="",
        orderflow="neutral",
        sentiment={"bias": "neutral"},
        macro_news={"safe": True, "reason": ""},
    )
    assert result["decision"] is False
    assert "not long" in result["reason"].lower()


def test_should_trade_requires_strong_quant_on_llm_error():
    result = should_trade(
        symbol="BTCUSDT",
        score=6.0,
        direction="long",
        indicators={},
        session="US",
        pattern_name="",
        orderflow="neutral",
        sentiment={"bias": "neutral"},
        macro_news={"safe": True, "reason": ""},
    )
    assert result["decision"] is True
    assert "fallback thresholds" in result["reason"].lower()
    assert result.get("llm_error") is True


def test_should_trade_rejects_in_extreme_fear():
    result = should_trade(
        symbol="BTCUSDT",
        score=6.0,
        direction="long",
        indicators={},
        session="US",
        pattern_name="",
        orderflow="neutral",
        sentiment={"bias": "neutral"},
        macro_news={"safe": True, "reason": ""},
        fear_greed=10,
    )
    assert result["decision"] is False


def test_should_trade_demands_more_in_low_vol():
    result = should_trade(
        symbol="BTCUSDT",
        score=5.6,
        direction="long",
        indicators={},
        session="US",
        pattern_name="",
        orderflow="neutral",
        sentiment={"bias": "neutral"},
        macro_news={"safe": True, "reason": ""},
        volatility=0.1,
    )
    assert result["decision"] is False


def test_should_trade_skips_breakout_in_balanced_regime():
    result = should_trade(
        symbol="ETHUSDT",
        score=6.5,
        direction="long",
        indicators={},
        session="US",
        pattern_name="flag",
        orderflow="neutral",
        sentiment={"bias": "neutral"},
        macro_news={"safe": True, "reason": ""},
        auction_state="balanced",
        setup_type="trend",
    )
    assert result["decision"] is False
    assert "balanced" in result["reason"].lower()
