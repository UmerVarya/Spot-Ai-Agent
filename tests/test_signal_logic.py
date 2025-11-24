import numpy as np
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


def test_evaluate_signal_handles_missing_trend_state():
    index = pd.date_range(end=pd.Timestamp.utcnow(), periods=120, freq="T")
    prices = pd.Series(np.linspace(100, 110, 120), index=index)
    df = pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": np.linspace(10, 20, 120),
        },
        index=index,
    )

    score, direction, size, pattern = evaluate_signal(df, symbol="BTCUSDT")

    assert isinstance(score, (int, float))
    assert size is not None
    assert pattern is None or isinstance(pattern, str)


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
    assert "quant-only auto-approval" in result["reason"].lower()
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


def test_evaluate_signal_quiet_in_backtest(caplog):
    index = pd.date_range(end=pd.Timestamp.utcnow(), periods=80, freq="T")
    base_prices = pd.Series(np.linspace(100, 101, len(index)), index=index)
    df = pd.DataFrame(
        {
            "open": base_prices * 0.999,
            "high": base_prices * 1.001,
            "low": base_prices * 0.999,
            "close": base_prices,
            "volume": np.ones(len(index)),
            "quote_volume": np.ones(len(index)) * 10,
        },
        index=index,
    )

    with caplog.at_level("INFO"):
        score, direction, _, _ = evaluate_signal(df, symbol="BTCUSDT", is_backtest=True)

    assert score == 0
    assert direction is None
    assert not any("VOL GATE" in message for message in caplog.messages)
