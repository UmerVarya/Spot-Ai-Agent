import pandas as pd
from trade_utils import evaluate_signal
from brain import should_trade


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


def test_should_trade_auto_approves_on_llm_error():
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
    assert "auto-approval" in result["reason"].lower()
