import time

import pandas as pd

from realtime_signal_cache import RealTimeSignalCache
from trade_utils import compute_macro_score_adjustments


async def _dummy_price_fetcher(symbol: str):  # pragma: no cover - helper for tests
    return None


def test_compute_macro_adjustment_penalizes_bearish_stack():
    delta, snapshot = compute_macro_score_adjustments(
        {"fear_greed": 10, "btc_dom": 61.0, "macro_bias": "bearish"}
    )
    assert snapshot["fear_greed"] == 10
    assert snapshot["btc_dom"] == 61.0
    assert delta < 0
    assert "btc_dom_60" in snapshot["macro_flags"]


def test_compute_macro_adjustment_rewards_bullish_bias():
    delta, snapshot = compute_macro_score_adjustments(
        {"fear_greed": 70, "macro_bias": "bullish"}
    )
    assert delta > 0
    assert snapshot["macro_bias"] == "bullish"


def test_signal_cache_passes_macro_context_to_evaluator():
    captured: dict[str, object] = {}

    def _fake_evaluator(price_data, symbol, **kwargs):
        captured["macro_context"] = kwargs.get("macro_context")
        return 0.0, "long", 0.0, None

    cache = RealTimeSignalCache(
        _dummy_price_fetcher,
        _fake_evaluator,
        refresh_interval=1.0,
        stale_after=10.0,
        use_streams=False,
    )
    cache.update_context(sentiment_bias="neutral", macro={"fear_greed": 25, "btc_dom": 55.0})
    frame = pd.DataFrame({
        "open": [1.0, 1.0],
        "high": [1.0, 1.0],
        "low": [1.0, 1.0],
        "close": [1.0, 1.0],
        "volume": [100.0, 110.0],
    })
    cache._update_cache("TEST", frame, attempt_ts=time.time(), prev_age=None)
    macro_ctx = captured.get("macro_context")
    assert isinstance(macro_ctx, dict)
    assert macro_ctx.get("fear_greed") == 25
    assert macro_ctx.get("btc_dom") == 55.0


def test_run_backtest_evaluate_signal_provides_macro(monkeypatch):
    import run_backtest

    captured: dict[str, object] = {}

    def _fake_eval(price_data, symbol, **kwargs):
        captured["macro_context"] = kwargs.get("macro_context")
        return 1.0, "long", 0.5, {}

    monkeypatch.setattr(run_backtest, "evaluate_signal_live", _fake_eval)
    monkeypatch.setattr(
        run_backtest,
        "get_macro_context",
        lambda: {
            "fear_greed": 33,
            "fear_greed_age_seconds": 120,
            "btc_dominance": 59.5,
            "btc_age_seconds": 240,
            "macro_sentiment": "bearish",
        },
    )

    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        }
    )

    run_backtest.evaluate_signal(frame, symbol="TEST")
    macro_ctx = captured.get("macro_context")
    assert isinstance(macro_ctx, dict)
    assert macro_ctx["fear_greed"] == 33
    assert macro_ctx["btc_dom"] == 59.5
    assert macro_ctx["macro_bias"] == "bearish"
