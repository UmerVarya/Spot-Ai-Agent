import math

from local_llm import (
    _BTC_TREND_TOLERANCE,
    _NEWS_VETO_MINUTES,
    _VOLATILITY_SPIKE_THRESHOLD,
    evaluate_risk_rules,
)


def _base_payload():
    return {
        "symbol": "ETHUSDT",
        "direction": "long",
        "confidence": 7.2,
        "ml_probability": 0.61,
        "volatility": 0.4,
        "htf_trend_pct": 1.3,
        "btc_trend_pct": 0.2,
        "minutes_to_news": 120.0,
        "max_rr": 1.6,
    }


def test_evaluate_risk_rules_blocks_imminent_news():
    payload = _base_payload()
    payload["minutes_to_news"] = max(0.0, _NEWS_VETO_MINUTES - 5.0)
    result = evaluate_risk_rules(payload)

    assert result["enter"] is False
    assert "news<30m" in result["conflicts"]
    assert any("event" in reason.lower() for reason in result["reasons"])


def test_evaluate_risk_rules_blocks_btc_trend_conflict():
    payload = _base_payload()
    payload["btc_trend_pct"] = -(_BTC_TREND_TOLERANCE + 0.05)
    result = evaluate_risk_rules(payload)

    assert result["enter"] is False
    assert "btc_trend_conflict" in result["conflicts"]


def test_evaluate_risk_rules_blocks_volatility_spike():
    payload = _base_payload()
    payload["volatility"] = _VOLATILITY_SPIKE_THRESHOLD + 0.05
    result = evaluate_risk_rules(payload)

    assert result["enter"] is False
    assert "volatility_spike" in result["conflicts"]
    assert math.isclose(result["max_rr"], 1.0, rel_tol=1e-6)
