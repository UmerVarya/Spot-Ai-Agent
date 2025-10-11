from risk_veto import evaluate_risk_veto


def _base_payload(**overrides):
    payload = {
        "direction": "long",
        "btc_trend": "up",
        "time_to_news_minutes": 120,
        "volatility": 0.4,
        "max_rr": 2.0,
    }
    payload.update(overrides)
    return payload


def test_veto_rejects_imminent_news():
    result = evaluate_risk_veto(_base_payload(time_to_news_minutes=15))
    assert result["enter"] is False
    assert "news<30m" in result["conflicts"]


def test_veto_rejects_btc_trend_conflict():
    result = evaluate_risk_veto(_base_payload(btc_trend="down"))
    assert result["enter"] is False
    assert "btc trend conflict" in result["conflicts"]


def test_veto_rejects_volatility_spike():
    result = evaluate_risk_veto(_base_payload(volatility=0.95))
    assert result["enter"] is False
    assert "volatility_spike" in result["conflicts"]
