import json


def _patch_brain_defaults(monkeypatch):
    import brain

    monkeypatch.setattr(brain, "summarise_technical_score", lambda *_: 6.0, raising=False)
    monkeypatch.setattr(brain, "get_adaptive_conf_threshold", lambda: 5.0, raising=False)
    monkeypatch.setattr(brain, "recall_pattern_confidence", lambda *_, **__: 0.1, raising=False)
    monkeypatch.setattr(
        brain,
        "get_pattern_posterior_stats",
        lambda *_, **__: {
            "mean": 0.55,
            "variance": 0.02,
            "alpha": 3.0,
            "beta": 2.0,
            "trades": 10,
        },
        raising=False,
    )
    monkeypatch.setattr(
        brain,
        "calculate_historical_confidence",
        lambda *_, **__: {"confidence": 60.0},
        raising=False,
    )
    monkeypatch.setattr(
        brain,
        "get_recent_trade_summary",
        lambda *_, **__: "Recent success",
        raising=False,
    )
    monkeypatch.setattr(
        brain,
        "generate_trade_narrative",
        lambda **_: "Narrative from generator",
        raising=False,
    )


def test_prepare_and_finalize_trade_decision(monkeypatch):
    import brain

    _patch_brain_defaults(monkeypatch)

    sentiment = {"bias": "bullish", "score": 6.5}
    macro_news = {"safe": True, "reason": "Calm conditions"}

    pre_result, prepared = brain.prepare_trade_decision(
        symbol="ETHUSDT",
        score=7.2,
        direction="long",
        indicators={"rsi": 55.0, "macd": 0.4, "adx": 28.0},
        session="US",
        pattern_name="hammer",
        orderflow="buyers",
        sentiment=sentiment,
        macro_news=macro_news,
        volatility=0.5,
        fear_greed=60,
        auction_state="trending",
        setup_type="trend",
        news_summary="Macro looks calm",
    )

    assert pre_result is None
    assert prepared is not None

    response = json.dumps(
        {
            "decision": "Yes",
            "confidence": 8.0,
            "reason": "Momentum aligned",
            "thesis": "Upside momentum building.",
        }
    )

    result = brain.finalize_trade_decision(prepared, response)

    assert result["decision"] is True
    assert result["llm_approval"] is True
    assert result["confidence"] > 0
    assert result["narrative"]


def test_finalize_trade_decision_handles_error(monkeypatch):
    import brain

    _patch_brain_defaults(monkeypatch)

    sentiment = {"bias": "neutral", "score": 5.0}
    macro_news = {"safe": True, "reason": "Quiet"}

    _, prepared = brain.prepare_trade_decision(
        symbol="BTCUSDT",
        score=6.0,
        direction="long",
        indicators={"rsi": 52.0, "macd": 0.2, "adx": 25.0},
        session="EU",
        pattern_name="marubozu_bullish",
        orderflow="buyers",
        sentiment=sentiment,
        macro_news=macro_news,
        volatility=0.4,
        fear_greed=50,
        auction_state="trending",
        setup_type="trend",
        news_summary="Quiet",
    )

    assert prepared is not None

    result = brain.finalize_trade_decision(prepared, "LLM error: unavailable")

    assert result["decision"] is True
    assert result["llm_error"] is True
    assert "auto-approval" in result["reason"].lower()
