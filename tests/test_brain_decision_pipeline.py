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
    monkeypatch.setattr(
        brain,
        "summarize_symbol_news",
        lambda *_, **__: "No material symbol-specific headlines detected in the last 72 hours.",
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


def test_finalize_trade_decision_parses_markdown_json(monkeypatch):
    import brain

    _patch_brain_defaults(monkeypatch)

    sentiment = {"bias": "bullish", "score": 6.5}
    macro_news = {"safe": True, "reason": "Calm conditions"}

    _, prepared = brain.prepare_trade_decision(
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

    assert prepared is not None

    response = """```json\n{\n  \"decision\": \"Yes\",\n  \"confidence\": 7.5,\n  \"reason\": \"Supports breakout\",\n  \"thesis\": \"Trend continuation expected.\"\n}\n```"""

    result = brain.finalize_trade_decision(prepared, response)

    assert result["decision"] is True
    assert result["llm_approval"] is True
    assert result["llm_confidence"] == 7.5


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

    # Ensure the fallback sees a truly high conviction quantitative score.
    prepared.final_confidence = 8.6

    result = brain.finalize_trade_decision(prepared, "LLM error: unavailable")

    assert result["decision"] is True
    assert result["llm_error"] is True
    assert "quant-only auto-approval" in result["reason"].lower()


def test_finalize_trade_decision_handles_json_error(monkeypatch):
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

    prepared.final_confidence = 8.6

    response = json.dumps({"error": "service unavailable"})
    result = brain.finalize_trade_decision(prepared, response)

    assert result["decision"] is True
    assert result["llm_error"] is True
    assert "quant-only auto-approval" in result["reason"].lower()


def test_finalize_trade_decision_error_blocks_weak_signals(monkeypatch):
    import brain

    _patch_brain_defaults(monkeypatch)

    prepared = brain.PreparedTradeDecision(
        symbol="BTCUSDT",
        direction="long",
        session="US",
        setup_type=None,
        score=5.6,
        indicators={"rsi": 68.0},
        sentiment_bias="neutral",
        sentiment_confidence=4.5,
        fear_greed=20,
        macro_news={"safe": True, "reason": ""},
        news_summary="",
        symbol_news_summary="",
        orderflow="neutral",
        auction_state=None,
        pattern_name="",
        pattern_memory_context={"posterior_mean": 0.5, "posterior_variance": 0.09, "trades": 0},
        technical_score=5.2,
        final_confidence=5.9,
        score_threshold=5.5,
        advisor_prompt="",
    )

    result = brain.finalize_trade_decision(prepared, "LLM error: unavailable")

    assert result["decision"] is False
    assert result["llm_error"] is True
    assert "quantitative conviction insufficient" in result["reason"].lower()
