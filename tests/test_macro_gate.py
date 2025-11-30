from macro_gate import evaluate_macro_gate


def test_macro_gate_passes_when_news_safe_none_defaults_allow():
    decision = evaluate_macro_gate({}, strict_mode=True)
    assert decision.macro_ok is True
    assert decision.strict_macro_ok is True
    assert "news_safe_missing_default_allow" in decision.reasons


def test_macro_gate_blocks_on_explicit_halt_in_strict_mode():
    decision = evaluate_macro_gate({"safe": False, "reason": "alert"}, strict_mode=True)
    assert decision.macro_ok is False
    assert decision.strict_macro_ok is False
    assert "news_marked_unsafe" in decision.reasons


def test_macro_gate_soft_mode_allows_but_records_reasons():
    decision = evaluate_macro_gate({"safe": False, "reason": "alert"}, strict_mode=False)
    assert decision.macro_ok is False
    assert decision.strict_macro_ok is True
    assert "news_marked_unsafe" in decision.reasons


def test_macro_gate_tracks_macro_filter_reasons():
    decision = evaluate_macro_gate(
        {"safe": True}, skip_alt=True, macro_filter_reasons=["low FNG"], strict_mode=True
    )
    assert decision.macro_ok is True
    assert decision.strict_macro_ok is True
    assert "macro_filter_skip_alt" in decision.reasons
    assert "macro_filter:low FNG" in decision.reasons
