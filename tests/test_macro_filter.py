from agent import macro_filter_decision


def test_macro_filter_extreme_fear():
    skip_all, skip_alt, reasons = macro_filter_decision(40.0, 5, "bearish", 9.0)
    assert skip_all and not skip_alt
    assert any("extreme fear" in r for r in reasons)


def test_macro_filter_alt_risk():
    skip_all, skip_alt, reasons = macro_filter_decision(65.0, 15, "bearish", 7.0)
    assert not skip_all and skip_alt
    assert "very high BTC dominance" in " ".join(reasons)


def test_macro_filter_alt_risk_missing_fear_greed():
    skip_all, skip_alt, reasons = macro_filter_decision(65.0, None, "bearish", 7.0)
    assert not skip_all and skip_alt
    joined = " ".join(reasons)
    assert "very high BTC dominance" in joined
    assert "Fear & Greed" not in joined
