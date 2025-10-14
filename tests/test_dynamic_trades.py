from agent import dynamic_max_active_trades, MAX_ACTIVE_TRADES

def test_trade_cap_always_one():
    assert MAX_ACTIVE_TRADES == 1
    assert dynamic_max_active_trades(10, "neutral", 0.5) == 1
    assert dynamic_max_active_trades(80, "bullish", 0.2) == 1
    assert dynamic_max_active_trades(80, "bullish", 0.9) == 1

