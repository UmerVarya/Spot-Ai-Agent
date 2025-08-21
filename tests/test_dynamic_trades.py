from agent import dynamic_max_active_trades, MAX_ACTIVE_TRADES

def test_reduces_trades_in_fear():
    assert dynamic_max_active_trades(10, "neutral", 0.5) == 1

def test_increases_trades_when_bullish_low_vol():
    assert dynamic_max_active_trades(80, "bullish", 0.2) == MAX_ACTIVE_TRADES + 1

def test_high_volatility_cuts_allocation():
    assert dynamic_max_active_trades(80, "bullish", 0.9) == MAX_ACTIVE_TRADES

