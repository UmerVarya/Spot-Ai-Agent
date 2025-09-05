from agent import (
    calculate_dynamic_trade_size,
    MIN_TRADE_USD,
    MAX_TRADE_USD,
)


def test_trade_size_bounds():
    assert calculate_dynamic_trade_size(0, 0.0, 0) == MIN_TRADE_USD
    assert calculate_dynamic_trade_size(10, 1.0, 10) == MAX_TRADE_USD
