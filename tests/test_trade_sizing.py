from agent import (
    calculate_dynamic_trade_size,
    MIN_TRADE_USD,
    MAX_TRADE_USD,
)


def test_trade_size_low_confidence():
    assert calculate_dynamic_trade_size(5.0, 0.2, 3.0) == MIN_TRADE_USD


def test_trade_size_mid_confidence():
    expected = (MIN_TRADE_USD + MAX_TRADE_USD) / 2.0
    assert calculate_dynamic_trade_size(6.5, 0.5, 5.0) == expected


def test_trade_size_high_confidence():
    assert calculate_dynamic_trade_size(9.0, 0.9, 9.5) == MAX_TRADE_USD
