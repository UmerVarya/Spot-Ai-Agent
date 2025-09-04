from agent import calculate_dynamic_trade_size


def test_trade_size_bounds():
    assert calculate_dynamic_trade_size(0, 0.0, 0) == 100.0
    assert calculate_dynamic_trade_size(10, 1.0, 10) == 200.0
