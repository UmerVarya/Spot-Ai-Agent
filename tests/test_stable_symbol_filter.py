import pytest

from trade_utils import is_stable_symbol, filter_stable_symbols


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("USDEUSDT", True),
        ("USDJUSDT", True),
        ("FDUSDUSDT", True),
        ("USDCUSDT", True),
        ("BTCUSDT", False),
    ],
)
def test_is_stable_symbol(symbol, expected):
    assert is_stable_symbol(symbol) is expected


def test_filter_stable_symbols_removes_stablecoins():
    raw = ["BTCUSDT", "USDEUSDT", "USDJUSDT"]
    filtered = filter_stable_symbols(raw)
    assert filtered == ["BTCUSDT"]
