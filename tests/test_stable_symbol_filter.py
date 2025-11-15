import pytest

from trade_utils import is_stable_symbol, filter_stable_symbols


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("USDCUSDT", True),
        ("FDUSDUSDT", True),
        ("BUSDUSDT", True),
        ("TUSDUSDT", True),
        ("BTCUSDT", False),
        ("ETHUSDT", False),
        ("SOLUSDT", False),
        ("AVAXUSDT", False),
    ],
)
def test_is_stable_symbol(symbol, expected):
    assert is_stable_symbol(symbol) is expected


def test_filter_stable_symbols_removes_stablecoins():
    raw = ["BTCUSDT", "ETHUSDT", "USDCUSDT", "FDUSDUSDT", "SOLUSDT"]
    filtered, dropped = filter_stable_symbols(raw)
    assert filtered == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert dropped == ["USDCUSDT", "FDUSDUSDT"]
