import pandas as pd
from backtest import Backtester


def dummy_eval(df, symbol):
    return 5.0, 'pattern', 7.0, None


def dummy_predict(score, conf, *args):
    return 0.6


def dummy_macro():
    return True


def dummy_pos(conf):
    return 1.0


def test_backtester_run():
    dates = pd.date_range("2023-01-01", periods=60, freq="H")
    prices = 1 + 0.01 * pd.Series(range(60), index=dates)
    data = pd.DataFrame({
        'open': prices,
        'high': prices + 0.01,
        'low': prices - 0.01,
        'close': prices
    }, index=dates)
    bt = Backtester({'BTCUSDT': data}, dummy_eval, dummy_predict, dummy_macro, dummy_pos)
    result = bt.run({})
    assert 'final_equity' in result
    assert isinstance(result['num_trades'], int)
