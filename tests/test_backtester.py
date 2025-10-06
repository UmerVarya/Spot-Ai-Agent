import pandas as pd
import pytest

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


def test_backtester_microstructure_stop_and_delta():
    dates = pd.date_range("2023-01-01", periods=80, freq="T")
    base_price = 100 + 0.05 * pd.Series(range(80), index=dates)
    data = pd.DataFrame(
        {
            'open': base_price,
            'high': base_price + 1.0,
            'low': base_price - 1.0,
            'close': base_price,
            'volume': 100.0,
            'taker_buy_base': 60.0,
        },
        index=dates,
    )
    shock_idx = dates[51]
    shock_open = float(base_price.loc[shock_idx])
    data.loc[shock_idx, ['open', 'high', 'low', 'close']] = [
        shock_open,
        shock_open + 1.5,
        shock_open - 5.0,
        shock_open + 0.25,
    ]
    start_time = dates[50]

    def selective_eval(df, symbol):
        if df.index[-1] >= start_time:
            return 7.0, 'long', 7.0, None
        return 0.0, None, 0.0, None

    bt = Backtester({'BTCUSDT': data}, selective_eval, dummy_predict, dummy_macro, dummy_pos)
    params = {'stop_multiplier': 0.5, 'tp_multipliers': [0.5]}
    result = bt.run(params)
    assert result['trade_log'], "Expected at least one trade"
    trade = result['trade_log'][0]
    assert trade['exit_reason'] == 'stop'
    micro = trade['microstructure']
    assert micro['prices'][0] == pytest.approx(float(data.iloc[51]['open']))
    assert micro['imbalance'] == pytest.approx(0.2)
