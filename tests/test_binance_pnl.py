import pandas as pd
from backtest import compute_buy_and_hold_pnl
from pytest import approx


def test_compute_buy_and_hold_pnl():
    df = pd.DataFrame({'close': [100, 110, 105]})
    out = compute_buy_and_hold_pnl(df)
    assert 'pnl' in out.columns
    assert 'equity' in out.columns
    # First return should be 0, second 0.1, third about -0.04545
    expected_pnl = [0.0, 0.1, -0.045454545454545456]
    assert out['pnl'].tolist() == approx(expected_pnl)
    expected_equity = [1.0, 1.1, 1.05]
    assert out['equity'].tolist() == approx(expected_equity)
