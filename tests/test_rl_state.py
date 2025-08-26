from trade_utils import get_rl_state


def test_get_rl_state(tmp_path):
    log = tmp_path / "trades.csv"
    # last trade: win
    log.write_text("2024-01-01 00:00:00,BTCUSDT,long,100,110,exit,0,0,0,bullish,0\n")
    state_high = get_rl_state(0.9, log_file=str(log))
    state_low = get_rl_state(0.1, log_file=str(log))
    assert state_high == "win_high_vol"
    assert state_low == "win_low_vol"


def test_get_rl_state_no_history(tmp_path):
    # No trade log -> neutral with volatility bucket
    state = get_rl_state(0.5, log_file=str(tmp_path / "missing.csv"))
    assert state == "neutral_mid_vol"
