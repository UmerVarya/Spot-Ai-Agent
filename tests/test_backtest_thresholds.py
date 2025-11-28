from backtest.engine import BacktestConfig
from config import DEFAULT_MIN_PROB_FOR_TRADE


def test_backtest_min_prob_defaults_to_shared_constant():
    cfg = BacktestConfig()
    assert cfg.min_prob == DEFAULT_MIN_PROB_FOR_TRADE

