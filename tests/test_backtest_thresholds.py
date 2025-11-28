from backtest.engine import BacktestConfig
from backtest.legacy import passes_min_score_gate
from config import DEFAULT_MIN_PROB_FOR_TRADE


def test_backtest_min_prob_defaults_to_shared_constant():
    cfg = BacktestConfig()
    assert cfg.min_prob == DEFAULT_MIN_PROB_FOR_TRADE


def test_backtest_min_score_defaults_to_none():
    cfg = BacktestConfig()
    assert cfg.min_score is None


def test_passes_min_score_gate_respects_optional_threshold():
    assert passes_min_score_gate(5.0, 5.0, None)
    assert not passes_min_score_gate(5.0, 5.0, 6.0)

