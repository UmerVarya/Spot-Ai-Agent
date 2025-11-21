import pandas as pd

from backtest.engine import BacktestConfig, ResearchBacktester
from backtest.metrics import equity_statistics, trade_distribution_metrics
from backtest.scenario import run_fee_slippage_scenarios


def _dummy_signal(df_slice: pd.DataFrame, symbol: str):
    # Emit a long signal with high confidence on every bar
    return {"score": 1.0, "direction": 1, "confidence": 1.0}


def _build_trending_data() -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=40, freq="T", tz="UTC")
    prices = pd.Series(range(1, 41), index=idx, dtype=float)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices + 0.25,
            "volume": 1000.0,
        }
    )
    return {"TEST": df}


def test_research_backtester_generates_trades():
    data = _build_trending_data()
    bt = ResearchBacktester(data, evaluate_signal=_dummy_signal)
    cfg = BacktestConfig(min_score=0.2, min_prob=0.0, initial_capital=1_000.0)
    result = bt.run(cfg)
    assert not result.trades.empty
    assert result.metrics.num_trades == len(result.trades)


def test_metrics_helpers():
    trades = pd.DataFrame(
        {
            "net_return": [0.05, -0.02, 0.01],
            "exit_time": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "symbol": ["A", "A", "B"],
            "pnl": [50, -20, 10],
            "holding_bars": [1, 2, 3],
        }
    )
    eq = (1 + trades["net_return"]).cumprod()
    stats = equity_statistics(eq)
    dist = trade_distribution_metrics(trades)
    assert stats["max_drawdown"] <= 0
    assert 0 <= dist["win_rate"] <= 1


def test_scenario_runner_changes_params():
    data = _build_trending_data()
    bt = ResearchBacktester(data, evaluate_signal=_dummy_signal)
    base_cfg = BacktestConfig(min_score=0.2, min_prob=0.0)
    scenarios = run_fee_slippage_scenarios(bt, base_cfg, [5, 10], [1, 2])
    assert set(scenarios.columns).issuperset({"fee_bps", "slippage_bps", "sharpe"})
    assert len(scenarios) == 4
