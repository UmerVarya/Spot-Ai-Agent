import dataclasses
import json

import pandas as pd
import pytest

from backtest.engine import BacktestConfig, ResearchBacktester, run_backtest_from_csv_paths
from backtest.metrics import equity_statistics, trade_distribution_metrics
from backtest.scenario import run_fee_slippage_scenarios
from backtest.filesystem import (
    BacktestRunMetadata,
    build_backtest_id,
    build_backtest_output_paths,
    write_csv_atomic,
    write_json_atomic,
)
from trade_schema import TRADE_HISTORY_COLUMNS


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


def test_research_backtester_preserves_legacy_net_returns():
    bt = ResearchBacktester({}, evaluate_signal=_dummy_signal)
    cfg = BacktestConfig(fee_bps=25, initial_capital=1_000.0)
    trades = pd.DataFrame(
        {
            "entry_time": [pd.Timestamp("2024-01-01", tz="UTC")],
            "exit_time": [pd.Timestamp("2024-01-02", tz="UTC")],
            "return": [0.05],
            "entry_price": [100.0],
            "position_multiplier": [0.5],
        }
    )

    enriched = bt._enrich_trades(trades, cfg)

    assert enriched["net_return"].iat[0] == pytest.approx(trades["return"].iat[0])


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


def test_progress_callback_reports_phases(tmp_path):
    data = _build_trending_data()["TEST"]
    df_with_ts = data.reset_index().rename(columns={"index": "timestamp"})
    csv_path = tmp_path / "TEST_1m.csv"
    df_with_ts.to_csv(csv_path, index=False)

    phases: list[str] = []

    def _progress_cb(progress):
        phases.append(progress.phase)

    cfg = BacktestConfig(min_score=0.0, min_prob=0.0, is_backtest=True)
    run_backtest_from_csv_paths([csv_path], cfg, progress_callback=_progress_cb)

    for phase in ["loading", "simulating", "finalizing", "done"]:
        assert phase in phases


def test_fixed_notional_backtest_outputs(tmp_path):
    data = _build_trending_data()
    df_with_ts = data["TEST"].reset_index().rename(columns={"index": "timestamp"})
    csv_path = tmp_path / "TEST_1m.csv"
    df_with_ts.to_csv(csv_path, index=False)

    start_ts = data["TEST"].index.min()
    end_ts = data["TEST"].index.max()
    base_cfg = BacktestConfig(
        start_ts=start_ts,
        end_ts=end_ts,
        min_score=0.0,
        min_prob=0.0,
        is_backtest=True,
        sizing_mode="fixed_notional",
        trade_size_usd=500.0,
        fee_bps=10.0,
        slippage_bps=0.0,
        initial_capital=10_000.0,
    )

    timeframe = "1m"
    for exit_mode in ["tp_trailing", "atr_trailing"]:
        cfg = dataclasses.replace(base_cfg, exit_mode=exit_mode)
        result = run_backtest_from_csv_paths([csv_path], cfg, symbols=["TEST"])
        backtest_id = (
            build_backtest_id(
                ["TEST"],
                timeframe,
                start_ts.strftime("%Y-%m-%d"),
                end_ts.strftime("%Y-%m-%d"),
            )
            + f"_{exit_mode}"
        )
        paths = build_backtest_output_paths(backtest_id, tmp_path)

        trades = result.trades if not result.trades.empty else pd.DataFrame(columns=list(TRADE_HISTORY_COLUMNS))
        equity_curve = result.equity_curve
        if equity_curve.empty:
            equity_curve = pd.DataFrame(
                {
                    "timestamp": [cfg.start_ts],
                    "equity": [cfg.initial_capital],
                    "peak_equity": [cfg.initial_capital],
                    "drawdown_pct": [0.0],
                    "drawdown": [0.0],
                }
            )

        write_csv_atomic(paths["trades"], trades)
        write_csv_atomic(paths["equity"], equity_curve)
        write_json_atomic(paths["metrics"], result.metrics or {})

        params = {
            "score_threshold": cfg.min_score,
            "min_prob": cfg.min_prob,
            "trade_size_usd": cfg.trade_size_usd,
            "exit_mode": cfg.exit_mode,
            "symbols": ["TEST"],
        }
        meta = BacktestRunMetadata(
            backtest_id=backtest_id,
            symbols=["TEST"],
            timeframe=timeframe,
            start_date=start_ts.strftime("%Y-%m-%d"),
            end_date=end_ts.strftime("%Y-%m-%d"),
            params=params,
            status="completed",
            progress=1.0,
        )
        write_json_atomic(paths["meta"], meta.to_dict())

        assert paths["trades"].exists()
        assert paths["equity"].exists()
        meta_payload = json.loads(paths["meta"].read_text())
        assert meta_payload.get("status") == "completed"
        assert meta_payload["params"].get("trade_size_usd") == cfg.trade_size_usd
        assert meta_payload["params"].get("exit_mode") == exit_mode
        assert meta_payload["params"].get("score_threshold") == cfg.min_score
        assert meta_payload["params"].get("min_prob") == cfg.min_prob
