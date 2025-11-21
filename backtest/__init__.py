"""Backtesting toolkit for the Spot-AI Agent.

This package exposes the legacy :class:`Backtester` for compatibility while
adding a research-oriented engine, metrics utilities and scenario helpers used
by the Streamlit dashboard.
"""
from __future__ import annotations

from .legacy import (
    Backtester,
    grid_search,
    compute_buy_and_hold_pnl,
    generate_trades_from_ohlcv,
)
from .engine import ResearchBacktester, BacktestConfig, BacktestResult
from .metrics import (
    BacktestMetrics,
    aggregate_symbol_metrics,
    equity_curve_from_trades,
    equity_statistics,
    trade_distribution_metrics,
)
from .scenario import run_fee_slippage_scenarios, run_parameter_scenarios

__all__ = [
    "Backtester",
    "grid_search",
    "compute_buy_and_hold_pnl",
    "generate_trades_from_ohlcv",
    "ResearchBacktester",
    "BacktestConfig",
    "BacktestResult",
    "BacktestMetrics",
    "aggregate_symbol_metrics",
    "equity_curve_from_trades",
    "equity_statistics",
    "trade_distribution_metrics",
    "run_fee_slippage_scenarios",
    "run_parameter_scenarios",
]
