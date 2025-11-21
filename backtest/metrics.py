from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    total_return: float
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_holding_time: float
    num_trades: int

    @classmethod
    def empty(cls) -> "BacktestMetrics":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)


def _safe_std(values: np.ndarray) -> float:
    return float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0


def _drawdown_curve(equity: Iterable[float]) -> np.ndarray:
    series = np.asarray(list(equity), dtype=float)
    if series.size == 0:
        return np.array([])
    cummax = np.maximum.accumulate(series)
    drawdown = (series - cummax) / cummax
    return drawdown


def equity_statistics(equity: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, float]:
    returns = equity.pct_change().dropna()
    if returns.empty:
        return {k: 0.0 for k in ["sharpe", "sortino", "calmar", "max_drawdown", "total_return"]}

    rf = risk_free_rate / 252.0
    excess = returns - rf
    downside = excess[excess < 0]

    sharpe = float(np.sqrt(252) * excess.mean() / (excess.std(ddof=1) or 1e-9))
    sortino = float(np.sqrt(252) * excess.mean() / (downside.std(ddof=1) or 1e-9))

    drawdown = _drawdown_curve(equity)
    max_dd = float(np.nanmin(drawdown)) if drawdown.size else 0.0
    calmar = float((equity.iloc[-1] / equity.iloc[0] - 1) / abs(max_dd or 1e-9))

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
    }


def trade_distribution_metrics(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {k: 0.0 for k in ["win_rate", "profit_factor", "expectancy", "avg_holding_time"]}
    returns = trades["net_return"].astype(float)
    winners = returns[returns > 0]
    losers = returns[returns <= 0]
    win_rate = float(len(winners) / len(returns)) if len(returns) else 0.0
    gross_profit = winners.sum()
    gross_loss = abs(losers.sum()) if len(losers) else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss else float("inf")
    expectancy = float(returns.mean())
    holding = trades.get("holding_bars")
    avg_holding = float(holding.mean()) if holding is not None else 0.0
    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_holding_time": avg_holding,
    }


def equity_curve_from_trades(trades: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame({"equity": [initial_capital]})
    returns = trades["net_return"].astype(float).fillna(0.0)
    equity = initial_capital * (1 + returns).cumprod()
    timestamps = pd.to_datetime(trades["exit_time"])
    return pd.DataFrame({"timestamp": timestamps, "equity": equity}).set_index("timestamp")


def aggregate_symbol_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    grouped = trades.groupby("symbol")
    rows = []
    for symbol, group in grouped:
        metrics = trade_distribution_metrics(group)
        pnl = group["pnl"].sum()
        rows.append({
            "symbol": symbol,
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "expectancy": metrics["expectancy"],
            "num_trades": len(group),
            "pnl": pnl,
        })
    return pd.DataFrame(rows).sort_values("pnl", ascending=False)
