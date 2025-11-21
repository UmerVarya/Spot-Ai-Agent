from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from backtest.legacy import Backtester
from log_utils import setup_logger
from trade_constants import TP1_TRAILING_ONLY_STRATEGY
from trade_utils import evaluate_signal as live_evaluate_signal

from .metrics import (
    BacktestMetrics,
    equity_curve_from_trades,
    equity_statistics,
    trade_distribution_metrics,
)

logger = setup_logger(__name__)


@dataclass
class BacktestConfig:
    start_ts: Optional[pd.Timestamp] = None
    end_ts: Optional[pd.Timestamp] = None
    min_score: float = float(os.getenv("BACKTEST_DEFAULT_SCORE_THRESHOLD", 0.2))
    min_prob: float = 0.55
    atr_mult_sl: float = 1.5
    tp_rungs: Iterable[float] = field(default_factory=lambda: (1.0, 2.0, 3.0, 4.0))
    fee_bps: float = float(os.getenv("BACKTEST_DEFAULT_FEE_BPS", 10.0))
    slippage_bps: float = float(os.getenv("BACKTEST_DEFAULT_SLIPPAGE_BPS", 2.0))
    latency_bars: int = 0
    max_concurrent: int = 1
    initial_capital: float = float(os.getenv("BACKTEST_INITIAL_CAPITAL", 10_000))
    take_profit_strategy: str = os.getenv("TAKE_PROFIT_STRATEGY", "atr_trailing")
    skip_fraction: float = 0.0
    entry_delay_bars: int = 0


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: BacktestMetrics
    raw: Dict[str, Any]


class ResearchBacktester:
    """Thin orchestrator that wraps the legacy :class:`Backtester`.

    The research engine preserves live logic by delegating entry/exit handling to
    the legacy backtester while enriching the resulting trade log with
    additional analytics-friendly columns and computing professional metrics.
    """

    def __init__(
        self,
        historical_data: Dict[str, pd.DataFrame],
        evaluate_signal: Callable[[pd.DataFrame, str], Any] = live_evaluate_signal,
        predict_prob: Optional[Callable[..., float]] = None,
        macro_filter: Optional[Callable[[], bool]] = None,
        position_size_func: Optional[Callable[[float], float]] = None,
        name: str = "default",
    ) -> None:
        self.historical_data = historical_data
        self.evaluate_signal = evaluate_signal
        self.predict_prob = predict_prob or (lambda *args, **kwargs: 0.6)
        self.macro_filter = macro_filter or (lambda: True)
        self.position_size_func = position_size_func or (lambda confidence: 0.01)
        self.name = name

    def _build_params(self, cfg: BacktestConfig) -> Dict[str, Any]:
        return {
            "min_score": cfg.min_score,
            "min_prob": cfg.min_prob,
            "atr_mult_sl": cfg.atr_mult_sl,
            "tp_rungs": tuple(cfg.tp_rungs),
            "fee_bps": cfg.fee_bps,
            "slippage_bps": cfg.slippage_bps,
            "latency_bars": cfg.latency_bars + cfg.entry_delay_bars,
            "max_concurrent": cfg.max_concurrent,
            "start_ts": cfg.start_ts,
            "end_ts": cfg.end_ts,
        }

    @staticmethod
    def _compute_size(entry_price: float, multiplier: float, capital: float) -> tuple[float, float]:
        quote = capital * multiplier
        base = quote / entry_price if entry_price else 0.0
        return base, quote

    def _enrich_trades(self, trades_df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
        df = trades_df.copy()
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df["side"] = df.get("direction", "long")
        df["take_profit_strategy"] = cfg.take_profit_strategy

        base_sizes: List[float] = []
        quote_sizes: List[float] = []
        fees_paid: List[float] = []
        mae_list: List[float] = []
        mfe_list: List[float] = []

        for _, row in df.iterrows():
            entry_price = float(row.get("entry_price", 0.0))
            pos_mult = float(row.get("position_multiplier", 0.0))
            base, quote = self._compute_size(entry_price, pos_mult, cfg.initial_capital)
            base_sizes.append(base)
            quote_sizes.append(quote)
            fee = 2 * (cfg.fee_bps / 10_000.0) * quote
            fees_paid.append(fee)

            prices = []
            micro = row.get("microstructure")
            if isinstance(micro, dict):
                prices = list(micro.get("prices") or [])
            elif isinstance(micro, (list, tuple)):
                prices = list(micro)
            if prices:
                entry = entry_price
                returns = [(p - entry) / entry for p in prices if entry]
                mae_list.append(float(np.min(returns)))
                mfe_list.append(float(np.max(returns)))
            else:
                mae_list.append(np.nan)
                mfe_list.append(np.nan)

        df["position_size_base"] = base_sizes
        df["position_size_quote"] = quote_sizes
        df["fees_paid"] = fees_paid
        df["mae"] = mae_list
        df["mfe"] = mfe_list

        gross_returns = df["return"].astype(float)
        df["net_return"] = gross_returns - (cfg.fee_bps / 10_000.0 * 2)
        df["pnl"] = df["net_return"] * cfg.initial_capital
        df["r_multiple"] = df["net_return"] / (cfg.atr_mult_sl / max(cfg.tp_rungs)) if cfg.tp_rungs else np.nan
        df["outcome_type"] = df.get("reason", df.get("exit_reason", "unknown"))
        return df

    def run(self, cfg: BacktestConfig) -> BacktestResult:
        params = self._build_params(cfg)
        legacy_bt = Backtester(
            historical_data=self.historical_data,
            evaluate_signal=self.evaluate_signal,
            predict_prob=self.predict_prob,
            macro_filter=self.macro_filter,
            position_size_func=self.position_size_func,
        )
        if cfg.skip_fraction > 0:
            import random

            def gated_macro() -> bool:
                return self.macro_filter() and random.random() > cfg.skip_fraction

            legacy_bt.macro_filter = gated_macro

        raw_result = legacy_bt.run(params)
        trades_df = raw_result.get("trades_df") or pd.DataFrame(raw_result.get("trades", []))
        if trades_df.empty:
            empty_equity = pd.DataFrame(
                {"equity": [cfg.initial_capital]}, index=[cfg.start_ts or pd.Timestamp.utcnow()]
            )
            metrics = BacktestMetrics.empty()
            return BacktestResult(cfg, trades_df, empty_equity, metrics, raw_result)

        enriched = self._enrich_trades(trades_df, cfg)
        equity_curve = equity_curve_from_trades(enriched, cfg.initial_capital)
        stats = equity_statistics(equity_curve["equity"], risk_free_rate=0.0)
        trade_stats = trade_distribution_metrics(enriched)
        metrics = BacktestMetrics(
            sharpe=stats["sharpe"],
            sortino=stats["sortino"],
            calmar=stats["calmar"],
            max_drawdown=stats["max_drawdown"],
            total_return=stats["total_return"],
            win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            expectancy=trade_stats["expectancy"],
            avg_holding_time=trade_stats["avg_holding_time"],
            num_trades=int(len(enriched)),
        )
        return BacktestResult(cfg, enriched, equity_curve, metrics, raw_result)


def evaluate_and_score(df_slice: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Convenience wrapper to invoke the live evaluate_signal with defaults."""

    score, direction, confidence, meta = live_evaluate_signal(df_slice.copy(), symbol=symbol)
    metadata = meta if isinstance(meta, dict) else {"detail": meta}
    metadata.setdefault("take_profit_strategy", TP1_TRAILING_ONLY_STRATEGY)
    return {
        "score": float(score or 0.0),
        "direction": direction,
        "confidence": float(confidence or 0.0),
        "metadata": metadata,
    }
