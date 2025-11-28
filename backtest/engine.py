from __future__ import annotations

import os
from pathlib import Path
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from backtest.legacy import Backtester
from config import DEFAULT_MIN_PROB_FOR_TRADE
from backtest.data import load_csv_paths
from log_utils import setup_logger
from trade_constants import TP1_TRAILING_ONLY_STRATEGY
from trade_utils import evaluate_signal as live_evaluate_signal
from trade_utils import precompute_backtest_indicators

from .types import BacktestProgress, ProgressCallback, emit_progress
from .presets import BacktestPresetConfig, resolve_preset

from .analysis import (
    build_equity_curve,
    per_symbol_breakdown,
    score_bucket_metrics,
    session_metrics,
    summarise_backtest_metrics,
)

logger = setup_logger(__name__)


def _count_total_bars(historical_data: Dict[str, pd.DataFrame]) -> int:
    if not historical_data:
        return 0
    return int(sum(len(df) for df in historical_data.values()))


@dataclass
class BacktestConfig:
    start_ts: Optional[pd.Timestamp] = None
    end_ts: Optional[pd.Timestamp] = None
    is_backtest: bool = True
    # Optional extra global confidence/score gate.
    # When None, only per-symbol/tier thresholds enforced by evaluate_signal apply.
    min_score: Optional[float] = None
    min_prob: float = DEFAULT_MIN_PROB_FOR_TRADE
    atr_mult_sl: float = 1.5
    tp_rungs: Iterable[float] = field(default_factory=lambda: (1.0, 2.0, 3.0, 4.0))
    fee_bps: float = float(os.getenv("BACKTEST_DEFAULT_FEE_BPS", 10.0))
    slippage_bps: float = float(os.getenv("BACKTEST_DEFAULT_SLIPPAGE_BPS", 2.0))
    latency_bars: int = 0
    max_concurrent: int = 1
    initial_capital: float = float(os.getenv("BACKTEST_INITIAL_CAPITAL", 10_000))
    risk_per_trade_pct: float = 1.0
    take_profit_strategy: str = os.getenv("TAKE_PROFIT_STRATEGY", "atr_trailing")
    skip_fraction: float = 0.0
    entry_delay_bars: int = 0
    sizing_mode: str = "risk_pct"
    trade_size_usd: float = 500.0
    exit_mode: str = "tp_trailing"
    random_seed: int | None = None


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: Dict[str, float]
    by_symbol: pd.DataFrame
    by_buckets: Dict[str, pd.DataFrame]
    scenarios: Optional[pd.DataFrame] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class ResearchBacktester:
    """Thin orchestrator that wraps the legacy :class:`Backtester`.

    The research engine preserves live logic by delegating entry/exit handling to
    the legacy backtester while enriching the resulting trade log with
    additional analytics-friendly columns and computing professional metrics.
    """

    def __init__(
        self,
        historical_data: Dict[str, pd.DataFrame],
        evaluate_signal: Callable[[pd.DataFrame, str, bool], Any] = live_evaluate_signal,
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
        for df in self.historical_data.values():
            precompute_backtest_indicators(df)

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
            "is_backtest": cfg.is_backtest,
            "sizing_mode": cfg.sizing_mode,
            "trade_size_usd": cfg.trade_size_usd,
            "risk_per_trade_pct": cfg.risk_per_trade_pct,
            "exit_mode": cfg.exit_mode,
        }

    @staticmethod
    def _compute_size(entry_price: float, multiplier: float, capital: float) -> tuple[float, float]:
        quote = capital * multiplier
        base = quote / entry_price if entry_price else 0.0
        return base, quote

    @staticmethod
    def _session_from_timestamp(ts: pd.Timestamp) -> str:
        hour = ts.tz_convert("UTC").hour if ts.tzinfo else ts.hour
        if 0 <= hour < 7 or hour >= 20:
            return "Asia"
        if 7 <= hour < 13:
            return "Europe"
        return "US"

    @staticmethod
    def _volatility_percentiles(df: pd.DataFrame, period: int = 14) -> pd.Series:
        required = {"high", "low", "close"}
        if df.empty or not required.issubset(df.columns):
            return pd.Series(dtype=float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(period).mean()
        return atr.rank(pct=True, method="average")

    @staticmethod
    def _label_vol_regime(percentile: float | None) -> str:
        if percentile is None or not np.isfinite(percentile):
            return "unknown"
        if percentile >= 0.66:
            return "high"
        if percentile <= 0.33:
            return "low"
        return "medium"

    def _enrich_trades(self, trades_df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
        df = trades_df.copy()
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df["side"] = df.get("direction", "long")
        df["take_profit_strategy"] = cfg.take_profit_strategy
        df["score_at_entry"] = df.get("score", df.get("signal", np.nan))

        vol_percentiles: dict[str, pd.Series] = {}
        for symbol, history in self.historical_data.items():
            vol_percentiles[symbol] = self._volatility_percentiles(history)

        base_sizes: List[float] = []
        quote_sizes: List[float] = []
        fees_paid: List[float] = []
        mae_list: List[float] = []
        mfe_list: List[float] = []
        sessions: List[str] = []
        vol_regimes: List[str] = []

        for _, row in df.iterrows():
            entry_price = float(row.get("entry_price", 0.0))
            pos_mult = float(row.get("position_multiplier", 0.0))
            if cfg.sizing_mode == "fixed_notional":
                quote = float(cfg.trade_size_usd)
                base = quote / entry_price if entry_price else 0.0
            else:
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

            entry_ts = pd.to_datetime(row.get("entry_time"))
            sessions.append(self._session_from_timestamp(entry_ts))
            symbol = row.get("symbol")
            vol_pct = None
            if isinstance(symbol, str) and symbol in vol_percentiles:
                vol_series = vol_percentiles[symbol]
                vol_pct = float(vol_series.get(entry_ts, np.nan)) if not vol_series.empty else np.nan
            vol_regimes.append(self._label_vol_regime(vol_pct))

        df["position_size_base"] = base_sizes
        df["position_size_quote"] = quote_sizes
        df["fees_paid"] = fees_paid
        df["mae"] = mae_list
        df["mfe"] = mfe_list
        df["session"] = sessions
        df["vol_regime"] = vol_regimes

        gross_returns = df["return"].astype(float)
        # The legacy backtester already deducts entry/exit fees from `return`, so
        # treat it as net to avoid double-charging costs here.
        df["net_return"] = gross_returns
        df["pnl"] = df["net_return"] * cfg.initial_capital
        df["r_multiple"] = df["net_return"] / (cfg.atr_mult_sl / max(cfg.tp_rungs)) if cfg.tp_rungs else np.nan
        df["outcome_type"] = df.get("reason", df.get("exit_reason", "unknown"))
        df["net_pnl_quote"] = df.get("net_pnl_quote", df.get("pnl", df["pnl"]))
        df["gross_pnl_quote"] = df.get("gross_pnl_quote", df.get("pnl", df["pnl"]))
        df["fees_quote"] = df.get("fees_quote", df.get("fees", fees_paid))
        df["risk_amount_quote"] = df.get("risk_amount_quote", np.nan)
        df["holding_time_minutes"] = (
            (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60.0
        )
        df["exit_mode"] = cfg.exit_mode
        df["sizing_mode"] = cfg.sizing_mode
        df["trade_size_usd"] = cfg.trade_size_usd
        return df

    def run(
        self,
        cfg: BacktestConfig,
        progress_callback: Optional[ProgressCallback] = None,
        preset: BacktestPresetConfig | str | None = None,
    ) -> BacktestResult:
        preset_cfg = resolve_preset(preset)
        if cfg.random_seed is not None:
            try:
                import random

                random.seed(cfg.random_seed)
            except Exception:
                pass
            try:
                np.random.seed(cfg.random_seed)
            except Exception:
                pass
        params = self._build_params(cfg)
        position_size_func = self.position_size_func
        if cfg.sizing_mode == "fixed_notional":
            trade_fraction = (cfg.trade_size_usd or 0.0) / max(cfg.initial_capital, 1e-9)

            def _fixed_notional_sizer(_: float) -> float:
                return max(trade_fraction, 0.0)

            position_size_func = _fixed_notional_sizer
        else:
            risk_pct = max(cfg.risk_per_trade_pct, 0.0)

            def _risk_pct_sizer(_: float) -> float:
                return risk_pct / 100.0

            position_size_func = _risk_pct_sizer

        legacy_bt = Backtester(
            historical_data=self.historical_data,
            evaluate_signal=self.evaluate_signal,
            predict_prob=self.predict_prob,
            macro_filter=self.macro_filter,
            position_size_func=position_size_func,
        )
        total_bars = _count_total_bars(self.historical_data)
        emit_progress(
            progress_callback,
            BacktestProgress(
                phase="simulating",
                current=0,
                total=total_bars,
                message=f"Running simulation for {len(self.historical_data)} symbols",
            ),
        )
        if cfg.skip_fraction > 0:
            import random

            def gated_macro() -> bool:
                return self.macro_filter() and random.random() > cfg.skip_fraction

            legacy_bt.macro_filter = gated_macro

        raw_result = legacy_bt.run(
            params,
            progress_callback=progress_callback,
            preset=preset_cfg,
        )
        trades_df = raw_result.get("trades_df") or pd.DataFrame(raw_result.get("trades", []))
        emit_progress(
            progress_callback,
            BacktestProgress(
                phase="finalizing",
                current=0,
                total=1,
                message="Computing equity curve and metrics",
            ),
        )
        if trades_df.empty:
            empty_equity = build_equity_curve(trades_df, cfg.initial_capital)
            metrics = summarise_backtest_metrics(empty_equity, trades_df, cfg.initial_capital)
            emit_progress(
                progress_callback,
                BacktestProgress(
                    phase="done",
                    current=total_bars,
                    total=total_bars,
                    message="Backtest complete.",
                ),
            )
            return BacktestResult(cfg, trades_df, empty_equity, metrics, pd.DataFrame(), {}, None, raw_result)

        enriched = self._enrich_trades(trades_df, cfg)
        equity_curve = build_equity_curve(enriched, cfg.initial_capital)
        metrics = summarise_backtest_metrics(equity_curve, enriched, cfg.initial_capital)
        by_symbol = per_symbol_breakdown(enriched)
        by_buckets = {
            "score_bucket": score_bucket_metrics(enriched),
            "session": session_metrics(enriched),
        }
        emit_progress(
            progress_callback,
            BacktestProgress(
                phase="done",
                current=total_bars,
                total=total_bars,
                message="Backtest complete.",
            ),
        )
        return BacktestResult(cfg, enriched, equity_curve, metrics, by_symbol, by_buckets, None, raw_result)


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


def run_backtest_from_csv_paths(
    csv_paths: list[Path],
    cfg: BacktestConfig,
    symbols: Optional[Iterable[str]] = None,
    scenario_runner: Optional[Callable[[ResearchBacktester, BacktestConfig], pd.DataFrame]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    preset: BacktestPresetConfig | str | None = None,
) -> BacktestResult:
    """Run a backtest using explicitly provided CSV files."""

    data = load_csv_paths(
        csv_paths,
        start=cfg.start_ts,
        end=cfg.end_ts,
    )
    if symbols:
        symbols_upper = {sym.upper() for sym in symbols}
        data = {k: v for k, v in data.items() if k.upper() in symbols_upper}

    total_bars = _count_total_bars(data)
    emit_progress(
        progress_callback,
        BacktestProgress(
            phase="loading",
            current=0,
            total=total_bars,
            message=f"Loading OHLCV for {', '.join(sorted(data.keys())) if data else 'no symbols'}",
        ),
    )

    bt = ResearchBacktester(data)
    result = bt.run(cfg, progress_callback=progress_callback, preset=preset)
    if scenario_runner:
        result.scenarios = scenario_runner(bt, cfg)
    return result
