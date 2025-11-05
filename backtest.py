"""
Historical backtesting framework for the Spot‑AI Agent.

This module allows you to simulate trading over historical OHLCV data
using the decision logic of the Spot‑AI Agent.  It is intentionally
agnostic about the specific strategy components – you must provide
callbacks for signal evaluation, macro gating and position sizing.

The implementation mirrors the live pipeline as closely as possible:
signals are evaluated on fully formed candles with no look‑ahead,
entries are filled at the next bar's open (optionally delayed by a
latency buffer) and trades are managed using ATR‑based trailing stops
and take‑profit ladders.  Fees, slippage and global concurrency limits
are also applied so the simulation reflects production constraints.

Usage::

    from backtest import Backtester, grid_search
    bt = Backtester(historical_data, evaluate_signal, predict_prob, macro_filter, position_size_func)
    results = bt.run(params)
    print(results["performance"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from risk_metrics import max_drawdown, sharpe_ratio
from log_utils import setup_logger

logger = setup_logger(__name__)


@dataclass(slots=True)
class PendingEntry:
    symbol: str
    signal_time: pd.Timestamp
    entry_index: int
    entry_time: pd.Timestamp
    direction: int
    score: float
    confidence: float
    probability: float
    position_multiplier: float
    atr_value: float
    atr_multiplier: float
    tp_rungs: Tuple[float, ...]
    fee_bps: float
    slippage_bps: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TradePosition:
    symbol: str
    entry_time: pd.Timestamp
    entry_index: int
    direction: int
    entry_price: float
    score: float
    confidence: float
    probability: float
    position_multiplier: float
    atr_multiplier: float
    stop_price: float
    take_profits: List[float]
    atr_value: float
    best_price: float
    metadata: Dict[str, Any]
    signal_time: pd.Timestamp
    fee_bps: float
    slippage_bps: float

    def direction_label(self) -> str:
        return "long" if self.direction >= 0 else "short"


class Backtester:
    """Backtesting harness that mirrors the live Spot‑AI execution flow."""

    def __init__(
        self,
        historical_data: Dict[str, pd.DataFrame],
        evaluate_signal: Callable[[pd.DataFrame, str], Any],
        predict_prob: Callable[..., float],
        macro_filter: Callable[[], bool],
        position_size_func: Callable[[float], float],
    ) -> None:
        self.historical_data = {
            symbol: df.copy()
            for symbol, df in (historical_data or {}).items()
        }
        for df in self.historical_data.values():
            if not df.index.is_monotonic_increasing:
                df.sort_index(inplace=True)
        self.evaluate_signal = evaluate_signal
        self.predict_prob = predict_prob
        self.macro_filter = macro_filter
        self.position_size_func = position_size_func

    @staticmethod
    def _direction_multiplier(direction: Any) -> int:
        """Normalise signal direction into ``+1`` (long) or ``-1`` (short)."""

        if direction is None:
            return 1
        if isinstance(direction, (int, float)):
            if direction > 0:
                return 1
            if direction < 0:
                return -1
        if isinstance(direction, str):
            lowered = direction.lower()
            if "short" in lowered or "sell" in lowered:
                return -1
        return 1

    @staticmethod
    def _simulate_intrabar_path(bar: pd.Series) -> Dict[str, Any]:
        """Create a plausible intrabar price path and flow imbalance."""

        def _to_float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")

        open_price = _to_float(bar.get("open"))
        high = _to_float(bar.get("high"))
        low = _to_float(bar.get("low"))
        close_price = _to_float(bar.get("close"))
        volume = _to_float(bar.get("volume", float("nan")))
        taker_buy = _to_float(bar.get("taker_buy_base", float("nan")))

        prices: List[float] = []

        def _append_price(value: float) -> None:
            if not math.isfinite(value):
                return
            if prices and math.isclose(prices[-1], value, rel_tol=1e-9, abs_tol=1e-9):
                return
            prices.append(value)

        _append_price(open_price)

        if math.isfinite(high) and math.isfinite(low):
            if math.isfinite(open_price) and math.isfinite(close_price):
                bullish_bias = close_price >= open_price
            else:
                bullish_bias = True
            wick_sequence = [low, high] if bullish_bias else [high, low]
            for price in wick_sequence:
                _append_price(price)

        _append_price(close_price)

        if not prices and math.isfinite(close_price):
            prices = [close_price]

        volume_delta = float("nan")
        imbalance = float("nan")
        taker_buy_ratio = float("nan")

        if math.isfinite(volume) and volume > 0:
            if math.isfinite(taker_buy):
                taker_buy = max(min(taker_buy, volume), 0.0)
                taker_sell = volume - taker_buy
                volume_delta = taker_buy - taker_sell
                imbalance = float(np.clip(volume_delta / volume, -1.0, 1.0))
                taker_buy_ratio = float(np.clip((taker_buy / volume) * 2 - 1.0, -1.0, 1.0))
            else:
                direction_bias = 0.0
                if math.isfinite(open_price) and math.isfinite(close_price):
                    if close_price > open_price:
                        direction_bias = 1.0
                    elif close_price < open_price:
                        direction_bias = -1.0
                volume_delta = direction_bias * volume
                imbalance = float(np.clip(direction_bias, -1.0, 1.0))

        return {
            "prices": prices,
            "volume": volume if math.isfinite(volume) else float("nan"),
            "taker_buy_volume": taker_buy if math.isfinite(taker_buy) else float("nan"),
            "volume_delta": volume_delta,
            "imbalance": imbalance,
            "taker_buy_ratio": taker_buy_ratio,
        }

    def _normalise_signal(self, signal: Any) -> Dict[str, Any]:
        """Return a normalised dictionary for a signal callback response."""

        if isinstance(signal, dict):
            score = float(signal.get("score", signal.get("confidence", 0.0)) or 0.0)
            confidence = float(signal.get("confidence", score) or score)
            direction = signal.get("direction")
            meta = {
                key: value
                for key, value in signal.items()
                if key not in {"score", "confidence", "direction"}
            }
            return {
                "score": score,
                "confidence": confidence,
                "direction": direction,
                "metadata": meta,
                "raw": signal,
            }

        if isinstance(signal, (list, tuple)) and signal:
            score = float(signal[0] or 0.0)
            direction = signal[1] if len(signal) > 1 else None
            confidence = float(signal[2] or score) if len(signal) > 2 else score
            meta = signal[3] if len(signal) > 3 else None
            return {
                "score": score,
                "confidence": confidence,
                "direction": direction,
                "metadata": meta if isinstance(meta, dict) else {"detail": meta},
                "raw": signal,
            }

        score = float(signal or 0.0)
        return {
            "score": score,
            "confidence": score,
            "direction": None,
            "metadata": {},
            "raw": signal,
        }

    @staticmethod
    def _compute_atr(window: pd.DataFrame, period: int = 14) -> float:
        if window is None or window.empty:
            return float("nan")
        if not {"high", "low", "close"}.issubset(window.columns):
            return float("nan")
        if len(window) < period + 1:
            return float("nan")
        high = window["high"].astype(float)
        low = window["low"].astype(float)
        close = window["close"].astype(float)
        prev_close = close.shift(1)
        tr_components = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        true_range = tr_components.max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        return float(atr)

    def _call_predict_prob(self, signal: Dict[str, Any], symbol: str) -> float:
        score = float(signal.get("score", 0.0))
        confidence = float(signal.get("confidence", score))
        try:
            return float(self.predict_prob(signal, symbol))
        except TypeError:
            try:
                metadata = signal.get("metadata", {}) or {}
                return float(
                    self.predict_prob(
                        score,
                        confidence,
                        symbol,
                        float(metadata.get("fear_greed", 0.0)),
                        float(metadata.get("btc_dom", 0.0)),
                        confidence,
                        str(metadata.get("pattern")),
                    )
                )
            except TypeError:
                return float(self.predict_prob(score, confidence))

    def _schedule_entry(
        self,
        pending: List[PendingEntry],
        symbol: str,
        df: pd.DataFrame,
        current_index: int,
        signal_time: pd.Timestamp,
        direction: int,
        score: float,
        confidence: float,
        probability: float,
        position_multiplier: float,
        atr_value: float,
        atr_multiplier: float,
        tp_rungs: Iterable[float],
        fee_bps: float,
        slippage_bps: float,
        latency_bars: int,
        metadata: Dict[str, Any],
    ) -> None:
        if latency_bars < 0:
            latency_bars = 0
        entry_index = current_index + 1 + latency_bars
        if entry_index >= len(df):
            return
        entry_time = df.index[entry_index]
        pending.append(
            PendingEntry(
                symbol=symbol,
                signal_time=signal_time,
                entry_index=entry_index,
                entry_time=entry_time,
                direction=direction,
                score=score,
                confidence=confidence,
                probability=probability,
                position_multiplier=position_multiplier,
                atr_value=atr_value,
                atr_multiplier=atr_multiplier,
                tp_rungs=tuple(float(r) for r in tp_rungs),
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                metadata=metadata,
            )
        )

    @staticmethod
    def _apply_slippage(price: float, slippage_bps: float, direction: int) -> float:
        if not math.isfinite(price):
            return price
        adjustment = 1.0 + (slippage_bps / 10_000.0) * (1 if direction > 0 else -1)
        return float(price * adjustment)

    @staticmethod
    def _update_trailing_stop(trade: TradePosition, price: float, atr_value: float) -> None:
        if not math.isfinite(price):
            return
        if trade.direction >= 0:
            trade.best_price = max(trade.best_price, price)
            if math.isfinite(atr_value) and atr_value > 0:
                candidate = trade.best_price - trade.atr_multiplier * atr_value
                trade.stop_price = max(trade.stop_price, candidate)
        else:
            trade.best_price = min(trade.best_price, price)
            if math.isfinite(atr_value) and atr_value > 0:
                candidate = trade.best_price + trade.atr_multiplier * atr_value
                trade.stop_price = min(trade.stop_price, candidate)

    def _process_trade_bar(
        self,
        trade: TradePosition,
        path: Iterable[float],
        atr_value: float,
    ) -> Tuple[Optional[str], Optional[float]]:
        prices = list(path)
        if not prices:
            return None, None

        take_profits = trade.take_profits[:]
        if trade.direction >= 0:
            take_profits.sort()
        else:
            take_profits.sort(reverse=True)

        exit_reason: Optional[str] = None
        exit_price: Optional[float] = None

        for price in prices:
            if not math.isfinite(price):
                continue
            self._update_trailing_stop(trade, price, atr_value)
            stop = trade.stop_price
            if trade.direction >= 0:
                if math.isfinite(stop) and price <= stop:
                    exit_reason = "stop"
                    exit_price = stop
                    break
                for target in take_profits:
                    if math.isfinite(target) and price >= target:
                        exit_reason = "take_profit"
                        exit_price = target
                        break
            else:
                if math.isfinite(stop) and price >= stop:
                    exit_reason = "stop"
                    exit_price = stop
                    break
                for target in take_profits:
                    if math.isfinite(target) and price <= target:
                        exit_reason = "take_profit"
                        exit_price = target
                        break
            if exit_reason:
                break

        if exit_reason is None and math.isfinite(prices[-1]):
            exit_price = float(prices[-1])

        return exit_reason, exit_price

    def _compute_trade_return(
        self,
        trade: TradePosition,
        exit_price: float,
    ) -> float:
        if not math.isfinite(exit_price) or not math.isfinite(trade.entry_price):
            return 0.0
        direction = 1 if trade.direction >= 0 else -1
        raw_return = direction * (exit_price - trade.entry_price) / trade.entry_price
        fee_penalty = 2.0 * (trade.fee_bps / 10_000.0)
        net_return = raw_return - fee_penalty
        return trade.position_multiplier * net_return

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        conf_thresh = params.get("min_score")
        if conf_thresh is None:
            conf_thresh = params.get("confidence_threshold", 0.0)
        prob_thresh = params.get("min_prob")
        if prob_thresh is None:
            prob_thresh = params.get("prob_threshold", 0.0)
        atr_mult = float(params.get("atr_mult_sl", params.get("stop_multiplier", 1.5)))
        tp_rungs = params.get("tp_rungs")
        if tp_rungs is None:
            tp_rungs = params.get("tp_multipliers", (1.0, 2.0, 3.0))
        fee_bps = float(params.get("fee_bps", params.get("fees_bps", 0.0)))
        slippage_bps = float(params.get("slippage_bps", params.get("slippage", 0.0)))
        latency_bars = int(params.get("latency_bars", 0))
        max_concurrent = int(params.get("max_concurrent", params.get("max_positions", 1)))
        start_ts: Optional[pd.Timestamp] = params.get("start_ts")
        end_ts: Optional[pd.Timestamp] = params.get("end_ts")

        if isinstance(tp_rungs, (list, tuple)):
            tp_rungs_tuple = tuple(float(x) for x in tp_rungs)
        else:
            tp_rungs_tuple = (float(tp_rungs),)

        timestamps = sorted(
            set().union(*(df.index for df in self.historical_data.values()))
        )

        equity = 1.0
        equity_curve: List[Tuple[pd.Timestamp, float]] = []
        trade_returns: List[float] = []
        trades: List[Dict[str, Any]] = []

        pending_entries: List[PendingEntry] = []
        open_trades: List[TradePosition] = []
        active_symbols: Dict[str, int] = {}

        def _within_range(ts: pd.Timestamp) -> bool:
            if start_ts is not None and ts < start_ts:
                return False
            if end_ts is not None and ts > end_ts:
                return False
            return True

        for current_time in timestamps:
            if not isinstance(current_time, pd.Timestamp):
                current_time = pd.Timestamp(current_time)

            # Update open trades first (trailing + exits)
            for trade in list(open_trades):
                df = self.historical_data.get(trade.symbol)
                if df is None or current_time not in df.index:
                    continue
                bar = df.loc[current_time]
                history = df.loc[:current_time]
                atr_value = self._compute_atr(history)
                if math.isfinite(atr_value) and atr_value > 0:
                    trade.atr_value = atr_value
                path_info = self._simulate_intrabar_path(bar)
                prices = path_info.get("prices", []) if isinstance(path_info, dict) else path_info
                exit_reason, exit_price = self._process_trade_bar(trade, prices, trade.atr_value)
                if exit_reason is None:
                    continue
                exit_price = self._apply_slippage(float(exit_price), trade.slippage_bps, -trade.direction)
                trade_return = self._compute_trade_return(trade, exit_price)
                equity *= (1.0 + trade_return)
                trade_returns.append(trade_return)
                holding_bars = df.index.get_loc(current_time) - trade.entry_index
                trades.append(
                    {
                        "symbol": trade.symbol,
                        "entry_time": trade.entry_time,
                        "exit_time": current_time,
                        "direction": trade.direction_label(),
                        "entry_price": trade.entry_price,
                        "exit_price": exit_price,
                        "score": trade.score,
                        "confidence": trade.confidence,
                        "probability": trade.probability,
                        "position_multiplier": trade.position_multiplier,
                        "return": trade_return,
                        "reason": exit_reason,
                        "holding_bars": holding_bars,
                        "metadata": trade.metadata,
                    }
                )
                open_trades.remove(trade)
                active_symbols.pop(trade.symbol, None)

            # Activate pending entries scheduled for this timestamp
            for entry in list(pending_entries):
                if current_time != entry.entry_time:
                    continue
                df = self.historical_data.get(entry.symbol)
                if df is None or entry.entry_time not in df.index:
                    pending_entries.remove(entry)
                    continue
                bar = df.loc[entry.entry_time]
                entry_price = float(bar.get("open"))
                if not math.isfinite(entry_price):
                    pending_entries.remove(entry)
                    continue
                entry_price = self._apply_slippage(entry_price, entry.slippage_bps, entry.direction)
                atr_value = entry.atr_value
                if not math.isfinite(atr_value) or atr_value <= 0:
                    history = df.iloc[: entry.entry_index]
                    atr_value = self._compute_atr(history)
                if not math.isfinite(atr_value) or atr_value <= 0:
                    pending_entries.remove(entry)
                    continue
                risk = entry.atr_multiplier * atr_value
                if entry.direction >= 0:
                    stop_price = entry_price - risk
                    take_profits = [entry_price + rung * risk for rung in entry.tp_rungs]
                else:
                    stop_price = entry_price + risk
                    take_profits = [entry_price - rung * risk for rung in entry.tp_rungs]
                trade = TradePosition(
                    symbol=entry.symbol,
                    entry_time=entry.entry_time,
                    entry_index=entry.entry_index,
                    direction=entry.direction,
                    entry_price=entry_price,
                    score=entry.score,
                    confidence=entry.confidence,
                    probability=entry.probability,
                    position_multiplier=entry.position_multiplier,
                    atr_multiplier=entry.atr_multiplier,
                    stop_price=stop_price,
                    take_profits=take_profits,
                    atr_value=atr_value,
                    best_price=entry_price,
                    metadata=entry.metadata,
                    signal_time=entry.signal_time,
                    fee_bps=entry.fee_bps,
                    slippage_bps=entry.slippage_bps,
                )
                open_trades.append(trade)
                active_symbols[trade.symbol] = active_symbols.get(trade.symbol, 0) + 1
                pending_entries.remove(entry)

            if not _within_range(current_time):
                equity_curve.append((current_time, equity))
                continue

            macro_ok = True
            try:
                macro_ok = bool(self.macro_filter())
            except Exception:
                macro_ok = True

            if macro_ok:
                available_slots = max_concurrent - len(open_trades) - len(pending_entries)
                if available_slots > 0:
                    for symbol, df in self.historical_data.items():
                        if current_time not in df.index:
                            continue
                        if active_symbols.get(symbol, 0) > 0:
                            continue
                        if any(pe.symbol == symbol for pe in pending_entries):
                            continue
                        idx = df.index.get_loc(current_time)
                        if idx <= 0 or idx >= len(df) - 1:
                            continue
                        window = df.iloc[: idx + 1]
                        atr_value = self._compute_atr(window)
                        if not math.isfinite(atr_value) or atr_value <= 0:
                            continue
                        try:
                            evaluation = self.evaluate_signal(window, symbol)
                        except Exception as exc:
                            logger.debug("Signal evaluation failed for %s: %s", symbol, exc, exc_info=True)
                            continue
                        signal = self._normalise_signal(evaluation)
                        score = float(signal.get("score", 0.0))
                        confidence = float(signal.get("confidence", score))
                        if score < conf_thresh or confidence < conf_thresh:
                            continue
                        try:
                            probability = self._call_predict_prob(signal, symbol)
                        except Exception as exc:
                            logger.debug("Probability model failed for %s: %s", symbol, exc, exc_info=True)
                            continue
                        if probability < prob_thresh:
                            continue
                        direction = self._direction_multiplier(signal.get("direction"))
                        position_multiplier = float(self.position_size_func(confidence))
                        if position_multiplier <= 0:
                            continue
                        metadata = signal.get("metadata") or {}
                        self._schedule_entry(
                            pending_entries,
                            symbol,
                            df,
                            idx,
                            current_time,
                            direction,
                            score,
                            confidence,
                            probability,
                            position_multiplier,
                            atr_value,
                            atr_mult,
                            tp_rungs_tuple,
                            fee_bps,
                            slippage_bps,
                            latency_bars,
                            metadata,
                        )
                        available_slots -= 1
                        if available_slots <= 0:
                            break

            equity_curve.append((current_time, equity))

        # Liquidate any residual positions at the final available close
        for trade in list(open_trades):
            df = self.historical_data.get(trade.symbol)
            if df is None:
                continue
            last_time = df.index[-1]
            last_price = float(df.iloc[-1]["close"])
            last_price = self._apply_slippage(last_price, trade.slippage_bps, -trade.direction)
            trade_return = self._compute_trade_return(trade, last_price)
            equity *= (1.0 + trade_return)
            trade_returns.append(trade_return)
            trades.append(
                {
                    "symbol": trade.symbol,
                    "entry_time": trade.entry_time,
                    "exit_time": last_time,
                    "direction": trade.direction_label(),
                    "entry_price": trade.entry_price,
                    "exit_price": last_price,
                    "score": trade.score,
                    "confidence": trade.confidence,
                    "probability": trade.probability,
                    "position_multiplier": trade.position_multiplier,
                    "return": trade_return,
                    "reason": "close_out",
                    "holding_bars": df.index.get_loc(last_time) - trade.entry_index,
                    "metadata": trade.metadata,
                }
            )
            active_symbols.pop(trade.symbol, None)

        equity_curve.append((timestamps[-1] if timestamps else pd.Timestamp.utcnow(), equity))

        equity_series = [value for _, value in equity_curve]
        performance = {
            "final_equity": equity,
            "total_return": equity - 1.0,
            "sharpe": sharpe_ratio(trade_returns),
            "max_drawdown": max_drawdown(equity_series) if equity_series else 0.0,
            "num_trades": len(trades),
        }

        result: Dict[str, Any] = {
            "performance": performance,
            "trades": trades,
            "equity_curve": pd.DataFrame(equity_curve, columns=["timestamp", "equity"]).set_index("timestamp"),
        }
        if trades:
            result["trades_df"] = pd.DataFrame(trades)
        return result


def grid_search(backtester: Backtester, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Evaluate the backtester over all combinations of parameters."""

    import itertools

    results: List[Dict[str, Any]] = []
    keys = list(param_grid.keys())
    for values in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, values))
        outcome = backtester.run(params)
        performance = outcome.get("performance", {})
        results.append({"params": params, "performance": performance})
    return results


def compute_buy_and_hold_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Compute buy-and-hold returns from a Binance OHLCV dataframe."""

    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")

    out = df.copy()
    close = out["close"].astype(float)
    out["pnl"] = close.pct_change().fillna(0.0)
    out["equity"] = (1 + out["pnl"]).cumprod()
    return out


def generate_trades_from_ohlcv(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    take_profit: float = 0.01,
    stop_loss: float = 0.01,
) -> List[Dict[str, Any]]:
    """Generate labelled trades from OHLCV data."""

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError("DataFrame must contain open, high, low and close columns")

    trades: List[Dict[str, Any]] = []
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    for i in range(len(df) - 1):
        entry_price = float(df.iloc[i]["open"])
        next_high = float(df.iloc[i + 1]["high"])
        next_low = float(df.iloc[i + 1]["low"])
        exit_price = float(df.iloc[i + 1]["close"])
        outcome = "sl"
        if next_low <= entry_price * (1 - stop_loss):
            exit_price = entry_price * (1 - stop_loss)
            outcome = "sl"
        elif next_high >= entry_price * (1 + take_profit):
            exit_price = entry_price * (1 + take_profit)
            outcome = "tp1"
        else:
            outcome = "tp1" if exit_price > entry_price else "sl"
        trades.append(
            {
                "symbol": symbol,
                "entry": entry_price,
                "exit": exit_price,
                "entry_time": df.index[i],
                "exit_time": df.index[i + 1],
                "outcome": outcome,
            }
        )
    return trades
