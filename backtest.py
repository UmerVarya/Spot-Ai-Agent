"""
Historical backtesting framework for the Spot‑AI Agent.

This module allows you to simulate trading over historical OHLCV data
using the decision logic of the Spot‑AI Agent.  It is intentionally
agnostic about the specific strategy components – you must provide
callbacks for signal evaluation, macro gating and position sizing.

Key features
------------

* Iterate through historical candles and replicate the agent's trade
  decisions (excluding LLM calls).  At each step the backtester calls
  your signal evaluation function, machine‑learning classifier and
  risk‑control functions.
* Maintain cash and open positions with ATR‑based stop‑loss and
  take‑profit logic.
* Compute performance statistics (final equity, Sharpe ratio, max
  drawdown) and log each trade for further analysis.
* Parameter sweeps: evaluate multiple parameter combinations via
  exhaustive grid search or random sampling.

Usage::

    from backtest import Backtester, grid_search
    bt = Backtester(historical_data, evaluate_signal, predict_prob, macro_filter)
    results = bt.run(params)
    # For parameter optimisation
    best = grid_search(bt, param_grid)
    logger.info(best)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from risk_metrics import sharpe_ratio, max_drawdown
from log_utils import setup_logger

logger = setup_logger(__name__)


class Backtester:
    """Simple backtester for the Spot‑AI decision logic."""

    def __init__(
        self,
        historical_data: Dict[str, pd.DataFrame],
        evaluate_signal: Callable[[pd.DataFrame, str], Tuple[float, Any, float, Any]],
        predict_prob: Callable[[float, float, str, float, float, float, str], float],
        macro_filter: Callable[[], bool],
        position_size_func: Callable[[float], float],
    ) -> None:
        """
        Initialise the backtester.

        Parameters
        ----------
        historical_data : dict
            Mapping from symbol to OHLCV DataFrame indexed by datetime.
        evaluate_signal : function
            Callback returning (score, pattern, confidence, other) given
            the recent price data for a symbol.
        predict_prob : function
            Callback returning success probability given the features
            computed by ``evaluate_signal`` and macro variables.
        macro_filter : function
            Returns True if trading is allowed under current macro
            conditions.
        position_size_func : function
            Determines the position size multiplier given the signal
            confidence.
        """
        self.historical_data = historical_data
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

    @staticmethod
    def _evaluate_intrabar_outcome(
        prices: List[float],
        entry_price: float,
        direction_mult: int,
        stop_price: float,
        take_profits: List[float],
    ) -> Tuple[float, str, float]:
        """Determine trade PnL by walking the synthetic intrabar path."""

        if not prices or not math.isfinite(entry_price):
            return 0.0, "close", entry_price

        tp_levels = take_profits[:]
        if direction_mult > 0:
            tp_levels.sort()
        else:
            tp_levels.sort(reverse=True)

        exit_reason = "close"
        exit_price = prices[-1]

        for price in prices:
            if direction_mult > 0:
                if math.isfinite(stop_price) and price <= stop_price:
                    exit_reason = "stop"
                    exit_price = stop_price
                    break
                for target in tp_levels:
                    if math.isfinite(target) and price >= target:
                        exit_reason = "take_profit"
                        exit_price = target
                        break
                if exit_reason != "close":
                    break
            else:
                if math.isfinite(stop_price) and price >= stop_price:
                    exit_reason = "stop"
                    exit_price = stop_price
                    break
                for target in tp_levels:
                    if math.isfinite(target) and price <= target:
                        exit_reason = "take_profit"
                        exit_price = target
                        break
                if exit_reason != "close":
                    break

        if exit_reason == "close" and math.isfinite(prices[-1]):
            exit_price = prices[-1]

        if direction_mult > 0:
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        return pnl, exit_reason, exit_price

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a backtest with the specified parameters.

        ``params`` can include keys such as ``confidence_threshold``,
        ``prob_threshold``, ``stop_multiplier`` and ``take_profit_levels``.
        Returns a dictionary of performance metrics and trade log.
        """
        # Hyper‑parameters
        conf_thresh = params.get('confidence_threshold', 6.5)
        prob_thresh = params.get('prob_threshold', 0.5)
        stop_mult = params.get('stop_multiplier', 1.5)
        tp_mults = params.get('tp_multipliers', [1.0, 2.0, 3.0])

        equity = 1.0  # start with 1 unit capital
        equity_curve: List[float] = [equity]
        trade_returns: List[float] = []
        trade_log: List[Dict[str, Any]] = []

        # Iterate over timestamps (assuming all symbols share the same index)
        # We'll synchronise on the smallest dataset
        timestamps = sorted(set().union(*(df.index for df in self.historical_data.values())))
        for t in timestamps:
            # Macro gating
            if not self.macro_filter():
                continue
            # Evaluate each symbol
            for symbol, df in self.historical_data.items():
                # Use the latest window of data up to time t
                window = df.loc[:t].tail(100)  # use last 100 bars for context
                if len(window) < 50:
                    continue
                evaluation = self.evaluate_signal(window, symbol)
                if not isinstance(evaluation, tuple) or not evaluation:
                    continue
                score = evaluation[0]
                direction = evaluation[1] if len(evaluation) > 1 else None
                position_hint = evaluation[2] if len(evaluation) > 2 else 0
                pattern = evaluation[3] if len(evaluation) > 3 else None
                if score < conf_thresh:
                    continue
                prob = self.predict_prob(score, score, 'unknown', 0.0, 50.0, 5.0, str(pattern))
                if prob < prob_thresh:
                    continue
                # Determine position size multiplier
                position_mult = self.position_size_func(score)
                if position_mult <= 0:
                    continue
                direction_mult = self._direction_multiplier(direction)
                # Compute entry/exit using simple ATR stops
                atr = (window['high'] - window['low']).rolling(14).mean().iloc[-1]
                if atr != atr or atr == 0:
                    continue
                entry_price = window['close'].iloc[-1]
                if direction_mult > 0:
                    stop_price = entry_price - stop_mult * atr
                    take_profits = [entry_price + m * atr for m in tp_mults]
                else:
                    stop_price = entry_price + stop_mult * atr
                    take_profits = [entry_price - m * atr for m in tp_mults]
                # Simulate next bar outcome
                future = df.loc[df.index > t]
                if future.empty:
                    continue
                next_bar = future.iloc[0]
                microstructure = self._simulate_intrabar_path(next_bar)
                pnl, exit_reason, exit_price = self._evaluate_intrabar_outcome(
                    microstructure.get("prices", []),
                    entry_price,
                    direction_mult,
                    stop_price,
                    take_profits,
                )
                # Update equity
                trade_return = position_mult * pnl
                equity *= (1 + trade_return)
                equity_curve.append(equity)
                trade_returns.append(trade_return)
                feature_snapshot = window.attrs.get("signal_features")
                trade_record = {
                    'timestamp': t,
                    'symbol': symbol,
                    'score': score,
                    'confidence': score,
                    'prob': prob,
                    'direction': direction,
                    'position_hint': position_hint,
                    'exit_reason': exit_reason,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'microstructure': microstructure,
                }
                if isinstance(feature_snapshot, dict) and feature_snapshot:
                    trade_record['signal_features'] = feature_snapshot.copy()
                if pattern is not None:
                    trade_record['pattern'] = pattern
                trade_log.append(trade_record)
        # Compute performance metrics
        performance = {
            'final_equity': equity,
            'total_return': equity - 1,
            'sharpe': sharpe_ratio(trade_returns),
            'max_drawdown': max_drawdown(equity_curve),
            'num_trades': len(trade_returns),
            'trade_log': trade_log,
        }
        return performance


def grid_search(backtester: Backtester, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Evaluate the backtester over all combinations of parameters.

    ``param_grid`` should map parameter names to lists of candidate
    values.  Returns the combination achieving the highest final
    equity.
    """
    import itertools
    best_perf = None
    best_params = None
    keys = list(param_grid.keys())
    for values in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, values))
        perf = backtester.run(params)
        if best_perf is None or perf['final_equity'] > best_perf['final_equity']:
            best_perf = perf
            best_params = params
    return {'best_params': best_params, 'performance': best_perf}


def compute_buy_and_hold_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Compute buy-and-hold returns from a Binance OHLCV dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a ``close`` column with price data.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` augmented with ``pnl`` (period returns) and
        ``equity`` (cumulative equity) columns.

    Notes
    -----
    The function assumes consecutive rows represent equally spaced time
    periods.  Returns are computed as percentage change of the close
    price and a starting equity of 1.0 is assumed.
    """

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
    """Generate labelled trades from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``open``, ``high``, ``low`` and ``close`` columns.
    symbol : str, optional
        Asset symbol associated with the data.
    take_profit : float, default 0.01
        Take‑profit threshold expressed as a fraction of the entry price.
    stop_loss : float, default 0.01
        Stop‑loss threshold expressed as a fraction of the entry price.

    Returns
    -------
    list of dict
        A list of dictionaries describing simulated trades.  Each
        dictionary contains ``entry``, ``exit``, ``entry_time``,
        ``exit_time`` and ``outcome`` (``"tp1"`` for wins,
        ``"sl"`` for losses).
    """

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError("DataFrame must contain open, high, low and close columns")

    trades: List[Dict[str, Any]] = []
    # Ensure we work with datetime index for logging purposes
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    for i in range(len(df) - 1):
        entry_price = float(df.iloc[i]["open"])
        next_high = float(df.iloc[i + 1]["high"])
        next_low = float(df.iloc[i + 1]["low"])
        exit_price = float(df.iloc[i + 1]["close"])
        outcome = "sl"
        # Determine which level was hit first.  Priority to stop‑loss if both
        # thresholds are crossed within the same bar.
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
