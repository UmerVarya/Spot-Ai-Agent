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

import numpy as np
import pandas as pd
from typing import Callable, Dict, Iterable, List, Tuple, Any

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
                score, pattern, conf, extra = self.evaluate_signal(window, symbol)
                if conf < conf_thresh:
                    continue
                prob = self.predict_prob(score, conf, 'unknown', 0.0, 50.0, 5.0, str(pattern))
                if prob < prob_thresh:
                    continue
                # Determine position size multiplier
                position_mult = self.position_size_func(conf)
                if position_mult <= 0:
                    continue
                # Compute entry/exit using simple ATR stops
                atr = (window['high'] - window['low']).rolling(14).mean().iloc[-1]
                if atr != atr or atr == 0:
                    continue
                stop = window['close'].iloc[-1] - stop_mult * atr
                take_profits = [window['close'].iloc[-1] + m * atr for m in tp_mults]
                entry_price = window['close'].iloc[-1]
                # Simulate next bar outcome
                # For demonstration we use actual next close price
                try:
                    next_price = df.loc[t:].iloc[1]['close']
                except Exception:
                    continue
                pnl = (next_price - entry_price) / entry_price
                # Apply stop/take profit logic
                if pnl < -stop_mult * atr / entry_price:
                    pnl = -stop_mult * atr / entry_price
                else:
                    for tp_level in take_profits:
                        if (tp_level - entry_price) / entry_price <= pnl:
                            pnl = (tp_level - entry_price) / entry_price
                            break
                # Update equity
                trade_return = position_mult * pnl
                equity *= (1 + trade_return)
                equity_curve.append(equity)
                trade_returns.append(trade_return)
                trade_log.append({
                    'timestamp': t,
                    'symbol': symbol,
                    'score': score,
                    'confidence': conf,
                    'prob': prob,
                    'return': trade_return,
                })
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
