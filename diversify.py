"""
Diversification utilities for the Spot AI Super Agent.

This module provides helper functions to compute correlations among
candidate symbols and select a subset of trades that are sufficiently
uncorrelated.  Diversification helps reduce portfolio drawdowns by
avoiding opening multiple highly correlated positions at the same time.

Functions
---------
select_diversified_signals(signals, max_corr=0.8, max_trades=2)
    Given a list of candidate trade signals, return a subset whose
    pairwise correlations are below ``max_corr``.  At most
    ``max_trades`` signals are selected.

Note
----
Correlation is computed using recent percentage returns from the
underlying price series (5‑minute candles by default).  If price
data for a symbol cannot be fetched or aligned, the symbol is
treated as uncorrelated and may be selected.  The first signal (the
highest score) is always selected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import List, Dict

# Note: Do not use relative import here because this module may be executed
# as a top-level module.  Import directly from trade_utils.
from trade_utils import get_price_data


def _compute_returns(symbol: str, lookback: int = 100) -> pd.Series | None:
    """Fetch recent price data for ``symbol`` and compute log returns.

    Parameters
    ----------
    symbol : str
        The trading symbol (e.g., "BTCUSDT").
    lookback : int, optional
        Number of recent 5‑minute candles to use.  Defaults to 100.

    Returns
    -------
    pandas.Series or None
        Series of log returns indexed by the candle index.  Returns
        ``None`` if data is missing or empty.
    """
    try:
        df = get_price_data(symbol)
    except Exception:
        return None
    if df is None or df.empty or len(df) < lookback:
        return None
    closes = df['close'].astype(float).tail(lookback)
    # Percentage returns; use log returns to stabilise variance
    pct = closes.pct_change().dropna()
    if pct.empty:
        return None
    log_ret = np.log1p(pct)
    return log_ret


def _compute_corr_matrix(symbols: List[str]) -> pd.DataFrame | None:
    """Compute a correlation matrix of log returns for a list of symbols.

    Parameters
    ----------
    symbols : list of str
        Candidate symbols.

    Returns
    -------
    pandas.DataFrame or None
        Correlation matrix indexed by symbol.  If insufficient data
        exists for all symbols, returns ``None``.
    """
    returns_map: Dict[str, pd.Series] = {}
    for sym in symbols:
        r = _compute_returns(sym)
        if r is not None:
            returns_map[sym] = r
    if not returns_map:
        return None
    # Align all return series by index; drop rows with missing values
    returns_df = pd.DataFrame(returns_map).dropna()
    if returns_df.empty:
        return None
    corr = returns_df.corr()
    return corr


def select_diversified_signals(signals: List[Dict], max_corr: float = 0.8, max_trades: int = 2) -> List[Dict]:
    """Select a diversified subset of trade signals.

    Parameters
    ----------
    signals : list of dict
        Candidate trade signals, each with at least a ``"symbol"`` key.
        The list should be sorted in descending order of desirability
        (e.g., by score).
    max_corr : float, optional
        Maximum allowed absolute correlation between any pair of selected
        signals.  Defaults to 0.8.
    max_trades : int, optional
        Maximum number of signals to select.  Defaults to 2.

    Returns
    -------
    list of dict
        The selected, diversified signals.

    Notes
    -----
    The function always selects the first signal.  It then iterates
    through the remaining signals and adds each to the selected set
    only if its correlation with every already selected symbol is less
    than ``max_corr`` (in absolute value).  If correlation cannot be
    computed (e.g., missing data), the signal is treated as
    sufficiently diversified.
    """
    if not signals:
        return []
    # Always select the top signal
    selected = [signals[0]]
    if len(signals) == 1 or max_trades == 1:
        return selected
    # Compute correlation matrix for all candidate symbols
    symbols = [s['symbol'] for s in signals]
    corr_matrix = _compute_corr_matrix(symbols)
    # Greedy selection of diversified signals
    for sig in signals[1:]:
        if len(selected) >= max_trades:
            break
        sym = sig['symbol']
        diversified = True
        for sel in selected:
            sel_sym = sel['symbol']
            if corr_matrix is not None:
                try:
                    corr_val = float(corr_matrix.loc[sym, sel_sym])
                except Exception:
                    corr_val = None
                # Use absolute correlation
                if corr_val is not None and abs(corr_val) >= max_corr:
                    diversified = False
                    break
        if diversified:
            selected.append(sig)
    return selected
    
