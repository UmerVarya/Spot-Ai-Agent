"""
Enhanced diversification utilities for the Spot AI Super Agent.

This module improves upon the original diversification logic by allowing
configurable correlation thresholds and lookback windows.  It computes
log‑return correlations across multiple rolling windows to better
capture market regimes and reduce the chance of selecting highly
correlated symbols.  The correlation horizon and maximum number of
trades may be tuned via environment variables.  If data for a symbol
is missing or cannot be aligned, that symbol is treated as
uncorrelated and may still be selected.

Functions
---------
select_diversified_signals(signals, max_corr=None, max_trades=None)
    Given a list of candidate trade signals, return a subset whose
    pairwise correlations are below the threshold.  Uses multi‑timeframe
    correlation if configured.

Notes
-----
Correlation is computed on log returns rather than raw prices to
stabilise variance.  Two lookback windows are used by default
(``DIVERSIFY_LOOKBACK_SHORT`` and ``DIVERSIFY_LOOKBACK_LONG``).  The
final correlation metric is the maximum absolute correlation across the
two windows so that periods of transient decoupling do not hide
structural relationships.
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from trade_utils import get_price_data  # avoid relative import issues

# --- Configuration via environment variables ---
DIVERSIFY_LOOKBACK_SHORT = int(os.getenv("DIVERSIFY_LOOKBACK_SHORT", 96))
DIVERSIFY_LOOKBACK_LONG = int(os.getenv("DIVERSIFY_LOOKBACK_LONG", 288))
DIVERSIFY_MAX_CORR = float(os.getenv("DIVERSIFY_MAX_CORR", 0.7))
DIVERSIFY_MAX_TRADES = int(os.getenv("DIVERSIFY_MAX_TRADES", 3))


def _compute_returns(symbol: str, lookback: int) -> Optional[pd.Series]:
    """Compute log returns for a symbol given a lookback.

    Returns ``None`` if insufficient data.
    """
    try:
        df = get_price_data(symbol)
    except Exception:
        return None
    if df is None or df.empty or len(df) < lookback:
        return None
    closes = df['close'].astype(float).tail(lookback)
    pct = closes.pct_change().dropna()
    if pct.empty:
        return None
    return np.log1p(pct)


def _compute_corr_matrix(symbols: List[str], lookback: int) -> Optional[pd.DataFrame]:
    """Compute a correlation matrix for a list of symbols and lookback.
    Returns ``None`` if insufficient data.
    """
    returns_map: Dict[str, pd.Series] = {}
    for sym in symbols:
        r = _compute_returns(sym, lookback)
        if r is not None:
            returns_map[sym] = r
    if not returns_map:
        return None
    returns_df = pd.DataFrame(returns_map).dropna()
    if returns_df.empty:
        return None
    return returns_df.corr()


def _max_abs_corr(sym: str, sel_sym: str, corr_short: Optional[pd.DataFrame], corr_long: Optional[pd.DataFrame]) -> float:
    """Return the maximum absolute correlation between two symbols across windows.
    If correlation cannot be computed, returns 0.0.
    """
    values = []
    for corr in (corr_short, corr_long):
        if corr is None:
            continue
        try:
            val = float(corr.loc[sym, sel_sym])
            values.append(abs(val))
        except Exception:
            continue
    return max(values) if values else 0.0


def select_diversified_signals(signals: List[Dict], max_corr: Optional[float] = None, max_trades: Optional[int] = None) -> List[Dict]:
    """Select a diversified subset of candidate signals.

    Parameters
    ----------
    signals : list of dict
        Candidate trade signals sorted by desirability.
    max_corr : float, optional
        Override the default correlation threshold.  If ``None`` uses the
        environment default ``DIVERSIFY_MAX_CORR``.
    max_trades : int, optional
        Maximum number of trades to select.  If ``None`` uses the
        environment default ``DIVERSIFY_MAX_TRADES``.

    Returns
    -------
    list of dict
        Selected trade signals that are sufficiently diversified.

    Notes
    -----
    The function always selects the top signal.  It then iterates
    through the remaining signals and adds each one only if its
    correlation with every already selected symbol is below the
    threshold.  Correlations are measured across both short and long
    lookbacks, and the maximum absolute value is used.
    """
    if not signals:
        return []
    max_corr_val = max_corr if max_corr is not None else DIVERSIFY_MAX_CORR
    max_trades_val = max_trades if max_trades is not None else DIVERSIFY_MAX_TRADES
    selected: List[Dict] = [signals[0]]
    if len(signals) == 1 or max_trades_val <= 1:
        return selected
    symbols = [s['symbol'] for s in signals]
    corr_short = _compute_corr_matrix(symbols, DIVERSIFY_LOOKBACK_SHORT)
    corr_long = _compute_corr_matrix(symbols, DIVERSIFY_LOOKBACK_LONG)
    for sig in signals[1:]:
        if len(selected) >= max_trades_val:
            break
        sym = sig['symbol']
        diversified = True
        for sel in selected:
            sel_sym = sel['symbol']
            corr_val = _max_abs_corr(sym, sel_sym, corr_short, corr_long)
            if corr_val >= max_corr_val:
                diversified = False
                break
        if diversified:
            selected.append(sig)
    return selected
