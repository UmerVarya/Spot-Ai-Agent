"""
Risk and performance metrics for the Spot‑AI Agent.

This module includes functions to compute risk‑adjusted performance
measures (Sharpe ratio, Calmar ratio, maximum drawdown) and
distribution‑based risk measures (Value‑at‑Risk and Expected
Shortfall).  These metrics can be used to monitor strategy health
over time and to adjust trading intensity when risk rises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable


def sharpe_ratio(returns: Iterable[float], risk_free: float = 0.0) -> float:
    """
    Compute the annualised Sharpe ratio of a series of returns.

    Parameters
    ----------
    returns : iterable of float
        Periodic returns (e.g. daily or per trade).
    risk_free : float, optional
        Risk‑free rate per period (default 0).

    Returns
    -------
    float
        The Sharpe ratio.  Returns NaN if insufficient data.
    """
    r = np.asarray(list(returns), dtype=float)
    if len(r) < 2:
        return float('nan')
    excess = r - risk_free
    mean = excess.mean()
    std = excess.std(ddof=1)
    if std == 0:
        return float('nan')
    # Annualise assuming 252 trading days (or adjust if returns frequency differs)
    return float(np.sqrt(252) * mean / std)


def max_drawdown(equity_curve: Iterable[float]) -> float:
    """
    Calculate the maximum drawdown of an equity curve.
    """
    equity = np.asarray(list(equity_curve), dtype=float)
    if len(equity) == 0:
        return float('nan')
    cum_max = np.maximum.accumulate(equity)
    drawdowns = (equity - cum_max) / cum_max
    return float(drawdowns.min())


def calmar_ratio(returns: Iterable[float]) -> float:
    """
    Compute the Calmar ratio (annualised return divided by max drawdown).
    """
    r = np.asarray(list(returns), dtype=float)
    if len(r) < 2:
        return float('nan')
    # Cumulative return over period
    total_return = np.prod(1 + r) - 1
    annualised_return = (1 + total_return) ** (252 / len(r)) - 1
    mdd = abs(max_drawdown(np.cumprod(1 + r)))
    if mdd == 0:
        return float('nan')
    return float(annualised_return / mdd)


def value_at_risk(returns: Iterable[float], alpha: float = 0.05) -> float:
    """
    Estimate Value‑at‑Risk (VaR) via historical simulation.

    Parameters
    ----------
    returns : iterable of float
        Historical returns.
    alpha : float
        Significance level (0.05 = 95 % VaR).

    Returns
    -------
    float
        The VaR threshold (negative number).  Values below this
        threshold are expected to occur with probability ``alpha``.
    """
    r = np.sort(np.asarray(list(returns), dtype=float))
    if len(r) == 0:
        return float('nan')
    idx = int(alpha * len(r))
    return float(r[idx])


def expected_shortfall(returns: Iterable[float], alpha: float = 0.05) -> float:
    """
    Compute the Expected Shortfall (Conditional VaR) at level ``alpha``.
    """
    r = np.sort(np.asarray(list(returns), dtype=float))
    if len(r) == 0:
        return float('nan')
    idx = int(alpha * len(r))
    tail = r[:idx]
    if len(tail) == 0:
        return float('nan')
    return float(tail.mean())
