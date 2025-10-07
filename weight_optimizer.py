import os
from typing import Iterable, Optional, Tuple

import pandas as pd

from log_utils import setup_logger
from trade_schema import normalise_history_columns
from trade_storage import TRADE_HISTORY_FILE

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNAL_LOG_FILE = os.getenv("SIGNAL_LOG_FILE", os.path.join(_MODULE_DIR, "signal_log.csv"))

logger = setup_logger(__name__)


MIN_CONTEXT_OBSERVATIONS = 5


def _clean_categorical(series: pd.Series, *, valid: Optional[Iterable[str]] = None) -> pd.Series:
    """Return a normalised categorical series with missing values filled."""

    if series is None:
        return pd.Series(dtype="object")
    normalised = series.astype(str).str.strip()
    normalised = normalised.replace({"nan": "", "NaT": "", "None": ""})
    normalised = normalised.replace({"": "unknown"})
    normalised = normalised.fillna("unknown")
    normalised = normalised.str.title()
    if valid is not None:
        normalised = normalised.where(normalised.isin({v.title() for v in valid}), "Other")
    return normalised


def _derive_trade_type(df: pd.DataFrame) -> pd.Series:
    """Derive a trade type label from available trade metadata."""

    if "pattern" in df.columns:
        base = df["pattern"].astype(str)
    elif "strategy" in df.columns:
        base = df["strategy"].astype(str)
    elif "direction" in df.columns:
        base = df["direction"].astype(str)
    else:
        return pd.Series(["Unknown"] * len(df), index=df.index)
    trade_type = base.str.strip().replace({"": "unknown", "nan": "unknown"})
    trade_type = trade_type.fillna("unknown")
    return trade_type.str.title()


def _volatility_regime(series: pd.Series) -> Tuple[pd.Series, Optional[pd.Series]]:
    """Return categorical volatility regime labels and the numeric series used."""

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.empty:
        return pd.Series(["Unknown"] * len(series), index=series.index), None
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(["Unknown"] * len(series), index=series.index), numeric
    quantiles = valid.quantile([0.25, 0.5, 0.75]).to_dict()
    q1 = quantiles.get(0.25, valid.min())
    q2 = quantiles.get(0.5, valid.median())
    q3 = quantiles.get(0.75, valid.max())

    def classify(value: float) -> str:
        if pd.isna(value):
            return "Unknown"
        if value <= q1:
            return "Low"
        if value <= q2:
            return "Moderate"
        if value <= q3:
            return "High"
        return "Extreme"

    regimes = numeric.map(classify)
    return regimes, numeric


def _win_rate_factor(scores: pd.Series, ema_span: Optional[int]) -> Optional[float]:
    """Return the scaling factor based on the win rate within ``scores``."""

    if scores.empty:
        return None
    if ema_span:
        win_rate = scores.ewm(span=ema_span, adjust=False).mean().iloc[-1]
    else:
        win_rate = scores.mean()
    if pd.isna(win_rate):
        return None
    win_rate = max(0.0, min(1.0, float(win_rate)))
    return 0.5 + win_rate / 2

def optimize_indicator_weights(
    base_weights: dict, lookback: int = 200, ema_span: Optional[int] = None
) -> dict:
    """Return adjusted indicator weights based on recent performance.

    Reads the signal log and completed trade log, joins them on timestamp and symbol,
    and calculates a context-aware win rate for each indicator. Each indicator's
    performance is segmented by market session, volatility regime and derived trade
    type. The base weight is scaled by a factor between 0.5 and 1.0 depending on the
    win rate (0..1), emphasising the contexts with sufficient trade history. When
    ``ema_span`` is provided the win rate uses an exponentially weighted mean so
    that more recent trades have a larger impact while still considering a deeper
    history. If insufficient data is available, the original weights are returned.
    """
    if not (os.path.exists(SIGNAL_LOG_FILE) and os.path.exists(TRADE_HISTORY_FILE)):
        return base_weights
    try:
        if ema_span is not None and ema_span <= 0:
            logger.warning(
                "Invalid ema_span %s supplied to optimize_indicator_weights; ignoring.",
                ema_span,
            )
            ema_span = None

        signals = pd.read_csv(SIGNAL_LOG_FILE).tail(lookback)
        trades = pd.read_csv(TRADE_HISTORY_FILE).tail(lookback)
        trades = normalise_history_columns(trades)

        # ``normalise_history_columns`` only renames headers – it does not
        # guarantee that the important ones actually exist.  Older trade logs
        # (or partially written rows) sometimes omit ``timestamp`` or
        # ``symbol`` which would make the ``merge`` below raise a ``KeyError``.
        # Gracefully bail out instead of letting the exception bubble up so the
        # caller keeps the default weights.
        required_trade_cols = {"timestamp", "symbol", "outcome"}
        missing_trade_cols = required_trade_cols - set(trades.columns)
        if missing_trade_cols:
            logger.warning(
                "Skipping weight optimisation – missing trade columns: %s",
                ", ".join(sorted(missing_trade_cols)),
            )
            return base_weights

        required_signal_cols = {"timestamp", "symbol"}
        missing_signal_cols = required_signal_cols - set(signals.columns)
        if missing_signal_cols:
            logger.warning(
                "Skipping weight optimisation – missing signal columns: %s",
                ", ".join(sorted(missing_signal_cols)),
            )
            return base_weights

        # Normalize keys consistently on both frames
        for df in (signals, trades):
            if "symbol" in df.columns:
                df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], errors="coerce", utc=True
                )

        if trades.empty or "outcome" not in trades.columns:
            return base_weights

        context_columns = [
            col
            for col in ["session", "volatility", "pattern", "strategy", "direction"]
            if col in trades.columns
        ]

        merged = pd.merge(
            signals,
            trades[["timestamp", "symbol", "outcome", *context_columns]],
            on=["timestamp", "symbol"],
            how="inner",
        )
        if merged.empty:
            return base_weights
        merged = merged.sort_values("timestamp")

        if "session" in merged.columns:
            session_values = _clean_categorical(
                merged["session"], valid={"Asia", "Europe", "US", "Unknown"}
            )
            merged["session"] = session_values
        else:
            merged["session"] = "Unknown"

        if "volatility" in merged.columns:
            volatility_labels, _ = _volatility_regime(merged["volatility"])
            merged["volatility_regime"] = volatility_labels
        else:
            merged["volatility_regime"] = "Unknown"

        merged["trade_type"] = _derive_trade_type(merged)
        context_group_cols = ["session", "volatility_regime", "trade_type"]
        new_weights = base_weights.copy()
        for key in base_weights.keys():
            col = f"{key}_trigger"
            if col not in merged.columns:
                continue
            triggered = merged[merged[col] > 0]
            if triggered.empty:
                continue
            outcome_score = (
                triggered["outcome"].astype(str).str.lower() == "win"
            ).astype(float)
            factors = []
            supports = []
            for _, group in triggered.groupby(context_group_cols, dropna=False):
                if len(group) < MIN_CONTEXT_OBSERVATIONS:
                    continue
                context_scores = outcome_score.loc[group.index]
                factor = _win_rate_factor(context_scores, ema_span)
                if factor is None:
                    continue
                factors.append(factor)
                supports.append(len(group))

            if factors:
                total_support = sum(supports)
                weighted_factor = sum(f * s for f, s in zip(factors, supports)) / total_support
                new_weights[key] = round(base_weights[key] * weighted_factor, 3)
                continue

            fallback_factor = _win_rate_factor(outcome_score, ema_span)
            if fallback_factor is not None:
                new_weights[key] = round(base_weights[key] * fallback_factor, 3)
        return new_weights
    except Exception as exc:
        logger.exception("Failed to optimize indicator weights: %s", exc)
        return base_weights
