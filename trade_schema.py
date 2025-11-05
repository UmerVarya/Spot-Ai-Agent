"""Utilities describing the canonical trade log schema.

This module centralises the column ordering and alias handling for the
trade history CSV produced by the agent. Historically each component
(dashboard, storage, analytics) maintained its own ad-hoc list of
columns and legacy aliases. When new fields were added to the log or
renamed, those assumptions drifted apart which caused subtle bugs such
as the dashboard looking for ``sent_conf`` while the logger emitted
``sentiment_confidence``.

By defining the schema in a single place the codebase can import the
same constants and helper functions, reducing the likelihood of those
mismatches. ``normalise_history_columns`` exposes the normalisation
logic so that callers consistently map legacy headers onto the canonical
names used throughout the project.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable

import pandas as pd

# Canonical column order for the trade history CSV.  The list mirrors the
# headers written by :func:`trade_storage.log_trade_result` and is reused by
# the dashboard to display columns in a stable order.
TRADE_HISTORY_COLUMNS = [
    "trade_id",
    "timestamp",
    "symbol",
    "direction",
    "entry_time",
    "exit_time",
    "entry",
    "exit",
    "size",
    "notional",
    "fees",
    "slippage",
    "pnl",
    "pnl_pct",
    "outcome",
    "outcome_desc",
    "exit_reason",
    "strategy",
    "session",
    "confidence",
    "btc_dominance",
    "fear_greed",
    "sentiment_bias",
    "sentiment_confidence",
    "score",
    "pattern",
    "narrative",
    "llm_decision",
    "llm_approval",
    "llm_confidence",
    "llm_error",
    "technical_indicator_score",
    "volatility",
    "htf_trend",
    "order_imbalance",
    "order_flow_score",
    "order_flow_flag",
    "order_flow_state",
    "cvd",
    "cvd_change",
    "cvd_divergence",
    "cvd_absorption",
    "cvd_accumulation",
    "taker_buy_ratio",
    "trade_imbalance",
    "aggressive_trade_rate",
    "spoofing_intensity",
    "spoofing_alert",
    "volume_ratio",
    "price_change_pct",
    "spread_bps",
    "macro_indicator",
    "tp1_partial",
    "tp2_partial",
    "pnl_tp1",
    "pnl_tp2",
    "pnl_tp3",
    "size_tp1",
    "size_tp2",
    "size_tp3",
    "notional_tp1",
    "notional_tp2",
    "notional_tp3",
    "auction_state",
    "volume_profile_leg_type",
    "volume_profile_poc",
    "volume_profile_lvns",
    "volume_profile_bin_width",
    "volume_profile_snapshot",
    "lvn_entry_level",
    "lvn_stop",
    "poc_target",
    "orderflow_state_detail",
    "orderflow_features",
    "orderflow_snapshot",
]

# Mapping of normalised legacy column names to their canonical equivalents.
# Keys are lower-case strings with whitespace and underscores removed to make
# matching tolerant of formatting differences such as ``Entry Price`` or
# ``entry_price``.
COLUMN_SYNONYMS: Dict[str, str] = {
    "tradeid": "trade_id",
    "time": "timestamp",
    "timestamp": "timestamp",
    "pair": "symbol",
    "ticker": "symbol",
    "symbol": "symbol",
    "entryprice": "entry",
    "entry": "entry",
    "entrytime": "entry_time",
    "entrytimestamp": "entry_time",
    "exitprice": "exit",
    "exit": "exit",
    "exittime": "exit_time",
    "timeexit": "exit_time",
    "exittimestamp": "exit_time",
    "positionsize": "position_size",
    "qty": "position_size",
    "quantity": "position_size",
    "usdsize": "size",
    "size": "size",
    "side": "direction",
    "position": "direction",
    "direction": "direction",
    "tradeoutcome": "outcome",
    "traderesult": "outcome",
    "result": "outcome",
    "outcome": "outcome",
    "pnlusd": "pnl",
    "pnl$": "pnl",
    "netpnl": "net_pnl",
    "pnl": "pnl",
    "pnlpercent": "pnl_pct",
    "pnl%": "pnl_pct",
    "pnlpct": "pnl_pct",
    "notionalvalue": "notional",
    "notionalusd": "notional",
    "notional": "notional",
    "sentimentconfidence": "sentiment_confidence",
    "sentconfidence": "sentiment_confidence",
    "sentimentconf": "sentiment_confidence",
    "sentconf": "sentiment_confidence",
    "sentiment_bias": "sentiment_bias",
    "sentimentbias": "sentiment_bias",
    "btcdominance": "btc_dominance",
    "feargreed": "fear_greed",
    "exitreason": "exit_reason",
    "llmdecision": "llm_decision",
    "llmdecisionoutput": "llm_decision",
    "llmsignal": "llm_decision",
    "llmapproval": "llm_approval",
    "llmconfidence": "llm_confidence",
    "llmconfidencescore": "llm_confidence",
    "llmconfidence_score": "llm_confidence",
    "llmerror": "llm_error",
    "technicalindicator": "technical_indicator_score",
    "technicalindicatorscore": "technical_indicator_score",
    "technicalscore": "technical_indicator_score",
    "technicalindicators": "technical_indicator_score",
    "volatilitypercent": "volatility",
    "volatilitypct": "volatility",
    "htftrend": "htf_trend",
    "orderimbalance": "order_imbalance",
    "orderflowscore": "order_flow_score",
    "orderflowflag": "order_flow_flag",
    "orderflowstate": "order_flow_state",
    "cvd": "cvd",
    "cvdchange": "cvd_change",
    "cvddivergence": "cvd_divergence",
    "cvdabsorption": "cvd_absorption",
    "cvdaccumulation": "cvd_accumulation",
    "takerbuyratio": "taker_buy_ratio",
    "tradeimbalance": "trade_imbalance",
    "aggressivetraderate": "aggressive_trade_rate",
    "spoofingintensity": "spoofing_intensity",
    "spoofingalert": "spoofing_alert",
    "volumeratio": "volume_ratio",
    "pricechangepct": "price_change_pct",
    "spreadbps": "spread_bps",
    "macroindicator": "macro_indicator",
    "tp": "tp1",
    "takeprofit": "tp1",
    "takeprofittarget": "tp1",
    "tppartial": "tp1_partial",
    "takeprofitpartial": "tp1_partial",
    "tp1partial": "tp1_partial",
    "tp2partial": "tp2_partial",
    "pnltp": "pnl_tp1",
    "takeprofitpnl": "pnl_tp1",
    "tp1pnl": "pnl_tp1",
    "pnltp1": "pnl_tp1",
    "tp2pnl": "pnl_tp2",
    "pnltp2": "pnl_tp2",
    "tp3pnl": "pnl_tp3",
    "pnltp3": "pnl_tp3",
    "sizetp": "size_tp1",
    "takeprofitsize": "size_tp1",
    "sizetp1": "size_tp1",
    "sizetp2": "size_tp2",
    "sizetp3": "size_tp3",
    "notionaltp": "notional_tp1",
    "takeprofitnotional": "notional_tp1",
    "notionaltp1": "notional_tp1",
    "notionaltp2": "notional_tp2",
    "notionaltp3": "notional_tp3",
    "auctionstate": "auction_state",
    "volumeprofilelegtype": "volume_profile_leg_type",
    "volumeprofilepoc": "volume_profile_poc",
    "volumeprofilelvns": "volume_profile_lvns",
    "volumeprofilebinwidth": "volume_profile_bin_width",
    "volumeprofilesnapshot": "volume_profile_snapshot",
    "lvnentrylevel": "lvn_entry_level",
    "lvnstop": "lvn_stop",
    "poctarget": "poc_target",
    "orderflowstatedetail": "orderflow_state_detail",
    "orderflowfeatures": "orderflow_features",
    "orderflowsnapshot": "orderflow_snapshot",
}


def _normalise_token(name: str) -> str:
    """Return a canonical token for a column name."""

    return re.sub(r"[\s_]+", "", str(name).strip().lower())


def _sanitise_default(name: str) -> str:
    """Fallback sanitisation used when no synonym is defined."""

    # Preserve underscores while normalising casing; this mirrors the previous
    # behaviour scattered across multiple modules.
    return str(name).strip().lower().replace(" ", "_")


def canonicalise_column(name: str) -> str:
    """Map ``name`` onto the canonical schema column."""

    token = _normalise_token(name)
    return COLUMN_SYNONYMS.get(token, _sanitise_default(name))


def build_rename_map(columns: Iterable[str]) -> Dict[str, str]:
    """Construct a rename map that resolves aliases without creating duplicates."""

    rename_map: Dict[str, str] = {}
    seen_targets: set[str] = set()
    for col in columns:
        canonical = canonicalise_column(col)
        if canonical in seen_targets and canonical != col:
            # The canonical name is already present (for example both ``entry``
            # and ``entry_price`` exist).  Keep a sanitised version of the
            # original column to avoid silent data loss while still removing
            # whitespace/odd casing.
            sanitised = _sanitise_default(col)
            if sanitised in seen_targets and sanitised != canonical:
                base = sanitised or "column"
                index = 1
                candidate = f"{base}_{index}"
                while candidate in seen_targets:
                    index += 1
                    candidate = f"{base}_{index}"
                rename_map[col] = candidate
                seen_targets.add(candidate)
            else:
                rename_map[col] = sanitised
                seen_targets.add(sanitised)
            continue
        rename_map[col] = canonical
        seen_targets.add(canonical)
    return rename_map


def normalise_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with legacy headers mapped onto canonical schema names."""

    if df.empty:
        return df
    rename_map = build_rename_map(df.columns)
    return df.rename(columns=rename_map)

