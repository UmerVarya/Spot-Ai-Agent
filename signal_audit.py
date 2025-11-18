import csv
import os
from datetime import datetime, timezone
from typing import Any, Dict

SIGNAL_AUDIT_COLUMNS = [
    "ts",
    "symbol",
    "tier",
    "session",
    "auction_state",
    "macro_bias",
    "fear_greed",
    "btc_dominance",
    "news_severity",
    "is_core_symbol",
    "bars_5m_count",
    "hourly_bar_age_min",
    "data_ok",
    "volume_gate_pass",
    "volume_gate_reason",
    "last_quote_vol_1m",
    "avg_quote_vol_20m",
    "session_volume_floor",
    "absolute_volume_floor",
    "profile_min_1m_vol",
    "profile_min_avg20_vol",
    "profile_vol_expansion_min",
    "profile_volume_pass",
    "spread_ratio",
    "spread_gate_fail",
    "orderbook_imbalance",
    "obi_gate_fail",
    "atr_percentile",
    "atr_value",
    "atr_score_contrib",
    "ema_spread_pct",
    "ema_score_contrib",
    "macd_value",
    "macd_slope_tanh",
    "macd_score_contrib",
    "rsi_14",
    "rsi_score_contrib",
    "adx_14",
    "di_plus",
    "di_minus",
    "adx_score_contrib",
    "vwma_slope",
    "vwma_score_contrib",
    "boll_position",
    "boll_score_contrib",
    "dema_diff",
    "dema_score_contrib",
    "stoch_k",
    "stoch_d",
    "stoch_score_contrib",
    "cci",
    "cci_score_contrib",
    "hurst",
    "hurst_score_contrib",
    "multi_tf_confluence_flag",
    "confluence_score_contrib",
    "pattern_candle_flag",
    "pattern_flag_flag",
    "pattern_double_bottom_flag",
    "pattern_cup_handle_flag",
    "pattern_head_shoulders_flag",
    "structure_score",
    "flow_trade_imbalance",
    "flow_ob_imbalance",
    "flow_cvd_delta",
    "flow_taker_buy_ratio",
    "flow_aggression_rate",
    "flow_spoofing_intensity",
    "volume_ratio_5bar",
    "price_change_5bar",
    "flow_score_raw",
    "flow_score_clipped",
    "flow_score_contrib",
    "flow_state",
    "trend_score_raw",
    "mean_rev_score_raw",
    "penalty_score",
    "trend_total",
    "mean_rev_total",
    "trend_norm",
    "mean_rev_norm",
    "setup_type_base",
    "dynamic_threshold",
    "activation_threshold",
    "sentiment_bias",
    "sentiment_score_adjustment",
    "normalized_score_pre_profile",
    "profile_name",
    "profile_min_score",
    "session_multiplier",
    "atr_ratio_15m",
    "atr_profile_min",
    "atr_profile_penalty",
    "profile_veto",
    "score_after_profile",
    "sr_guard_pass",
    "near_resistance",
    "near_support",
    "sr_required_min_score",
    "alt_adjustment",
    "adjusted_score",
    "alt_adj_block",
    "macro_veto",
    "news_veto",
    "macro_skip_all",
    "macro_skip_alt",
    "size_bucket",
    "volume_ok_from_size",
    "base_direction_from_signal",
    "forced_long_applied",
    "final_direction_after_force",
    "cooldown_active",
    "auction_breakout_veto",
    "selected_for_candidate_list",
    "brain_veto",
    "ml_probability",
    "ml_veto",
    "risk_veto",
    "final_trade_taken",
    "final_skip_reason",
]


def _get_audit_dir() -> str:
    base = os.getenv("SIGNAL_AUDIT_DIR", "/home/ubuntu/spot_data/analytics")
    os.makedirs(base, exist_ok=True)
    return base


def _get_audit_path(ts: datetime) -> str:
    day_str = ts.strftime("%Y-%m-%d")
    return os.path.join(_get_audit_dir(), f"signal_audit_{day_str}.csv")


def _ensure_header(path: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(SIGNAL_AUDIT_COLUMNS)


def log_signal_audit(row: Dict[str, Any]) -> None:
    if os.getenv("ENABLE_SIGNAL_AUDIT", "true").lower() not in ("1", "true", "yes"):
        return

    ts_value = row.get("ts")
    if isinstance(ts_value, datetime):
        ts = ts_value.astimezone(timezone.utc)
        row["ts"] = ts.isoformat()
    elif isinstance(ts_value, str):
        try:
            ts = datetime.fromisoformat(ts_value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            ts = datetime.now(timezone.utc)
            row["ts"] = ts.isoformat()
    else:
        ts = datetime.now(timezone.utc)
        row["ts"] = ts.isoformat()

    path = _get_audit_path(ts)
    _ensure_header(path)

    flat = []
    for col in SIGNAL_AUDIT_COLUMNS:
        val = row.get(col, "")
        if isinstance(val, bool):
            val = int(val)
        flat.append(val)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(flat)
