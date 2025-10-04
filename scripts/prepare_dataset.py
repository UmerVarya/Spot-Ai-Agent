#!/usr/bin/env python3
import os, json, math
import pandas as pd
import numpy as np
from pathlib import Path

# --- Paths ---
DEFAULT_HISTORY = os.getenv(
    "COMPLETED_TRADES_FILE", "/home/ubuntu/spot_data/trades/historical_trades.csv"
)
OUT_DIR = Path(os.getenv("DATASET_DIR", "/home/ubuntu/spot_data/datasets"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Header normalization map (add any variants you've used before) ---
HEADER_MAP = {
    # identifiers & timing
    "TradeID": "trade_id", "trade_id": "trade_id",
    "Symbol": "symbol", "SymbolPair": "symbol",
    "EntryTime": "entry_time", "exit_time": "exit_time",

    # prices & sizes
    "EntryPrice": "entry_price", "ExitPrice": "exit_price",
    "entry": "entry_price", "exit": "exit_price",
    "Size": "size_quote", "SizeQuote": "size_quote", "QuoteSize": "size_quote",
    "Qty": "size_base", "BaseSize": "size_base",

    # economics
    "Fees": "fees", "fees": "fees",
    "Slippage": "slippage", "slippage": "slippage",
    "PNL": "pnl_quote", "pnl": "pnl_quote", "PnL": "pnl_quote",
    "PNL_USDT": "pnl_quote", "pnl_pct": "pnl_pct", "PnL_pct": "pnl_pct",

    # meta / features
    "Side": "side", "direction": "side",
    "Score": "score", "ModelScore": "score",
    "Session": "session",
    "Sentiment": "sentiment_score", "SentimentScore": "sentiment_score",
    "sentiment_bias": "sentiment_score", "sentiment_confidence": "sentiment_confidence",
    "Volatility": "volatility", "volatility": "volatility",
    "order_imbalance": "order_imbalance", "htf_trend": "htf_trend",
    "order_flow_score": "order_flow_score",
    "order_flow_flag": "order_flow_flag",
    "order_flow_state": "order_flow_state",
    "cvd": "cvd",
    "cvd_change": "cvd_change",
    "taker_buy_ratio": "taker_buy_ratio",
    "trade_imbalance": "trade_imbalance",
    "aggressive_trade_rate": "aggressive_trade_rate",
    "spoofing_intensity": "spoofing_intensity",
    "spoofing_alert": "spoofing_alert",
    "volume_ratio": "volume_ratio",
    "price_change_pct": "price_change_pct",
    "spread_bps": "spread_bps",
    "macro_indicator": "macro_indicator", "btc_dominance": "btc_dominance",
    "fear_greed": "fear_greed",
    "pattern": "pattern", "narrative": "narrative",
    "llm_decision": "llm_decision", "llm_confidence": "llm_confidence", "llm_error": "llm_error",
    "llm_approval": "llm_approval", "LLMApproval": "llm_approval",
    "technical_indicator_score": "technical_indicator_score", "TechnicalIndicatorScore": "technical_indicator_score",
    "exit_reason": "exit_reason", "ExitReason": "exit_reason",

    # labels
    "Trade_Outcome": "outcome", "Outcome": "outcome", "outcome": "outcome",
    "outcome_desc": "outcome_desc"
}

REQUIRED_COLS_MIN = {"symbol", "entry_time", "exit_time", "entry_price", "exit_price"}
POSSIBLE_FEATURES = [
    "score","technical_indicator_score","rsi","ema_diff","bb_width","volatility","volume_z",
    "sentiment_score","session","spread_bps","slippage_bps",
    "order_imbalance","order_flow_score","order_flow_flag","cvd","cvd_change","taker_buy_ratio",
    "trade_imbalance","aggressive_trade_rate","spoofing_intensity","spoofing_alert",
    "volume_ratio","price_change_pct"
]

def load_history_df():
    # Prefer your repo loader if available
    try:
        # If this import fails, we’ll fall back to CSV
        from trade_storage import load_trade_history_df
        df = load_trade_history_df()
    except Exception:
        df = pd.read_csv(DEFAULT_HISTORY)
    return df

def normalize_headers(df):
    # Lowercase current names first
    df = df.rename(columns={c: c for c in df.columns})
    # Apply explicit mapping (case-sensitive keys from HEADER_MAP)
    ren = {k: v for k, v in HEADER_MAP.items() if k in df.columns}
    df = df.rename(columns=ren)
    # Lowercase final colnames for safety
    df.columns = [c.lower() for c in df.columns]
    return df

def coerce_types(df):
    for col in ["entry_time","exit_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    for col in [
        "entry_price","exit_price","size_quote","size_base","fees","pnl_quote","pnl_pct","score",
        "rsi","ema_diff","bb_width","volatility","volume_z","spread_bps","slippage_bps","sentiment_score",
        "order_imbalance","order_flow_score","order_flow_flag","cvd","cvd_change","taker_buy_ratio","trade_imbalance",
        "aggressive_trade_rate","spoofing_intensity","spoofing_alert","volume_ratio","price_change_pct"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.upper()
    return df

def derive_missing(df):
    # Duration
    if {"entry_time","exit_time"}.issubset(df.columns):
        df["holding_minutes"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds()/60.0

    # Compute pnl_quote if missing (long-only default; if you use shorts, add logic)
    if "pnl_quote" not in df.columns and {"entry_price","exit_price","size_quote"}.issubset(df.columns):
        # gross PnL in quote (assumes buy at entry, sell at exit)
        df["pnl_quote"] = (df["exit_price"] - df["entry_price"]) * (df["size_quote"] / df["entry_price"])

    # Fees default 0
    if "fees" not in df.columns:
        df["fees"] = 0.0

    # Net pnl & pct
    if "pnl_quote" in df.columns:
        df["pnl_net_quote"] = df["pnl_quote"] - df["fees"]
        if "size_quote" in df.columns:
            df["pnl_pct"] = df["pnl_net_quote"] / df["size_quote"]

    # Binary outcome (label) based on pnl instead of outcome text
    if "pnl" in df.columns:
        df["pnl_quote"] = pd.to_numeric(df["pnl"], errors="coerce")
    if "pnl_pct" in df.columns:
        df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")

    if "pnl_quote" in df.columns:
        df["outcome"] = (df["pnl_quote"] > 0).astype(int)
    elif "pnl_pct" in df.columns:
        df["outcome"] = (df["pnl_pct"] > 0).astype(int)

    return df

def basic_clean(df):
    # Keep only completed trades with both timestamps
    if {"entry_time","exit_time"}.issubset(df.columns):
        df = df[df["entry_time"].notna() & df["exit_time"].notna()]
    # Remove impossible prices/sizes
    for col in ["entry_price","exit_price","size_quote"]:
        if col in df.columns:
            df = df[df[col] > 0]
    # Drop exact duplicates
    subset = [c for c in ["trade_id","symbol","entry_time","exit_time"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="last")
    return df

def chronological_split(df, train_frac=0.70, val_frac=0.15):
    df = df.sort_values("exit_time")
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train]
    val   = df.iloc[n_train:n_train+n_val]
    test  = df.iloc[n_train+n_val:]
    return train, val, test

def main():
    print(f"Loading history from {DEFAULT_HISTORY} (or repo loader if available)")
    df = load_history_df()
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    df = normalize_headers(df)
    df = coerce_types(df)
    df = derive_missing(df)
    df = basic_clean(df)

    missing = REQUIRED_COLS_MIN - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns after normalization: {missing}")

    # Select columns to keep
    base_cols = ["trade_id","symbol","side","entry_time","exit_time","entry_price","exit_price",
                 "size_quote","fees","holding_minutes"]
    base_cols = [c for c in base_cols if c in df.columns]

    feat_cols = [c for c in POSSIBLE_FEATURES if c in df.columns]
    target_cols = [c for c in ["outcome","pnl_pct","pnl_quote","pnl_net_quote"] if c in df.columns]

    keep_cols = list(dict.fromkeys(base_cols + feat_cols + target_cols))  # preserve order, dedupe
    df = df[keep_cols].copy()

    # Quick stats (class imbalance)
    if "outcome" in df.columns:
        win_rate = float(df["outcome"].mean())*100 if len(df)>0 else float("nan")
        print(f"Win rate: {win_rate:.2f}%  (n={len(df)})")

    # Split chronologically (NO shuffling to avoid leakage)
    train, val, test = chronological_split(df)

    # Save splits (CSV is fine; Parquet is also good if you prefer)
    train.to_csv(OUT_DIR/"train.csv", index=False)
    val.to_csv(OUT_DIR/"val.csv", index=False)
    test.to_csv(OUT_DIR/"test.csv", index=False)

    # Optional Parquet
    train.to_parquet(OUT_DIR/"train.parquet", index=False)
    val.to_parquet(OUT_DIR/"val.parquet", index=False)
    test.to_parquet(OUT_DIR/"test.parquet", index=False)

    print("Saved:")
    print(OUT_DIR/"train.csv")
    print(OUT_DIR/"val.csv")
    print(OUT_DIR/"test.csv")

    # Summary
    for name, part in [("train", train), ("val", val), ("test", test)]:
        if "outcome" in part.columns and len(part):
            print(f"{name}: n={len(part):>5}  win%={part['outcome'].mean()*100:5.2f}")
        else:
            print(f"{name}: n={len(part):>5}")

if __name__ == "__main__":
    main()
