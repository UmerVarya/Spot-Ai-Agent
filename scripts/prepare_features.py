#!/usr/bin/env python3
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
import joblib

DATASET_DIR = Path("/home/ubuntu/spot_data/datasets")
OUT_DIR     = Path("/home/ubuntu/spot_data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns we must NOT use as features (leakage or non-predictive IDs)
BAN_COLS = {
    "pnl", "pnl_quote", "pnl_net_quote", "pnl_pct",
    "outcome", "outcome_desc",
    "entry_time", "exit_time", "timestamp",
    "trade_id", "narrative", "llm_error"
}

# Candidate numeric/categorical features from your file. Add/remove as needed.
CANDIDATES_NUM = [
    "score", "confidence",
    "btc_dominance", "fear_greed",
    "sentiment_bias", "sentiment_confidence",
    "volatility", "htf_trend",
    "order_imbalance", "macro_indicator",
    "fees", "slippage",
    "size", "notional"   # keep if you want position-size info
]

CANDIDATES_CAT = [
    "symbol", "direction", "session", "strategy", "pattern", "llm_decision"
]

def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATASET_DIR / f"{name}.csv")
    # keep lowercase columns for safety
    df.columns = [c.lower() for c in df.columns]
    return df

def pick_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if c not in BAN_COLS]
    # intersect with candidates to be explicit
    num_cols = [c for c in CANDIDATES_NUM if c in df.columns]
    cat_cols = [c for c in CANDIDATES_CAT if c in df.columns]
    return num_cols, cat_cols

def make_xy(df: pd.DataFrame, num_cols, cat_cols):
    y = df["outcome"].astype(int).to_numpy()
    X = df[num_cols + cat_cols].copy()
    return X, y

def main():
    train = load_split("train")
    val   = load_split("val")
    test  = load_split("test")

    # verify target exists
    if "outcome" not in train.columns:
        raise SystemExit("Target column 'outcome' missing. Run prepare_dataset.py first.")

    num_cols, cat_cols = pick_columns(train)
    print("Using numeric features:", num_cols)
    print("Using categorical features:", cat_cols)

    # Build preprocessing: impute numerics with median; categoricals with most_frequent + one-hot
    numeric_tf = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
    ])
    categorical_tf = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Fit preprocessing on TRAIN only (no leakage)
    X_train_raw, y_train = make_xy(train, num_cols, cat_cols)
    X_val_raw,   y_val   = make_xy(val,   num_cols, cat_cols)
    X_test_raw,  y_test  = make_xy(test,  num_cols, cat_cols)

    pre.fit(X_train_raw)

    X_train = pre.transform(X_train_raw)
    X_val   = pre.transform(X_val_raw)
    X_test  = pre.transform(X_test_raw)

    # Recover final feature names after one-hot
    feature_names = pre.get_feature_names_out().tolist()

    # Class weights suggestion (for imbalanced classes)
    classes = np.array([0,1])
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(k): float(v) for k, v in zip(classes, cw)}

    # Basic stats
    def stats(y, name):
        n = len(y); pos = int(y.sum()); neg = n - pos
        wp = (pos / n * 100) if n else 0.0
        print(f"{name:5s} -> n={n:4d} | wins={pos:4d} | losses={neg:4d} | win%={wp:5.2f}")

    stats(y_train, "train")
    stats(y_val,   "val")
    stats(y_test,  "test")
    print("Suggested class_weight:", class_weight)

    # Save artifacts
    joblib.dump(pre, OUT_DIR/"preprocessor.joblib")
    np.save(OUT_DIR/"X_train.npy", X_train)
    np.save(OUT_DIR/"X_val.npy",   X_val)
    np.save(OUT_DIR/"X_test.npy",  X_test)
    np.save(OUT_DIR/"y_train.npy", y_train)
    np.save(OUT_DIR/"y_val.npy",   y_val)
    np.save(OUT_DIR/"y_test.npy",  y_test)

    with open(OUT_DIR/"feature_names.json","w") as f:
        json.dump(feature_names, f, indent=2)

    print("\nSaved to:", OUT_DIR)
    print("  preprocessor.joblib")
    print("  X_*.npy, y_*.npy")
    print("  feature_names.json")

if __name__ == "__main__":
    main()
