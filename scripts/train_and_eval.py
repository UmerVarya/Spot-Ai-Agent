#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, roc_curve
)

DATASET_DIR = Path("/home/ubuntu/spot_data/datasets")
PROC_DIR    = Path("/home/ubuntu/spot_data/processed")
OUT_DIR     = Path("/home/ubuntu/spot_data/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_proc():
    X_train = np.load(PROC_DIR/"X_train.npy")
    X_val   = np.load(PROC_DIR/"X_val.npy")
    X_test  = np.load(PROC_DIR/"X_test.npy")
    y_train = np.load(PROC_DIR/"y_train.npy")
    y_val   = np.load(PROC_DIR/"y_val.npy")
    y_test  = np.load(PROC_DIR/"y_test.npy")
    with open(PROC_DIR/"feature_names.json") as f:
        feature_names = json.load(f)
    return (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)

def load_raw_splits():
    val_df  = pd.read_csv(DATASET_DIR/"val.csv")
    test_df = pd.read_csv(DATASET_DIR/"test.csv")
    # lowercase columns for safety
    val_df.columns  = [c.lower() for c in val_df.columns]
    test_df.columns = [c.lower() for c in test_df.columns]
    return val_df, test_df

# ------- Utility metrics for trading -------
def pick_pnl_column(df: pd.DataFrame) -> str:
    for cand in ["pnl", "pnl_quote", "pnl_net_quote"]:
        if cand in df.columns:
            return cand
    # derive from pct * notional if needed
    if "pnl_pct" in df.columns and "notional" in df.columns:
        df["pnl"] = df["pnl_pct"] * df["notional"]
        return "pnl"
    # fallback zero (shouldn't happen ideally)
    df["pnl"] = 0.0
    return "pnl"

def equity_max_drawdown(pnl_series: pd.Series):
    equity = pnl_series.cumsum()
    roll_max = equity.cummax()
    dd = equity - roll_max
    max_dd = dd.min() if len(dd) else 0.0
    return equity, float(max_dd)

def sharpe_ratio(returns: pd.Series):
    r = returns.dropna().astype(float)
    if len(r) < 2 or r.std() == 0:
        return 0.0
    # per-trade Sharpe
    return float(np.sqrt(len(r)) * r.mean() / r.std())

def sweep_threshold_for_val_pnl(y_true, prob, pnl, grid=None):
    if grid is None:
        grid = np.linspace(0.1, 0.9, 81)  # 0.01 steps
    best = {"thr":0.5, "pnl":-1e18, "f1":0.0}
    for t in grid:
        take = prob >= t
        pnl_sum = float(pnl[take].sum()) if take.any() else 0.0
        f1 = f1_score(y_true, (prob>=t).astype(int), zero_division=0)
        if pnl_sum > best["pnl"]:
            best = {"thr":float(t), "pnl":pnl_sum, "f1":float(f1)}
    return best

def summarize_selected(df, take_mask, pnl_col):
    sub = df.loc[take_mask].copy()
    n = len(sub)
    wins = int((sub[pnl_col] > 0).sum())
    losses = n - wins
    winp = (wins/n*100) if n else 0.0
    pnl_sum = float(sub[pnl_col].sum())
    sr = sharpe_ratio(sub.get("pnl_pct", pd.Series(dtype=float)))
    _, mdd = equity_max_drawdown(sub[pnl_col])
    return {
        "n_trades": int(n),
        "win_rate_pct": float(winp),
        "total_pnl": pnl_sum,
        "avg_pnl": float(pnl_sum / n) if n else 0.0,
        "sharpe_per_trade": sr,
        "max_drawdown": float(mdd),
    }, sub

def main():
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_proc()
    val_raw, test_raw = load_raw_splits()
    pnl_col_val  = pick_pnl_column(val_raw)
    pnl_col_test = pick_pnl_column(test_raw)

    # ------- Train baseline Logistic Regression -------
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
        solver="lbfgs"
    )
    clf.fit(X_train, y_train)

    # ------- Choose threshold on validation by PnL -------
    val_prob = clf.predict_proba(X_val)[:,1]
    best = sweep_threshold_for_val_pnl(y_val, val_prob, val_raw[pnl_col_val].astype(float))
    thr = best["thr"]

    # Also compute ML metrics at that threshold for val:
    val_pred = (val_prob >= thr).astype(int)
    val_auc = roc_auc_score(y_val, val_prob)
    val_f1  = f1_score(y_val, val_pred, zero_division=0)
    val_pr  = precision_score(y_val, val_pred, zero_division=0)
    val_rc  = recall_score(y_val, val_pred, zero_division=0)

    # ------- Evaluate on TEST -------
    test_prob = clf.predict_proba(X_test)[:,1]
    test_pred = (test_prob >= thr).astype(int)

    test_auc = roc_auc_score(y_test, test_prob)
    test_f1  = f1_score(y_test, test_pred, zero_division=0)
    test_pr  = precision_score(y_test, test_pred, zero_division=0)
    test_rc  = recall_score(y_test, test_pred, zero_division=0)

    # Trading metrics on TEST (selected trades only)
    take_mask = test_pred.astype(bool)
    trade_stats, taken = summarize_selected(test_raw, take_mask, pnl_col_test)

    # Baseline: trade everything on TEST
    base_stats, base_taken = summarize_selected(test_raw, np.ones(len(test_raw), dtype=bool), pnl_col_test)

    # Equity curve for selected trades
    equity, mdd = equity_max_drawdown(taken[pnl_col_test].astype(float))
    taken = taken.assign(equity_curve=equity.values)

    # ------- Save artifacts -------
    preds = test_raw.copy()
    preds["prob_win"] = test_prob
    preds["pred_label"] = test_pred
    preds.to_csv(OUT_DIR/"predictions_test.csv", index=False)
    taken.to_csv(OUT_DIR/"equity_curve_selected_test.csv", index=False)

    report = {
        "threshold_selected_on_val": thr,
        "val_metrics": {"AUC": float(val_auc), "F1": float(val_f1), "Precision": float(val_pr), "Recall": float(val_rc),
                        "val_best_pnl": float(best["pnl"])},
        "test_metrics": {"AUC": float(test_auc), "F1": float(test_f1), "Precision": float(test_pr), "Recall": float(test_rc)},
        "test_trading_selected": trade_stats,
        "test_trading_baseline_trade_all": base_stats,
        "feature_count_after_encoding": int(X_train.shape[1]),
    }
    with open(OUT_DIR/"report.json","w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Threshold (chosen on VAL by max PnL) ===")
    print(f"threshold = {thr:.3f}")
    print("\nVAL  -> AUC={val_auc:.3f}  F1={val_f1:.3f}  P={val_pr:.3f}  R={val_rc:.3f}  | best_val_PnL={best_pnl:.2f}"\n          .format(val_auc=val_auc, val_f1=val_f1, val_pr=val_pr, val_rc=val_rc, best_pnl=best['pnl']))
    print("TEST -> AUC={:.3f}  F1={:.3f}  P={:.3f}  R={:.3f}".format(test_auc, test_f1, test_pr, test_rc))

    print("\n--- Trading results on TEST (selected trades only) ---")
    for k, v in trade_stats.items():
        print(f"{k:>22s}: {v}")
    print("\n--- Baseline on TEST (trade all) ---")
    for k, v in base_stats.items():
        print(f"{k:>22s}: {v}")

    print("\nSaved:")
    print(" -", OUT_DIR/"report.json")
    print(" -", OUT_DIR/"predictions_test.csv")
    print(" -", OUT_DIR/"equity_curve_selected_test.csv")

if __name__ == "__main__":
    main()
