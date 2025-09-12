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
    # Prefer NET first
    if "pnl_net_quote" in df.columns:
        return "pnl_net_quote"
    if "pnl" in df.columns:  # if your repo’s `pnl` is already net
        return "pnl"
    # Derive NET if possible
    if "pnl_pct" in df.columns and "notional" in df.columns:
        df["pnl_net_quote"] = pd.to_numeric(df["pnl_pct"], errors="coerce") * pd.to_numeric(df["notional"], errors="coerce")
        return "pnl_net_quote"
    # Fallback: gross
    if "pnl_quote" in df.columns:
        return "pnl_quote"
    # Last resort
    df["pnl_net_quote"] = 0.0
    return "pnl_net_quote"

def equity_max_drawdown(pnl_series: pd.Series):
    """Return cumulative equity curve and max drawdown from PnL series.

    The input series is coerced to numeric once to avoid drift between
    calculations that rely on different representations of the same data.
    """
    pnl = pd.to_numeric(pnl_series, errors="coerce").astype(float)
    equity = pnl.cumsum()
    roll_max = equity.cummax()
    dd = equity - roll_max
    return equity, float(dd.min() if len(dd) else 0.0)

def sharpe_ratio(returns: pd.Series):
    r = returns.dropna().astype(float)
    if len(r) < 2 or r.std() == 0:
        return 0.0
    # per-trade Sharpe
    return float(np.sqrt(len(r)) * r.mean() / r.std())

def sweep_threshold_for_val_pnl(y_true, prob, pnl, grid=None, min_take=10, min_frac=0.12):
    """
    Pick threshold that maximizes PnL on VAL, but only among thresholds
    that select at least `min_take` trades (or >= min_frac of VAL).
    Fallback: use quantile threshold targeting min_frac if no candidate qualifies.
    """
    import numpy as np
    if grid is None:
        grid = np.linspace(0.1, 0.9, 81)

    n = len(prob)
    min_take_abs = max(min_take, int(np.ceil(min_frac * n)))

    best = {"thr": 0.5, "pnl": -1e18, "f1": 0.0, "takes": 0}
    for t in grid:
        take = prob >= t
        k = int(take.sum())
        if k < min_take_abs:
            continue
        pnl_sum = float(pnl[take].sum()) if k else 0.0
        # we keep f1 just for tie-breaking / inspection
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, (prob >= t).astype(int), zero_division=0)
        if pnl_sum > best["pnl"]:
            best = {"thr": float(t), "pnl": pnl_sum, "f1": float(f1), "takes": k}

    # Fallback if nothing met the min-take constraint
    if best["pnl"] == -1e18:
        q = 1.0 - min_frac     # e.g., keep top 12% by prob
        thr = float(np.quantile(prob, q))
        take = prob >= thr
        return {"thr": thr, "pnl": float(pnl[take].sum()) if take.any() else 0.0,
                "f1": 0.0, "takes": int(take.sum()), "fallback": True}
    return best

def summarize_selected(df, take_mask, pnl_col):
    """Summarize performance metrics for selected trades."""
    sub = df.loc[take_mask].copy()
    pnl = pd.to_numeric(sub[pnl_col], errors="coerce").astype(float)

    n = int(pnl.count())
    total = float(pnl.sum())
    avg = float(total / n) if n else 0.0
    wins = int((pnl > 0).sum())
    losses = n - wins
    winp = float(wins / n * 100) if n else 0.0

    # Sharpe on per-trade returns if available; else on pnl (scaled)
    returns = pd.to_numeric(sub.get("pnl_pct", pnl), errors="coerce").astype(float)

    def sharpe_ratio(r):
        r = r.dropna().astype(float)
        return 0.0 if len(r) < 2 or r.std() == 0 else float(np.sqrt(len(r)) * r.mean() / r.std())

    sr = sharpe_ratio(returns)
    _, mdd = equity_max_drawdown(pnl)

    # Safety print to console
    print(
        f"[DEBUG] n={n} total={total:.6f} avg={avg:.6f} (should equal total/n={total/n if n else 0:.6f}) using {pnl_col}"
    )

    return {
        "n_trades": n,
        "win_rate_pct": winp,
        "total_pnl": total,
        "avg_pnl": avg,
        "sharpe_per_trade": sr,
        "max_drawdown": mdd,
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

    val_take_cnt  = int((val_prob  >= thr).sum())
    test_take_cnt = int((test_prob >= thr).sum())
    print(f"[INFO] threshold={thr:.3f} | VAL takes={val_take_cnt} | TEST takes={test_take_cnt}")

    # Trading metrics on TEST (selected trades only)
    take_mask = test_pred.astype(bool)
    trade_stats, _ = summarize_selected(test_raw, take_mask, pnl_col_test)

    # Baseline: trade everything on TEST
    base_stats, _ = summarize_selected(test_raw, np.ones(len(test_raw), dtype=bool), pnl_col_test)

    # ------- Save artifacts -------
    # Standardize PnL column name for outputs
    pnl_col_test = pick_pnl_column(test_raw)
    preds = test_raw.copy()
    preds["prob_win"] = test_prob
    preds["pred_label"] = test_pred
    preds["pnl_standard"] = pd.to_numeric(preds[pnl_col_test], errors="coerce")
    preds.to_csv(OUT_DIR/"predictions_test.csv", index=False)

    taken = test_raw.loc[take_mask].copy()
    taken["pnl_standard"] = pd.to_numeric(taken[pnl_col_test], errors="coerce")
    equity, mdd = equity_max_drawdown(taken["pnl_standard"])
    taken = taken.assign(equity_curve=equity.values)
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
    print(
        f"\nVAL  -> AUC={val_auc:.3f}  F1={val_f1:.3f}  P={val_pr:.3f}  R={val_rc:.3f}  | best_val_PnL={best['pnl']:.2f}"
    )
    print(f"TEST -> AUC={test_auc:.3f}  F1={test_f1:.3f}  P={test_pr:.3f}  R={test_rc:.3f}")

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
