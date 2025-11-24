"""Utility script to run Spot-AI backtests with the live decision stack."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from backtest import Backtester, grid_search
from backtest.data import load_csv_folder

# Import live modules -------------------------------------------------------
import trade_utils as trade_utils_module
from trade_utils import evaluate_signal as evaluate_signal_live
from ml_model import predict_success_probability
from agent import macro_filter_decision
from trade_storage import MAX_CONCURRENT_TRADES
from macro_filter import get_macro_context

# Ensure the signal stack runs in offline/backtest mode.
os.environ.setdefault("TRAINING_MODE", "true")
trade_utils_module.get_market_stream = None  # type: ignore[attr-defined]
trade_utils_module.get_order_book = lambda *args, **kwargs: None  # type: ignore
trade_utils_module._get_binance_client = lambda *args, **kwargs: None  # type: ignore


def _coerce_float(value: object) -> Optional[float]:
    try:
        candidate = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate):
        return None
    return candidate


def _coerce_int(value: object) -> Optional[int]:
    try:
        candidate = int(float(value))
    except (TypeError, ValueError):
        return None
    if 0 <= candidate <= 100:
        return candidate
    return None


def _build_macro_context(snapshot: Mapping[str, object]) -> Dict[str, object | None]:
    """Normalize macro snapshot from :func:`get_macro_context`."""

    fear_greed = _coerce_int(snapshot.get("fear_greed")) if snapshot else None
    btc_dom = _coerce_float(snapshot.get("btc_dominance")) if snapshot else None
    fg_age = _coerce_float(snapshot.get("fear_greed_age_seconds")) if snapshot else None
    btc_age = _coerce_float(snapshot.get("btc_age_seconds")) if snapshot else None
    bias_raw = None
    if snapshot:
        bias_candidate = snapshot.get("macro_sentiment") or snapshot.get("macro_bias")
        if isinstance(bias_candidate, str) and bias_candidate.strip():
            bias_raw = bias_candidate.strip().lower()

    return {
        "fear_greed": fear_greed,
        "fear_greed_age_sec": fg_age,
        "btc_dom": btc_dom,
        "btc_dom_age_sec": btc_age,
        "macro_bias": bias_raw,
    }


def evaluate_signal(df_slice: pd.DataFrame, symbol: str, is_backtest: bool = True):
    """Wrapper around the live signal stack with no look-ahead."""

    macro_snapshot = get_macro_context() or {}
    macro_context = _build_macro_context(macro_snapshot)
    score, direction, confidence, meta = evaluate_signal_live(
        df_slice.copy(), symbol=symbol, macro_context=macro_context, is_backtest=is_backtest
    )
    metadata = meta if isinstance(meta, dict) else {"detail": meta}
    metadata.setdefault("macro_context", macro_context)
    return {
        "score": float(score or 0.0),
        "direction": direction,
        "confidence": float(confidence or 0.0),
        "metadata": metadata,
    }


def predict_prob(signal_dict, symbol: str) -> float:
    """Map signal outputs into the live ML probability model."""

    meta = signal_dict.get("metadata") or {}
    session = meta.get("session", "Backtest")

    def _to_float(value: object) -> float:
        if value is None:
            return float("nan")
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return candidate

    btc_dom = _to_float(meta.get("btc_dominance"))
    fear_greed = _to_float(meta.get("fear_greed"))
    sentiment_conf = float(meta.get("sentiment_confidence", 5.0))
    pattern = str(meta.get("pattern", meta.get("setup_type", "unknown")))
    return float(
        np.clip(
            predict_success_probability(
                float(signal_dict.get("score", 0.0)),
                float(signal_dict.get("confidence", 0.0)),
                session,
                btc_dom,
                fear_greed,
                sentiment_conf,
                pattern,
            ),
            0.0,
            1.0,
        )
    )


def macro_filter() -> bool:
    """Deterministic macro gate derived from the live macro filter."""

    try:
        context = get_macro_context()
        bias = str(context.get("macro_sentiment", "neutral"))
        btc_raw = context.get("btc_dominance")
        fg_raw = context.get("fear_greed")
        btc_dom = float(btc_raw) if btc_raw is not None else None
        try:
            fear_greed = int(float(fg_raw)) if fg_raw is not None else None
        except (TypeError, ValueError):
            fear_greed = None
    except Exception:
        bias = "neutral"
        btc_dom = None
        fear_greed = None
    skip_all, skip_alt, _ = macro_filter_decision(btc_dom, fear_greed, bias, 7.0)
    return not skip_all


def position_size_func(confidence: float) -> float:
    """Live-like position sizing curve bounded between 0.3% and 3%."""

    base = 0.003
    extra = np.clip(confidence, 0.0, 10.0) * 0.0005
    return float(np.clip(base + extra, 0.003, 0.03))


def run_single_backtest(
    data_glob: str = "data/*_1m.csv",
    min_score: float = 0.2,
    min_prob: float = 0.55,
    atr_mult_sl: float = 1.5,
    tp_rungs: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    fee_bps: float = 10.0,
    slippage_bps: float = 2.0,
    latency_bars: int = 0,
    start: str | None = None,
    end: str | None = None,
    max_concurrent: int = MAX_CONCURRENT_TRADES,
) -> None:
    """Run a single backtest and persist results to ``backtests/out``."""

    historical_data = load_csv_folder(data_glob)

    bt = Backtester(
        historical_data=historical_data,
        evaluate_signal=evaluate_signal,
        predict_prob=predict_prob,
        macro_filter=macro_filter,
        position_size_func=position_size_func,
    )

    params = {
        "min_score": min_score,
        "min_prob": min_prob,
        "atr_mult_sl": atr_mult_sl,
        "tp_rungs": tp_rungs,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "latency_bars": latency_bars,
        "max_concurrent": max_concurrent,
        "is_backtest": True,
    }
    if start:
        params["start_ts"] = pd.to_datetime(start, utc=True)
    if end:
        params["end_ts"] = pd.to_datetime(end, utc=True)

    results = bt.run(params)
    performance = results.get("performance", {})
    trades_df = results.get("trades_df")

    out_dir = Path("backtests/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / "historical_trades.csv"
    perf_path = out_dir / "performance.json"

    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
        trades_df.to_csv(trades_path, index=False)
    else:
        trades = results.get("trades", [])
        if trades:
            pd.DataFrame(trades).to_csv(trades_path, index=False)

    with open(perf_path, "w", encoding="utf-8") as fh:
        json.dump(performance, fh, indent=2, default=str)

    print("\n=== Backtest Complete ===")
    print(json.dumps(performance, indent=2, default=str))
    if trades_path.exists():
        print(f"Saved trades → {trades_path}")
    else:
        print("No trades generated.")


def run_grid(
    data_glob: str = "data/*_1m.csv",
    param_grid: Dict[str, list] | None = None,
) -> None:
    """Execute a simple grid search over backtest parameters."""

    if param_grid is None:
        param_grid = {
            "min_score": [0.15, 0.2, 0.25],
            "min_prob": [0.55, 0.6, 0.65],
            "atr_mult_sl": [1.2, 1.5, 1.8],
            "tp_rungs": [(1, 2, 3, 4), (0.8, 1.6, 2.4, 3.2)],
            "fee_bps": [8, 10, 12],
            "slippage_bps": [2, 5],
            "latency_bars": [0, 1],
            "max_concurrent": [MAX_CONCURRENT_TRADES],
        }

    bt = Backtester(
        historical_data=load_csv_folder(data_glob),
        evaluate_signal=evaluate_signal,
        predict_prob=predict_prob,
        macro_filter=macro_filter,
        position_size_func=position_size_func,
    )

    results = grid_search(bt, param_grid)
    rows = [
        {**res.get("params", {}), **(res.get("performance") or {})}
        for res in results
    ]

    df = pd.DataFrame(rows)
    if not df.empty and "sharpe" in df.columns:
        df.sort_values("sharpe", ascending=False, inplace=True)
    out_dir = Path("backtests/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / "grid_results.csv"
    if not df.empty:
        df.to_csv(grid_path, index=False)
        print(f"Saved {grid_path}")
        print(df.head(5).to_string(index=False))
    else:
        print("No grid search results to save.")


if __name__ == "__main__":
    run_single_backtest()
    # To run a grid search, uncomment the following line:
    # run_grid()
