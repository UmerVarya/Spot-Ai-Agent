#!/usr/bin/env python3
"""
Warm-up helper for RealTimeSignalCache.

- Forces REST path
- Kicks concurrent refreshes for a few symbols
- Logs clearly so you can see success/failure per symbol
- Falls back to raw REST mirror fetch if RealTimeSignalCache cannot be imported

Usage:
  python rtsc_warmup_hotfix.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --limit 2 --timeout 8
"""

import os
import sys
import argparse
import asyncio
import logging
from typing import List, Optional

# ---- Logging ---------------------------------------------------------------
root = logging.getLogger()
if not root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
else:
    root.setLevel(logging.INFO)

log = logging.getLogger("rtsc_hotfix")
log.propagate = True

# ---- Force REST in this process -------------------------------------------
os.environ["RTSC_FORCE_REST"] = "1"

# ---- Optional: try to import your cache class ------------------------------
RTSC = None
try:
    import importlib
    rtsc_mod = importlib.import_module("realtime_signal_cache")
    RTSC = getattr(rtsc_mod, "RealTimeSignalCache", None)
    if RTSC is None:
        log.warning("[HOTFIX] Found realtime_signal_cache.py but no RealTimeSignalCache class.")
    else:
        log.info("[HOTFIX] RealTimeSignalCache imported successfully.")
except Exception as e:
    log.warning(f"[HOTFIX] Could not import RealTimeSignalCache: {e}")

# ---- Lightweight direct REST mirror client (fallback) ----------------------
import requests
import pandas as pd

BINANCE_MIRRORS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://data-api.binance.vision",
]

async def fetch_klines_any(symbol: str, interval: str, limit: int, timeout: float) -> Optional[pd.DataFrame]:
    """
    Mirror-only fetch (no proxies), returns a small DataFrame or None.
    """
    loop = asyncio.get_running_loop()
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    for base in BINANCE_MIRRORS:
        url = f"{base}/api/v3/klines"
        try:
            log.info(f"[HOTFIX] mirror try {url} for {symbol}")
            r = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: requests.get(url, params=params, timeout=timeout, proxies={})),
                timeout=timeout + 1,
            )
            r.raise_for_status()
            raw = r.json()
            if not raw:
                log.warning(f"[HOTFIX] mirror EMPTY {base} for {symbol}")
                continue

            # shape into DF
            cols = [
                "open_time","open","high","low","close","volume",
                "close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
            ]
            try:
                df = pd.DataFrame(raw, columns=cols)
            except Exception:
                df = pd.DataFrame(raw)
                df.columns = cols[:len(df.columns)]
            for c in ("open","high","low","close","volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["close_dt"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            df = df.set_index("close_dt").sort_index()
            df = df[["open","high","low","close","volume"]]

            if df.empty:
                log.warning(f"[HOTFIX] mirror EMPTY DF {base} for {symbol}")
                continue

            log.info(f"[HOTFIX] mirror OK {base} for {symbol} (n={len(df)}) [{df.index[0]} -> {df.index[-1]}]")
            return df
        except asyncio.TimeoutError:
            log.warning(f"[HOTFIX] mirror TIMEOUT {base} for {symbol} after {timeout}s")
        except Exception as e:
            log.warning(f"[HOTFIX] mirror ERROR {base} for {symbol}: {e}")
    return None

def quick_score(df: pd.DataFrame) -> float:
    try:
        last = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2]) if len(df) > 1 else last
        return 0.6 if last > prev else 0.4 if last < prev else 0.5
    except Exception:
        return 0.5

# ---- Main warm-up routines --------------------------------------------------
async def warmup_via_rtsc(symbols: List[str], interval: str, limit: int, timeout: float):
    """
    Use your RealTimeSignalCache directly (best case).
    """
    if RTSC is None:
        log.error("[HOTFIX] RealTimeSignalCache not importable in this process.")
        return

    cache = RTSC()
    sem = asyncio.Semaphore(5)

    async def one(sym: str):
        async with sem:
            try:
                log.info(f"[HOTFIX] ENTER _refresh_symbol_via_rest({sym})")
                # many implementations name it exactly like this
                method = getattr(cache, "_refresh_symbol_via_rest", None)
                if method is None:
                    raise RuntimeError("Cache has no _refresh_symbol_via_rest method.")
                await asyncio.wait_for(method(sym), timeout=timeout + 2)
                data = cache.get(sym)
                ok = data is not None
                detail = f"score={data.score}" if ok else "no payload"
                log.info(f"[HOTFIX] cache update for {sym}: {ok} | {detail}")
            except asyncio.TimeoutError:
                log.warning(f"[HOTFIX] TIMEOUT _refresh_symbol_via_rest({sym}) after {timeout+2:.1f}s")
            except Exception as e:
                log.warning(f"[HOTFIX] FAIL _refresh_symbol_via_rest({sym}): {e}")

    await asyncio.gather(*(one(s) for s in symbols))

async def warmup_direct(symbols: List[str], interval: str, limit: int, timeout: float):
    """
    If we cannot reach your cache, just fetch mirrors to prove data flow.
    """
    sem = asyncio.Semaphore(5)

    async def one(sym: str):
        async with sem:
            df = await fetch_klines_any(sym, interval=interval, limit=limit, timeout=timeout)
            if df is None or df.empty:
                log.warning(f"[HOTFIX] {sym} failed to fetch")
                return
            score = quick_score(df)
            log.info(f"[HOTFIX] {sym} score={score:.2f} last={df['close'].iloc[-1]}")

    await asyncio.gather(*(one(s) for s in symbols))

# ---- CLI --------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="RTSC warm-up hotfix")
    p.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT",
                   help="Comma-separated symbols to warm up")
    p.add_argument("--interval", type=str, default=os.environ.get("RTSC_REST_INTERVAL", "1m"),
                   help="Kline interval (default 1m)")
    p.add_argument("--limit", type=int, default=int(os.environ.get("RTSC_REST_LIMIT", "2")),
                   help="How many recent bars to fetch (default 2)")
    p.add_argument("--timeout", type=float, default=float(os.environ.get("RTSC_REST_TIMEOUT", "8")),
                   help="Per-request timeout seconds (default 8)")
    return p.parse_args()

async def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    log.info(f"[HOTFIX] Starting warm-up for {symbols} interval={args.interval} limit={args.limit} timeout={args.timeout}s")
    if RTSC is not None:
        await warmup_via_rtsc(symbols, args.interval, args.limit, args.timeout)
    else:
        await warmup_direct(symbols, args.interval, args.limit, args.timeout)
    log.info("[HOTFIX] Done.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("[HOTFIX] Interrupted by user")
