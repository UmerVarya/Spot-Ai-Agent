#!/usr/bin/env python3
"""
Extended RealTimeSignalCache diagnostic tool.
Shows symbol freshness, live scores, and open trade info.
"""

import os, json
from datetime import datetime

# tolerant imports (repo-root modules)
try:
    from realtime_signal_cache import RealTimeSignalCache
except Exception:
    RealTimeSignalCache = None

try:
    from trade_storage import TradeStorage
except Exception:
    TradeStorage = None


def load_rtsc():
    # Preferred: use class if available
    if RealTimeSignalCache is not None and hasattr(RealTimeSignalCache, "load"):
        return RealTimeSignalCache.load()

    # Fallback: read the diagnostics snapshot (rtsc_diagnostics.json)
    import json, os
    path = os.getenv("RTSC_DIAG_PATH", "rtsc_diagnostics.json")
    with open(path, "r") as f:
        data = json.load(f)

    # emulate a minimal object with .cache
    class _Obj:
        pass

    o = _Obj()
    # Accept either {"symbols": {...}} or {"cache": {...}}
    o.cache = data.get("symbols") or data.get("cache") or {}
    return o


def load_open_trades():
    if TradeStorage is not None and hasattr(TradeStorage, "load"):
        try:
            store = TradeStorage.load()
            return {t["symbol"]: t for t in store.open_trades.values()}
        except Exception:
            return {}
    return {}

def main():
    print(f"\n🧠  RealTimeSignalCache Status ({datetime.utcnow():%H:%M:%S UTC})\n")

    # Load RTSC
    rtsc = load_rtsc()
    open_trades = load_open_trades()
    cache = getattr(rtsc, "cache", {})
    print(f"Symbols tracked: {len(cache)}")

    header = f"{'SYMBOL':10} | {'SCORE':6} | {'CONF':5} | {'DIR':6} | {'STATE':8} | {'AGE':>4}s | {'TRADE?'}"
    print(header)
    print("-" * len(header))

    for sym, info in sorted(cache.items()):
        score = info.get("score", 0)
        conf = info.get("confidence", 0)
        direction = info.get("direction", "-")
        age = round(info.get("age", 0), 1)
        state = "stale" if info.get("stale_flag") else "fresh"
        has_trade = "🟢 open" if sym in open_trades else "⚪ none"
        print(f"{sym:10} | {score:<6.2f} | {conf:<5.1f} | {direction:<6} | {state:<8} | {age:>4} | {has_trade}")

    print("\nSummary:")
    print(f"  Ready for scoring: {len([i for i in cache.values() if not i.get('stale_flag')])}")
    print(f"  Waiting/stale: {len([i for i in cache.values() if i.get('stale_flag')])}")
    print(f"  Open trades: {len(open_trades)}")

if __name__ == "__main__":
    main()
