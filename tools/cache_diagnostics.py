#!/usr/bin/env python3
"""
Extended RealTimeSignalCache diagnostic tool.
Shows symbol freshness, live scores, and open trade info.
"""

import os, json
from datetime import datetime

from core.realtime_signal_cache import RealTimeSignalCache
from core.trade_storage import TradeStorage

def main():
    print(f"\n🧠  RealTimeSignalCache Status ({datetime.utcnow():%H:%M:%S UTC})\n")

    # Load RTSC
    rtsc = RealTimeSignalCache.load()
    cache = getattr(rtsc, "cache", {})
    print(f"Symbols tracked: {len(cache)}")

    # Load open trades
    try:
        store = TradeStorage.load()
        open_trades = {t["symbol"]: t for t in store.open_trades.values()}
    except Exception:
        open_trades = {}

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
