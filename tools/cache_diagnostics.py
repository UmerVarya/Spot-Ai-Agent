import os
import json
import time
from datetime import datetime

from realtime_signal_cache import cache_state_file


def load_cache():
    if not os.path.exists(cache_state_file):
        print("⚠️ No cache file found — RTSC not initialized yet.")
        return

    with open(cache_state_file, "r") as f:
        data = json.load(f)

    now = time.time()
    ready, stale = [], []

    for sym, meta in data.items():
        age = now - meta.get("last_update", 0)
        if meta.get("bars", 0) >= 20 and age < 180:
            ready.append(sym)
        else:
            stale.append((sym, age))

    print(
        f"\n🔍 RealTimeSignalCache Status ({datetime.utcnow().strftime('%H:%M:%S')} UTC)"
    )
    print("─────────────────────────────────────────────")
    print(f"Symbols tracked: {len(data)}")
    print(f"Ready for scoring: {len(ready)} ✅")
    print(f"Waiting / stale: {len(stale)} 🕒\n")

    for sym, age in sorted(stale, key=lambda x: -x[1]):
        print(f"  - {sym:10s} (age={age:.1f}s)")

    print("─────────────────────────────────────────────")


if __name__ == "__main__":
    load_cache()
