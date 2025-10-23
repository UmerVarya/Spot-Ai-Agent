import json
import time
from datetime import datetime
from pathlib import Path

import realtime_signal_cache as rtsc

if hasattr(rtsc, "cache_diagnostics_path"):
    CACHE_PATH = Path(rtsc.cache_diagnostics_path())
else:  # pragma: no cover - legacy compatibility
    CACHE_PATH = Path(rtsc.cache_state_file())


def _metric_display(metric):
    if isinstance(metric, dict):
        value = metric.get("display")
        if isinstance(value, (int, float)):
            return float(value)
        value = metric.get("raw")
        if isinstance(value, (int, float)):
            return float(value)
        return None
    if isinstance(metric, (int, float)):
        return float(metric)
    return None


def load_cache():
    if not CACHE_PATH.exists():
        print(
            "⚠️ No diagnostics snapshot found — start the agent or trigger pending_diagnostics()."
        )
        print(f"Expected at: {CACHE_PATH}")
        return

    with CACHE_PATH.open("r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "pending" in data:
        generated = data.get("generated_at")
        timestamp = (
            datetime.utcfromtimestamp(float(generated)).strftime("%H:%M:%S")
            if isinstance(generated, (int, float))
            else datetime.utcnow().strftime("%H:%M:%S")
        )
        ready = list(map(str, data.get("ready", [])))
        pending = list(data.get("pending", []))
        universe = data.get("universe")
        tracked = (
            len(universe)
            if isinstance(universe, (list, tuple, set))
            else len(set(ready)) + len(pending)
        )
        print(f"\n🔍 RealTimeSignalCache Status ({timestamp} UTC)")
        print("─────────────────────────────────────────────")
        print(f"Symbols tracked: {tracked}")
        print(f"Ready for scoring: {len(ready)} ✅")
        print(f"Waiting / stale: {len(pending)} 🕒\n")

        def sort_key(entry):
            waiting = _metric_display(entry.get("waiting_for"))
            stale_age = _metric_display(entry.get("stale_age"))
            return (
                waiting if waiting is not None else -1.0,
                stale_age if stale_age is not None else -1.0,
            )

        for entry in sorted(pending, key=sort_key, reverse=True):
            symbol = str(entry.get("symbol", "?"))
            pieces = []
            waiting_for = _metric_display(entry.get("waiting_for"))
            if waiting_for is not None:
                pieces.append(f"pending={waiting_for:.1f}s")
            stale_age = _metric_display(entry.get("stale_age"))
            if stale_age is not None:
                pieces.append(f"stale={stale_age:.1f}s")
            request_wait = _metric_display(entry.get("request_wait"))
            if request_wait is not None:
                pieces.append(f"since-request={request_wait:.1f}s")
            error_msg = entry.get("last_error")
            if error_msg:
                pieces.append(f"error={error_msg}")
            print(f"  - {symbol:10s} {' | '.join(pieces)}")

        print("─────────────────────────────────────────────")
        return

    # Legacy flat JSON mapping fallback
    if not isinstance(data, dict):
        print("⚠️ Unsupported diagnostics format.")
        return

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
