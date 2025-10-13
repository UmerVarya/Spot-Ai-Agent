import asyncio
import time
from datetime import datetime, timezone
from news_monitor import LLMNewsMonitor


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def test_monitor_emits_alert_and_persists_state(tmp_path):
    events = [
        {"event": "Exchange Hack", "datetime": _iso_now(), "impact": "high"},
    ]

    async def fetcher():
        return events

    async def analyzer(received):
        assert list(received)
        return {"safe": False, "sensitivity": 0.92, "reason": "Major exchange hack"}

    alerts = []

    def callback(alert):
        alerts.append(alert)

    state_path = tmp_path / "state.json"
    monitor = LLMNewsMonitor(
        interval=30,
        alert_threshold=0.5,
        halt_threshold=0.8,
        status_path=str(state_path),
        fetcher=fetcher,
        analyzer=analyzer,
        alert_callback=callback,
    )

    state = asyncio.run(monitor.evaluate_now())
    assert state["halt_trading"] is True
    assert state["alert_triggered"] is True
    assert alerts and alerts[0].reason == "Major exchange hack"
    assert state_path.exists()

    # Running again with identical state should not duplicate alerts.
    asyncio.run(monitor.evaluate_now())
    assert len(alerts) == 1


def test_monitor_marks_state_stale():
    async def fetcher():
        return []

    async def analyzer(received):
        return {"safe": True, "sensitivity": 0.0, "reason": "Calm"}

    monitor = LLMNewsMonitor(
        interval=5,
        alert_threshold=0.7,
        halt_threshold=0.9,
        stale_after=0.05,
        fetcher=fetcher,
        analyzer=analyzer,
    )

    asyncio.run(monitor.evaluate_now())
    latest = monitor.get_latest_assessment()
    assert latest is not None
    assert latest["stale"] is False

    time.sleep(0.06)
    latest = monitor.get_latest_assessment()
    assert latest is not None
    assert latest["stale"] is True


def test_monitor_skips_alert_when_safe():
    async def fetcher():
        return [
            {"event": "Upgrade", "datetime": _iso_now(), "impact": "medium"},
        ]

    async def analyzer(received):
        return {"safe": True, "sensitivity": 0.2, "reason": "Routine upgrade"}

    alerts = []

    monitor = LLMNewsMonitor(
        interval=30,
        alert_threshold=0.5,
        halt_threshold=0.8,
        fetcher=fetcher,
        analyzer=analyzer,
        alert_callback=lambda alert: alerts.append(alert),
    )

    state = asyncio.run(monitor.evaluate_now())
    assert state["alert_triggered"] is False
    assert alerts == []
