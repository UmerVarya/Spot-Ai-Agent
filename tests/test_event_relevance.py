from datetime import datetime, timedelta, timezone

import pandas as pd

from event_relevance import EventRelevanceScorer


def _ts(hour: int) -> datetime:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return base + timedelta(hours=hour)


def test_score_event_identifies_macro_relevance():
    scorer = EventRelevanceScorer()
    event = {
        "event": "US CPI release",
        "impact": "high",
        "datetime": _ts(0).isoformat(),
    }

    relevance = scorer.score_event(event)

    assert relevance.score > 0
    assert relevance.category == "macro:cpi"
    assert relevance.halt_relevant is True


def test_score_event_filters_fx_forecasts():
    scorer = EventRelevanceScorer()
    event = {
        "event": "EUR/JPY technical forecast for tomorrow",
        "impact": "high",
        "datetime": _ts(1).isoformat(),
    }

    relevance = scorer.score_event(event)

    assert relevance.score == 0
    assert relevance.category is None
    assert relevance.halt_relevant is False


def test_score_events_returns_aggregate_metrics():
    scorer = EventRelevanceScorer()
    events = [
        {"event": "US CPI release", "impact": "high", "datetime": _ts(0).isoformat()},
        {
            "event": "Regional FX outlook",
            "impact": "medium",
            "datetime": _ts(1).isoformat(),
        },
    ]

    summary = scorer.score_events(events)

    assert summary["relevant_events"] == 1
    assert summary["halt_relevant_events"] == 1
    assert summary["aggregate_score"] > 0
    assert summary["events"] and summary["events"][0]["relevance"]["category"] == "macro:cpi"


def test_fit_updates_halt_categories_based_on_volatility():
    index = pd.date_range(_ts(0), periods=60, freq="H")
    prices = pd.Series(100.0, index=index)

    # Inject two high volatility windows following CPI and exchange hack events.
    prices.loc[_ts(10):] = 105.0
    prices.loc[_ts(11):] = 120.0
    prices.loc[_ts(12):] = 135.0
    prices.loc[_ts(16):] = 135.0
    prices.loc[_ts(30):] = 140.0
    prices.loc[_ts(31):] = 110.0
    prices.loc[_ts(32):] = 80.0
    prices.loc[_ts(36):] = 80.0

    events = [
        {"event": "US CPI release", "impact": "high", "datetime": _ts(10).isoformat()},
        {
            "event": "Major exchange hack triggers withdrawal halt",
            "impact": "high",
            "datetime": _ts(30).isoformat(),
        },
    ]

    scorer = EventRelevanceScorer(min_observations=1, quantile=0.6, baseline_window="6H")
    scorer.fit(prices, events)

    assert scorer.halt_categories.issuperset({"macro:cpi", "crypto:exchange"})
    assert set(scorer.category_stats) == {"macro:cpi", "crypto:exchange"}
    assert scorer.category_stats["macro:cpi"]["hits"] >= 1
