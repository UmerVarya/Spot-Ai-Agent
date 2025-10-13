from datetime import datetime, timedelta, timezone


def test_select_symbol_news_headlines_filters_noise():
    import brain

    now = datetime.now(timezone.utc)
    articles = [
        {
            "title": "Bitcoin ETF inflows hit new record",
            "description": "Institutional demand surges after approval.",
            "publishedAt": now.isoformat(),
            "source": {"name": "Bloomberg"},
        },
        {
            "title": "BTC price analysis for the day",
            "description": "Purely technical view",
            "publishedAt": now.isoformat(),
            "source": {"name": "Blog"},
        },
    ]

    selected = brain._select_symbol_news_headlines(articles, limit=2)

    assert len(selected) == 1
    assert selected[0]["title"].startswith("Bitcoin ETF inflows")


def test_summarize_symbol_news_returns_curated_headlines(monkeypatch):
    import brain

    now = datetime.now(timezone.utc)
    recent = now - timedelta(hours=2)
    articles = [
        {
            "title": "Major bank discloses $100M Bitcoin purchase",
            "description": "Regulatory filing confirms significant treasury allocation.",
            "publishedAt": recent.isoformat(),
            "source": {"name": "Reuters"},
        },
        {
            "title": "Random blog post",
            "description": "Not relevant",
            "publishedAt": recent.isoformat(),
            "source": {"name": "Some Blog"},
        },
    ]

    calls = []

    def fake_fetch(symbol):
        calls.append(symbol)
        return articles

    monkeypatch.setattr(brain, "fetch_symbol_news_sync", fake_fetch, raising=False)
    monkeypatch.setattr(brain, "_symbol_news_cache", {}, raising=False)

    summary = brain.summarize_symbol_news("BTCUSDT")

    assert "Major bank discloses" in summary
    assert calls == ["BTCUSDT"]

    # Cached result should avoid additional fetches
    summary_again = brain.summarize_symbol_news("BTCUSDT")

    assert summary_again == summary
    assert calls == ["BTCUSDT"]
