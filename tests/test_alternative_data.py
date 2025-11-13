import math
import math
import pytest

from alternative_data import AlternativeDataBundle, OnChainMetrics, SocialSentiment


def test_alternative_data_score_adjustment_directional():
    onchain = OnChainMetrics(
        exchange_inflow=100.0,
        exchange_outflow=180.0,
        whale_inflow=10.0,
        whale_outflow=30.0,
        whale_ratio=0.6,
        net_exchange_flow=(180.0 - 100.0) / (180.0 + 100.0),
        large_holder_netflow=(30.0 - 10.0) / (30.0 + 10.0),
        composite_score=0.4,
    )
    social = SocialSentiment(
        bias="bullish",
        score=0.65,
        confidence=0.8,
        posts_analyzed=42,
        source_models=("groq",),
    )
    bundle = AlternativeDataBundle(onchain=onchain, social=social, fetched_at=0.0)
    long_adj = bundle.score_adjustment("long")
    short_adj = bundle.score_adjustment("short")
    assert long_adj > 0
    assert short_adj < 0
    assert math.isclose(long_adj, -short_adj, rel_tol=1e-6)
    features = bundle.to_features("long")
    assert features["social_bias"] == "bullish"
    assert features["onchain_score"] == pytest.approx(0.4)
    assert features["score_adjustment"] == pytest.approx(long_adj)


def test_get_alternative_data_defaults_without_keys(monkeypatch):
    monkeypatch.setenv("GLASSNODE_API_KEY", "")
    monkeypatch.setenv("TWITTER_BEARER_TOKEN", "")
    monkeypatch.setenv("ENABLE_REDDIT_SCRAPE", "0")
    # Ensure module-level globals reflect the patched environment
    import importlib
    import alternative_data as alt_mod

    alt_mod = importlib.reload(alt_mod)
    bundle = alt_mod.get_alternative_data("BTCUSDT", ttl=0.0, force_refresh=True)
    assert isinstance(bundle, alt_mod.AlternativeDataBundle)
    features = bundle.to_features("long")
    assert features["onchain_score"] == 0.0
    assert features["social_bias"] == "neutral"
    assert features["social_posts"] == 0
