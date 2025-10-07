import math

import fused_sentiment as fs


def test_analyze_headlines_returns_two_models(monkeypatch):
    def fake_finllama(headlines):
        return 1, 0.9, "Bullish"

    def fake_fingpt(headlines):
        return 1, 0.8, "Reinforces upside"

    monkeypatch.setattr(fs, "_finllama_sentiment", fake_finllama)
    monkeypatch.setattr(fs, "_fingpt_sentiment", fake_fingpt)

    weights = {"finllama": 0.5, "fingpt": 0.5}
    result = fs.analyze_headlines(["headline"], fusion_weights=weights)

    assert set(result.keys()) >= {"finllama", "fingpt", "fused", "weights"}
    assert math.isclose(sum(result["weights"].values()), 1.0)
    # FinLlama and FinGPT should drive the fused score
    assert result["fused"]["score"] >= result["finllama"]["score"]
    assert result["fused"]["bias"] == "bullish"


def test_evaluate_and_calibrate_weights(monkeypatch):
    up_scores = {"finllama": 0.82, "fingpt": 0.88}
    down_scores = {"finllama": -0.96, "fingpt": -0.6}

    def fake_analyzer(headlines, fusion_weights=None):
        mapping = {"up": up_scores, "down": down_scores}
        scores = mapping[headlines[0]]
        weights = fs._normalise_weights(fusion_weights or fs.DEFAULT_FUSION_WEIGHTS)
        fused_score = sum(scores[name] * weights[name] for name in weights)
        bias = "bullish" if fused_score > 0 else "bearish" if fused_score < 0 else "neutral"
        return {
            "finllama": {"score": scores["finllama"], "confidence": 0.9},
            "fingpt": {"score": scores["fingpt"], "confidence": 0.85},
            "fused": {"score": fused_score, "bias": bias, "confidence": 0.8},
        }

    validation = [(["up"], 0.9), (["down"], -0.95)]

    metrics = fs.evaluate_models(validation, analyzer=fake_analyzer)
    assert metrics["fused"]["rmse"] >= 0

    weights = fs.calibrate_fusion_weights(validation, analyzer=fake_analyzer, step=0.1)
    assert math.isclose(sum(weights.values()), 1.0)
