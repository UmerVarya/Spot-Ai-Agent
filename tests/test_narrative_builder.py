from types import SimpleNamespace

import narrative_builder


def test_generate_trade_narrative_uses_narrative_model(monkeypatch) -> None:
    fake_client = object()

    captured = {}

    def fake_safe_chat_completion(client, *, model, messages, **kwargs):
        captured["client"] = client
        captured["model"] = model
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Generated narrative")
                )
            ]
        )

    monkeypatch.setattr(narrative_builder, "safe_chat_completion", fake_safe_chat_completion)
    monkeypatch.setattr(narrative_builder, "get_groq_client", lambda: fake_client)
    monkeypatch.setattr(
        narrative_builder.config, "get_narrative_model", lambda: "narrative-model"
    )

    result = narrative_builder.generate_trade_narrative(symbol="BTCUSDT")

    assert result == "Generated narrative"
    assert captured["client"] is fake_client
    assert captured["model"] == "narrative-model"
