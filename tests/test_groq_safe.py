import logging

import pytest

import groq_safe
from groq_safe import GroqAuthError
from groq_http import reset_groq_key_state


class DummyClient:
    pass


def test_missing_key_disables(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    groq_safe.reset_auth_state()
    reset_groq_key_state()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(GroqAuthError):
            groq_safe.require_groq_api_key()

    assert any("Groq disabled: no GROQ_API_KEY in environment" in rec.message for rec in caplog.records)
    assert groq_safe._AUTH_DISABLED is True


def test_successful_call_with_key(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    groq_safe.reset_auth_state()
    reset_groq_key_state()

    monkeypatch.setattr(
        groq_safe,
        "http_chat_completion",
        lambda **_kwargs: ("ok", None, None),
    )

    response = groq_safe.safe_chat_completion(
        DummyClient(),
        messages=[{"role": "user", "content": "hi"}],
        model="llama",
    )

    assert response.choices[0].message.content == "ok"
    assert any("Groq setup: key_present=True" in rec.message for rec in caplog.records)


def test_auth_failure_disables(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    groq_safe.reset_auth_state()
    reset_groq_key_state()

    def _auth_fail(**_kwargs):
        return None, 401, {"error": {"code": "invalid_api_key", "message": "invalid"}}

    monkeypatch.setattr(groq_safe, "http_chat_completion", _auth_fail)

    with pytest.raises(GroqAuthError):
        groq_safe.safe_chat_completion(
            DummyClient(),
            messages=[{"role": "user", "content": "hi"}],
            model="llama",
        )

    assert groq_safe._AUTH_DISABLED is True

    with pytest.raises(GroqAuthError):
        groq_safe.safe_chat_completion(
            DummyClient(),
            messages=[{"role": "user", "content": "hi again"}],
            model="llama",
        )
