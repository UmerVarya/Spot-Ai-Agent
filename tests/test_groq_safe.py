import logging

import pytest

import groq_http
import groq_safe
from groq_safe import GroqAuthError


class DummyClient:
    pass


def test_missing_key_disables(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    groq_safe.reset_auth_state()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(GroqAuthError):
            groq_safe.require_groq_api_key()

    assert any(
        "Groq disabled: no GROQ_API_KEY in environment" in rec.message
        for rec in caplog.records
    )
    assert groq_safe._groq_auth_disabled is True


def test_successful_call_with_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    groq_safe.reset_auth_state()

    assert groq_safe.require_groq_api_key() == "gsk_test"
    assert groq_safe._groq_auth_disabled is False


def test_auth_failure_disables(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key")
    groq_safe.reset_auth_state()

    def _auth_fail(**_kwargs):
        groq_safe._groq_auth_disabled = True
        raise GroqAuthError("Groq authentication disabled")

    monkeypatch.setattr(groq_safe, "http_chat_completion", _auth_fail)

    with pytest.raises(GroqAuthError):
        groq_safe.safe_chat_completion(
            DummyClient(),
            messages=[{"role": "user", "content": "hi"}],
            model="llama",
        )

    assert groq_safe._groq_auth_disabled is True

    with pytest.raises(GroqAuthError):
        groq_safe.require_groq_api_key()


def test_http_client_disables_on_401(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_valid")
    groq_safe.reset_auth_state()

    class DummyResponse:
        status_code = 401

        def json(self):
            return {"error": {"code": "invalid_api_key", "message": "invalid"}}

    monkeypatch.setattr(groq_http.requests, "post", lambda *_args, **_kwargs: DummyResponse())

    with pytest.raises(GroqAuthError):
        groq_http.http_chat_completion(
            model="llama",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=1,
        )

    assert groq_safe._groq_auth_disabled is True

    with pytest.raises(GroqAuthError):
        groq_safe.require_groq_api_key()
