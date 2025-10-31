import importlib
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def stub_websocket_module(monkeypatch):
    module = types.ModuleType("websocket")
    module.WebSocketApp = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "websocket", module)
    try:
        yield
    finally:
        monkeypatch.delitem(sys.modules, "websocket", raising=False)


def reload_backend():
    sys.modules.pop("ws_backend_client", None)
    return importlib.import_module("ws_backend_client")


def test_run_forever_kwargs_defaults(monkeypatch):
    module = reload_backend()
    assert module._run_forever_kwargs() == {
        "ping_interval": 20.0,
        "ping_timeout": 10.0,
    }


def test_run_forever_kwargs_custom_interval(monkeypatch):
    monkeypatch.setenv("WSCLIENT_PING_INTERVAL", "30")
    monkeypatch.setenv("WSCLIENT_PING_TIMEOUT", "12")
    module = reload_backend()
    assert module._run_forever_kwargs() == {
        "ping_interval": 30.0,
        "ping_timeout": 12.0,
    }


def test_run_forever_kwargs_disable_ping(monkeypatch):
    monkeypatch.setenv("WSCLIENT_PING_INTERVAL", "0")
    module = reload_backend()
    assert module._run_forever_kwargs() == {}


def test_run_forever_kwargs_fallback(monkeypatch):
    monkeypatch.delenv("WSCLIENT_PING_INTERVAL", raising=False)
    monkeypatch.delenv("WSCLIENT_PING_TIMEOUT", raising=False)
    monkeypatch.setenv("WS_PING_INTERVAL", "6")
    monkeypatch.setenv("WS_PING_TIMEOUT", "3")
    module = reload_backend()
    assert module._run_forever_kwargs() == {
        "ping_interval": 6.0,
        "ping_timeout": 3.0,
    }
