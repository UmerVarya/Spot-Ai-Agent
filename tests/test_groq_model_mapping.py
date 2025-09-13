import importlib
import config


def reload_config():
    importlib.reload(config)


def test_default_model(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    reload_config()
    assert config.get_groq_model() == "llama-3.1-70b"


def test_deprecated_model(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    reload_config()
    assert config.get_groq_model() == "llama-3.1-70b"
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    reload_config()


def test_custom_model(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "custom-model")
    reload_config()
    assert config.get_groq_model() == "custom-model"
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    reload_config()
