import importlib
import config


def reload_config():
    importlib.reload(config)


def clear_env(monkeypatch):
    monkeypatch.delenv("TRADE_LLM_MODEL", raising=False)
    monkeypatch.delenv("MACRO_LLM_MODEL", raising=False)
    monkeypatch.delenv("NEWS_LLM_MODEL", raising=False)
    monkeypatch.delenv("GROQ_OVERFLOW_MODEL", raising=False)


def test_default_models(monkeypatch):
    clear_env(monkeypatch)
    reload_config()
    assert config.get_groq_model() == config.DEFAULT_GROQ_MODEL
    assert config.get_macro_model() == config.DEFAULT_MACRO_MODEL
    assert config.get_news_model() == config.DEFAULT_NEWS_MODEL
    assert config.get_overflow_model() == config.DEFAULT_OVERFLOW_MODEL


def test_deprecated_models_map_to_overflow(monkeypatch):
    clear_env(monkeypatch)
    reload_config()

    getters = {
        "TRADE_LLM_MODEL": config.get_groq_model,
        "MACRO_LLM_MODEL": config.get_macro_model,
        "NEWS_LLM_MODEL": config.get_news_model,
    }

    for env_var, getter in getters.items():
        for deprecated in ("llama3-70b-8192", "llama-3.1-70b", "llama-3.1-70b-versatile"):
            monkeypatch.setenv(env_var, deprecated)
            reload_config()
            assert getter() == config.DEFAULT_OVERFLOW_MODEL
        monkeypatch.delenv(env_var, raising=False)
        reload_config()


def test_custom_models(monkeypatch):
    clear_env(monkeypatch)
    reload_config()

    monkeypatch.setenv("TRADE_LLM_MODEL", "custom-trade")
    monkeypatch.setenv("MACRO_LLM_MODEL", "custom-macro")
    monkeypatch.setenv("NEWS_LLM_MODEL", "custom-news")
    monkeypatch.setenv("GROQ_OVERFLOW_MODEL", "custom-overflow")
    reload_config()

    assert config.get_groq_model() == "custom-trade"
    assert config.get_macro_model() == "custom-macro"
    assert config.get_news_model() == "custom-news"
    assert config.get_overflow_model() == "custom-overflow"

    clear_env(monkeypatch)
    reload_config()
