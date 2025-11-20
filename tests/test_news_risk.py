import importlib

import pytest

import news_llm
import news_risk


@pytest.fixture(autouse=True)
def _reset_state():
    news_risk.reset_news_halt_state()
    yield
    news_risk.reset_news_halt_state()


@pytest.fixture
def reload_news(monkeypatch):
    def _reload(**env):
        global news_risk, news_llm

        for key in [
            "NEWS_LLM_ENABLED",
            "NEWS_LLM_ALLOW_UPGRADE",
            "NEWS_LLM_ALLOW_DOWNGRADE",
            "NEWS_LLM_CONFIRM_FOR_HALT",
        ]:
            monkeypatch.delenv(key, raising=False)

        for key, value in env.items():
            monkeypatch.setenv(key, str(value))

        news_llm = importlib.reload(news_llm)
        news_risk = importlib.reload(news_risk)
        news_risk.reset_news_halt_state()
        return news_risk, news_llm

    return _reload


def test_llm_confirmation_applies_halt_for_systemic_candidate(reload_news):
    news_risk, news_llm = reload_news(
        NEWS_LLM_ENABLED=True, NEWS_LLM_CONFIRM_FOR_HALT=True
    )
    now = 1_000.0

    news_risk.process_news_item("SEC sues Binance", "", "Reuters", now=now)

    assert news_risk.halt_state.halt_until == 0

    event_id = news_risk.make_event_id("SEC sues Binance", "Reuters")
    news_risk.apply_llm_decision(
        news_llm.NewsLLMDecision(
            event_id=event_id,
            systemic_risk=3,
            direction="down",
            suggested_category="CRYPTO_SYSTEMIC",
            reason="High risk",
        )
    )

    assert news_risk.halt_state.halt_until == pytest.approx(now + 120 * 60)
    gate_state = news_risk.get_news_gate_state(now=now + 60)
    assert gate_state["mode"] == "HARD_HALT"


def test_llm_rejection_prevents_halt_and_suppresses_event(reload_news):
    news_risk, news_llm = reload_news(
        NEWS_LLM_ENABLED=True, NEWS_LLM_CONFIRM_FOR_HALT=True
    )
    now = 2_000.0
    news_risk.process_news_item("SEC sues Binance", "", "Reuters", now=now)

    event_id = news_risk.make_event_id("SEC sues Binance", "Reuters")
    news_risk.apply_llm_decision(
        news_llm.NewsLLMDecision(
            event_id=event_id,
            systemic_risk=0,
            direction="neutral",
            suggested_category="CRYPTO_MEDIUM",
            reason="Not systemic",
        )
    )

    assert news_risk.halt_state.halt_until == 0
    assert not news_risk.should_apply_halt_for_event(event_id, now + 30)


def test_llm_disabled_with_confirmation_prevents_rule_halt(reload_news):
    news_risk, _ = reload_news(NEWS_LLM_ENABLED=False, NEWS_LLM_CONFIRM_FOR_HALT=True)
    now = 3_000.0

    result = news_risk.process_news_item("FOMC rate decision", "", "Bloomberg", now=now)

    assert result["halt_applied"] is False
    assert news_risk.halt_state.halt_until == 0
    assert news_risk.get_news_gate_state(now=now + 10)["mode"] == "NONE"


def test_rules_only_mode_matches_legacy_behavior(reload_news):
    news_risk, _ = reload_news(NEWS_LLM_CONFIRM_FOR_HALT=False)
    now = 4_000.0

    news_risk.process_news_item("SEC sues Binance", "", "Reuters", now=now)
    assert news_risk.halt_state.halt_until == pytest.approx(now + 120 * 60)

    news_risk.reset_news_halt_state()
    news_risk.process_news_item("FOMC rate decision", "Fed to hold rates", "Bloomberg", now=now)
    assert news_risk.halt_state.category == "MACRO_USD_T1"
    assert news_risk.halt_state.halt_until == pytest.approx(now + 30 * 60)


def test_non_halting_categories_do_not_block_trading(reload_news):
    news_risk, _ = reload_news(NEWS_LLM_CONFIRM_FOR_HALT=False)
    now = 5_000.0
    news_risk.process_news_item("Binance lists new token", "", "Binance", now=now)
    assert news_risk.halt_state.halt_until == 0
    news_risk.process_news_item("US ISM services beats", "", "Reuters", now=now + 10)
    assert news_risk.halt_state.halt_until == 0
    news_risk.process_news_item("Random stock earnings", "", "CNBC", now=now + 20)
    assert news_risk.halt_state.halt_until == 0


def test_classify_news_crypto_systemic_and_policy_detections():
    assert (
        news_risk.classify_news(
            "SEC sues Binance for unregistered securities violations",
            "",
        )
        == "CRYPTO_SYSTEMIC"
    )
    assert (
        news_risk.classify_news(
            "USDT depeg: Tether falls to $0.92 on major exchange",
            "",
        )
        == "CRYPTO_SYSTEMIC"
    )
    assert (
        news_risk.classify_news(
            "Senate banking panel advances Trump’s FDIC pick Travis Hill as agency shifts its crypto approach",
            "",
        )
        == "CRYPTO_MEDIUM"
    )
    assert (
        news_risk.classify_news(
            "Senate banking committee holds hearing on FDIC’s approach to crypto oversight",
            "",
        )
        == "CRYPTO_MEDIUM"
    )
    assert (
        news_risk.classify_news(
            "FDIC orders major crypto exchange to halt withdrawals amid insolvency concerns",
            "",
        )
        == "CRYPTO_SYSTEMIC"
    )


def test_macro_t1_releases_vs_previews():
    assert (
        news_risk.classify_news(
            "Gold remains on the defensive below $4,100 amid sustained USD buying, ahead of US NFP",
            "",
        )
        == "MACRO_USD_T2"
    )
    assert (
        news_risk.classify_news(
            "US nonfarm payrolls rise by 250K in October, beating expectations",
            "",
        )
        == "MACRO_USD_T1"
    )
    assert (
        news_risk.classify_news(
            "FOMC leaves rates unchanged, signals higher for longer",
            "",
        )
        == "MACRO_USD_T1"
    )
    assert (
        news_risk.classify_news(
            "US CPI data: inflation cools more than expected in November",
            "",
        )
        == "MACRO_USD_T1"
    )


def test_expansion_news_classified_as_crypto_medium():
    assert (
        news_risk.classify_news(
            "Another Bank Secures Key Hong Kong License to Launch Institutional Crypto Trading Services",
            "",
        )
        == "CRYPTO_MEDIUM"
    )


def test_systemic_classification_not_masked_by_expansion_terms():
    assert (
        news_risk.classify_news(
            "SEC launches enforcement action against Binance",
            "",
        )
        == "CRYPTO_SYSTEMIC"
    )
