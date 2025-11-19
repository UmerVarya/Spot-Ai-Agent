import pytest

import news_risk


@pytest.fixture(autouse=True)
def _reset_state():
    news_risk.reset_news_halt_state()
    yield
    news_risk.reset_news_halt_state()


def test_crypto_systemic_triggers_two_hour_halt():
    now = 1_000.0
    result = news_risk.process_news_item("SEC sues Binance", "", "Reuters", now=now)
    assert result["category"] == "CRYPTO_SYSTEMIC"
    assert news_risk.halt_state.halt_until == pytest.approx(now + 120 * 60)
    gate_state = news_risk.get_news_gate_state(now=now + 60)
    assert gate_state["mode"] == "HARD_HALT"
    assert gate_state["ttl_secs"] == 120 * 60 - 60


def test_macro_usd_t1_triggers_thirty_minute_halt():
    now = 2_000.0
    news_risk.process_news_item("FOMC rate decision", "Fed to hold rates", "Bloomberg", now=now)
    assert news_risk.halt_state.category == "MACRO_USD_T1"
    assert news_risk.halt_state.halt_until == pytest.approx(now + 30 * 60)


def test_non_halting_categories_do_not_block_trading():
    now = 3_000.0
    news_risk.process_news_item("Binance lists new token", "", "Binance", now=now)
    assert news_risk.halt_state.halt_until == 0
    news_risk.process_news_item("US ISM services beats", "", "Reuters", now=now + 10)
    assert news_risk.halt_state.halt_until == 0
    news_risk.process_news_item("Random stock earnings", "", "CNBC", now=now + 20)
    assert news_risk.halt_state.halt_until == 0


def test_duplicate_systemic_event_does_not_extend_halt():
    now = 4_000.0
    news_risk.process_news_item("SEC sues Binance", "", "Reuters", now=now)
    first_until = news_risk.halt_state.halt_until
    news_risk.process_news_item("SEC sues Binance", "", "Reuters", now=now + 60)
    assert news_risk.halt_state.halt_until == first_until


def test_shorter_event_does_not_shorten_existing_halt():
    now = 5_000.0
    news_risk.process_news_item("SEC sues Binance", "", "Reuters", now=now)
    long_until = news_risk.halt_state.halt_until
    news_risk.process_news_item("FOMC rate decision", "Fed signals pause", "Bloomberg", now=now + 30)
    assert news_risk.halt_state.halt_until == long_until
    assert news_risk.halt_state.category == "CRYPTO_SYSTEMIC"
