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


def test_get_news_status_reflects_active_halt():
    now = 6_000.0
    news_risk.process_news_item("SEC sues Binance", "", "Reuters", now=now)
    status = news_risk.get_news_status(now=now + 60)
    assert status["mode"] == "HARD_HALT"
    assert status["category"] == "CRYPTO_SYSTEMIC"
    assert status["ttl_secs"] == 120 * 60 - 60
    assert status["last_event_headline"] == "SEC sues Binance"
    assert status["last_event_ts"] == now


def test_format_news_status_line_custom_status():
    status = {
        "mode": "HARD_HALT",
        "category": "CRYPTO_SYSTEMIC",
        "ttl_secs": 125,
        "reason": "CRYPTO_SYSTEMIC: SEC sues Binance",
        "last_event_headline": "SEC sues Binance",
        "last_event_ts": 1_000.0,
    }
    line = news_risk.format_news_status_line(status=status)
    assert "HARD_HALT" in line
    assert "CRYPTO_SYSTEMIC" in line
    assert "SEC sues Binance" in line
    assert "2m" in line  # 125 seconds should round down to 2 minutes left


def test_write_and_load_news_status(tmp_path, monkeypatch):
    now = 7_000.0
    news_risk.process_news_item("SEC sues Binance", "", "Reuters", now=now)
    status_file = tmp_path / "news_status.json"
    monkeypatch.setattr(news_risk, "NEWS_STATUS_FILE", str(status_file))
    monkeypatch.setenv("NEWS_STATUS_FILE", str(status_file))
    news_risk.write_news_status(now=now)
    loaded = news_risk.load_news_status()
    assert loaded["mode"] == "HARD_HALT"
    assert loaded["category"] == "CRYPTO_SYSTEMIC"
    assert loaded["last_event_headline"] == "SEC sues Binance"


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


def test_expansion_news_classified_as_crypto_medium():
    assert (
        news_risk.classify_news(
            "Another Bank Secures Key Hong Kong License to Launch Institutional Crypto Trading Services",
            "",
        )
        == "CRYPTO_MEDIUM"
    )
