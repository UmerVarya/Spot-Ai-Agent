import importlib
import logging

import groq_llm


def test_group_rate_limit_headers_splits_buckets():
    headers = {
        "X-RateLimit-Limit-Requests-1m": "30",
        "X-RateLimit-Remaining-Requests-1m": "25",
        "X-RateLimit-Limit-Tokens-1m": "15000",
        "X-RateLimit-Remaining-Tokens-1m": "12000",
    }

    grouped = groq_llm._group_rate_limit_headers(headers)

    assert grouped["requests-1m"]["limit"] == "30"
    assert grouped["requests-1m"]["remaining"] == "25"
    assert grouped["tokens-1m"]["limit"] == "15000"
    assert grouped["tokens-1m"]["remaining"] == "12000"


def test_log_rate_limit_health_warns_when_almost_empty():
    importlib.reload(groq_llm)
    headers = {
        "X-RateLimit-Limit-Requests-1m": "30",
        "X-RateLimit-Remaining-Requests-1m": "2",
        "X-RateLimit-Used-Requests-1m": "28",
        "X-RateLimit-Reset-Requests-1m": "2024-01-01T00:00:30Z",
    }

    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
            records.append(record)

    handler = Capture()
    logger = groq_llm.logger
    logger.addHandler(handler)
    try:
        groq_llm._log_rate_limit_health(headers, 200)
    finally:
        logger.removeHandler(handler)

    assert any("Groq rate limit for requests-1m" in record.getMessage() for record in records)
    assert any("reducing request concurrency" in record.getMessage() for record in records)
