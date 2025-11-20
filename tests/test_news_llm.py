import asyncio
from datetime import datetime

import pytest

import news_llm
import news_risk


def _drain_queue():
    news_llm._LLM_QUEUE.clear()
    news_llm._LLM_QUEUE_IDS.clear()


def test_queue_for_llm_deduplicates(monkeypatch):
    _drain_queue()
    monkeypatch.setattr(news_llm, "NEWS_LLM_ENABLED", True)
    item = news_llm.NewsLLMInput(
        event_id="abc",
        headline="test",
        body="",
        source="wire",
        rule_category="CRYPTO_MEDIUM",
        ts=datetime.utcnow().timestamp(),
    )
    assert news_llm.queue_for_llm(item) is True
    assert news_llm.queue_for_llm(item) is False
    assert len(news_llm._LLM_QUEUE) == 1


@pytest.mark.asyncio
async def test_run_llm_batch_loop_respects_batch(monkeypatch):
    _drain_queue()
    monkeypatch.setattr(news_llm, "NEWS_LLM_ENABLED", True)
    monkeypatch.setattr(news_llm, "NEWS_LLM_MAX_ITEMS_PER_BATCH", 2)

    captured = []

    async def fake_call(batch):
        assert len(batch) <= 2
        return [
            news_llm.NewsLLMDecision(
                event_id=item.event_id,
                systemic_risk=0,
                direction="unclear",
                suggested_category=item.rule_category,
                reason="",
            )
            for item in batch
        ]

    monkeypatch.setattr(news_llm, "_call_llm_batch", fake_call)

    def apply(decision):
        captured.append(decision.event_id)

    monkeypatch.setattr(news_risk, "apply_llm_decision", apply)

    for idx in range(3):
        news_llm.queue_for_llm(
            news_llm.NewsLLMInput(
                event_id=f"id-{idx}",
                headline="h",
                body="b",
                source="s",
                rule_category="CRYPTO_MEDIUM",
                ts=datetime.utcnow().timestamp(),
            )
        )

    await asyncio.sleep(0)
    stop_event = asyncio.Event()
    task = asyncio.create_task(news_llm.run_llm_batch_loop(stop_event=stop_event))
    await asyncio.sleep(0.1)
    stop_event.set()
    await task

    assert set(captured) >= {"id-0", "id-1"}


def test_apply_llm_decision_upgrade_and_downgrade(monkeypatch):
    news_risk.reset_news_halt_state()
    monkeypatch.setattr(news_risk, "NEWS_LLM_ALLOW_UPGRADE", True)
    monkeypatch.setattr(news_risk, "NEWS_LLM_ALLOW_DOWNGRADE", True)

    # Seed medium event context
    result = news_risk.process_news_item(
        headline="Bank expands crypto services",
        body="",
        source="wire",
        now=0,
    )
    event_id = news_risk.make_event_id("Bank expands crypto services", "wire")

    decision = news_llm.NewsLLMDecision(
        event_id=event_id,
        systemic_risk=2,
        direction="bearish",
        suggested_category="CRYPTO_SYSTEMIC",
        reason="",
    )
    news_risk.apply_llm_decision(decision)
    assert news_risk.halt_state.category == "CRYPTO_SYSTEMIC"

    # Downgrade should mark suppression without shortening halt
    sys_event_id = news_risk.make_event_id("usdt depeg risk", "wire")
    news_risk.process_news_item(
        headline="usdt depeg risk",
        body="",
        source="wire",
        now=0,
    )
    downgrade = news_llm.NewsLLMDecision(
        event_id=sys_event_id,
        systemic_risk=0,
        direction="unclear",
        suggested_category="CRYPTO_MEDIUM",
        reason="",
    )
    news_risk.apply_llm_decision(downgrade)
    assert sys_event_id in news_risk._LLM_SUPPRESS_REHALT
