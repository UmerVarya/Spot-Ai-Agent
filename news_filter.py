import json
import os
from dotenv import load_dotenv
from log_utils import setup_logger
import config
from llm_tasks import LLMTask, call_llm_for_task
from news_guardrails import (
    parse_llm_json,
    quantify_event_risk,
    reconcile_with_quant_filters,
)

load_dotenv()
logger = setup_logger(__name__)


def load_events(path="news_events.json"):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def format_events_for_prompt(events, metrics):
    lines = [
        "You are a professional crypto macro analyst.",
        "Evaluate the following economic or crypto-related events and determine if they could impact the crypto market significantly within the next 6 hours (UTC).",
        "Respond ONLY with a JSON object containing the keys `safe_decision` (\"yes\" or \"no\") and `reason` (string).",
        "Do not include any extra commentary.",
        "",
        f"Events considered ({metrics['considered_events']} total, {metrics['high_impact_events']} high impact):",
    ]
    if metrics["considered_events"] == 0:
        lines.append("- No events are occurring in the next 6 hours.")
    else:
        for event in metrics["events_in_window"]:
            relevance = event.get("relevance", {})
            category = relevance.get("category")
            score = relevance.get("score")
            extras = []
            if category:
                extras.append(f"category: {category}")
            if score:
                extras.append(f"relevance: {score}")
            extra_text = f", {'; '.join(extras)}" if extras else ""
            lines.append(
                f"- {event.get('event')} at {event.get('datetime')} "
                f"(impact: {event.get('impact')}{extra_text})"
            )
    return "\n".join(lines)


def analyze_news_with_llm(prompt, metrics):
    try:
        response, model_used = call_llm_for_task(
            LLMTask.NEWS,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a crypto macro risk analyst. Respond ONLY with a JSON object containing "
                        "`safe_decision` (\"yes\" or \"no\") and `reason` (string)."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        if response is None:
            logger.warning("Groq client unavailable for news filter; assuming safe")
            return {"safe": True, "sensitivity": 0, "reason": "LLM unavailable"}

        reply = response.choices[0].message.content
        safe_decision, reason = parse_llm_json(reply, logger)
        if safe_decision is None:
            return {"safe": True, "sensitivity": 0.0, "reason": reason or "LLM error"}

        safe, sensitivity, reconciled_reason = reconcile_with_quant_filters(
            safe_decision, reason or "No reason provided.", metrics
        )
        logger.info(
            "News LLM decision: task=news model=%s safe=%s reason=%s",
            model_used,
            safe,
            reconciled_reason,
        )
        return {"safe": safe, "sensitivity": sensitivity, "reason": reconciled_reason}
    except Exception as e:
        logger.warning("Groq analysis failed: %s", e, exc_info=True)
        return {"safe": True, "sensitivity": 0, "reason": "LLM error or no response. Assuming safe."}


def news_filter():
    events = load_events()
    if not events:
        return {"safe": True, "sensitivity": 0, "reason": "No scheduled events found. Proceeding safely."}
    metrics = quantify_event_risk(events)
    if metrics["considered_events"] == 0:
        return {"safe": True, "sensitivity": 0.0, "reason": "No events within the risk window."}
    prompt = format_events_for_prompt(events, metrics)
    return analyze_news_with_llm(prompt, metrics)


if __name__ == "__main__":
    result = news_filter()
    logger.info("News Filter Result: %s", result)
