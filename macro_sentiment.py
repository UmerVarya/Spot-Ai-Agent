from news_scraper import get_combined_headlines
from news_retriever import build_retrieval_context, load_structured_events
from log_utils import setup_logger
from llm_tasks import LLMTask, call_llm_for_task

logger = setup_logger(__name__)

def analyze_macro_sentiment():
    headlines = get_combined_headlines(limit=None)
    structured_events = load_structured_events()

    if not headlines and not structured_events:
        return {
            "summary": "No news headlines available.",
            "bias": "neutral",
            "confidence": 0
        }

    documents = headlines + structured_events
    sources = ["headline"] * len(headlines) + ["event"] * len(structured_events)

    context = build_retrieval_context(
        query=(
            "macro environment for crypto markets including regulatory "
            "shifts, macroeconomic releases, institutional flows and on-chain signals"
        ),
        documents=documents,
        sources=sources,
        top_k=18,
    )

    prompt = f"""
You are a professional macroeconomic and crypto news analyst.
You will receive a curated context window that includes the most relevant
headlines and structured events retrieved from a semantic search corpus.

Context:
{context}

Using only this context, analyze and summarize the prevailing market sentiment.
Highlight notable macro drivers, regulatory actions and crypto-native signals.

Return your response exactly in this format:
Summary: <concise summary>
Bias: <bullish / bearish / neutral>
Confidence: <0-10 score>
"""

    try:
        response, model_used = call_llm_for_task(
            LLMTask.NEWS,
            messages=[
                {"role": "system", "content": "You are a crypto macro market analyst."},
                {"role": "user", "content": prompt}
            ],
        )

        if response is None:
            logger.warning("Groq client unavailable for macro sentiment; returning default")
            return {
                "summary": "LLM analysis unavailable.",
                "bias": "neutral",
                "confidence": 0,
            }

        result = response.choices[0].message.content.strip()
        logger.debug("Raw LLM Response: %s", result)

        summary = "No summary extracted"
        bias = "neutral"
        confidence = 0

        for line in result.splitlines():
            lower = line.lower()
            if "summary:" in lower:
                summary = line.split(":", 1)[-1].strip()
            elif "bias:" in lower:
                b = line.split(":", 1)[-1].strip().lower()
                if b in ["bullish", "bearish", "neutral"]:
                    bias = b
            elif "confidence:" in lower:
                try:
                    confidence = float(line.split(":", 1)[-1].strip())
                except:
                    confidence = 0

        logger.info(
            "News LLM decision: task=news model=%s bias=%s confidence=%s",
            model_used,
            bias,
            confidence,
        )

        return {
            "summary": summary,
            "bias": bias,
            "confidence": confidence
        }

    except Exception as e:
        logger.error("News sentiment LLM error: %s", e, exc_info=True)
        return {
            "summary": "Error during LLM analysis.",
            "bias": "neutral",
            "confidence": 0
        }

# === Optional test ===
if __name__ == "__main__":
    result = analyze_macro_sentiment()
    logger.info("LLM Market Sentiment Analysis")
    logger.info("Summary: %s", result['summary'])
    logger.info("Bias: %s | Confidence: %s", result['bias'].upper(), result['confidence'])
