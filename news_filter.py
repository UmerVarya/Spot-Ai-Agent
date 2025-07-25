import json
from datetime import datetime, timedelta
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

def load_events(path="news_events.json"):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def format_events_for_prompt(events):
    prompt = """You are a professional crypto macro analyst.

Evaluate the following economic or crypto-related events and determine if they could impact the crypto market significantly within the next 6 hours (UTC).

Respond in the following JSON format:
{
  "safe": true/false,
  "sensitivity": 0-10,
  "reason": "..."
}

Now here are the upcoming events:
"""

    now = datetime.utcnow()
    event_count = 0
    for event in events:
        time_str = event.get("datetime")
        try:
            event_time = datetime.fromisoformat(time_str.replace("Z", ""))
            time_diff = (event_time - now).total_seconds() / 3600.0
            if 0 <= time_diff <= 6:
                prompt += f"- {event['event']} at {event['datetime']} (impact: {event['impact']})\n"
                event_count += 1
        except Exception:
            continue

    if event_count == 0:
        prompt += "- No events are occurring in the next 6 hours.\n"

    return prompt

def analyze_news_with_llm(prompt):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        reply = response.choices[0].message.content
        return json.loads(reply)
    except Exception as e:
        print(f"⚠️ Groq analysis failed: {e}")
        return {
            "safe": True,
            "sensitivity": 0,
            "reason": "LLM error or no response. Assuming safe."
        }

def news_filter():
    events = load_events()
    if not events:
        return {
            "safe": True,
            "sensitivity": 0,
            "reason": "No scheduled events found. Proceeding safely."
        }

    prompt = format_events_for_prompt(events)
    return analyze_news_with_llm(prompt)

if __name__ == "__main__":
    result = news_filter()
    print("\n📰 News Filter Result:", result)
