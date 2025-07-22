import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from groq import Groq
import json
from dotenv import load_dotenv
import re

load_dotenv()

# === Fetch Crypto News ===
def fetch_crypto_news():
    url = "https://cryptopanic.com/news/rss/"
    events = []
    try:
        response = requests.get(url, timeout=10)
    except Exception as e:
        print("⚠️ Crypto news fetch failed:", e)
        return []
    content = response.content
    try:
        soup = BeautifulSoup(content, features="xml")
        items = soup.find_all("item")
        if not items:
            raise Exception("No items found in XML response")
    except Exception as e_xml:
        print("⚠️ Crypto news XML parse error:", e_xml)
        try:
            soup = BeautifulSoup(content, "html.parser")
            items = soup.find_all("item")
            if not items:
                raise Exception("No items found with HTML parser")
        except Exception as e_html:
            print("⚠️ Crypto news parse failed:", e_html)
            return []
    for item in items[:10]:
        events.append({
            "event": item.title.text,
            "datetime": datetime.utcnow().isoformat() + "Z",
            "impact": "medium"
        })
    return events

# === Fetch Macro News ===
def fetch_macro_news():
    url = "https://www.fxstreet.com/rss/news"
    events = []
    try:
        response = requests.get(url, timeout=10)
    except Exception as e:
        print("⚠️ Macro news fetch failed:", e)
        return []
    content = response.content
    try:
        soup = BeautifulSoup(content, features="xml")
        items = soup.find_all("item")
        if not items:
            raise Exception("No items found in XML response")
    except Exception as e_xml:
        print("⚠️ Macro news XML parse error:", e_xml)
        try:
            soup = BeautifulSoup(content, "html.parser")
            items = soup.find_all("item")
            if not items:
                raise Exception("No items found with HTML parser")
        except Exception as e_html:
            print("⚠️ Macro news parse failed:", e_html)
            return []
    for item in items[:10]:
        events.append({
            "event": item.title.text,
            "datetime": datetime.utcnow().isoformat() + "Z",
            "impact": "high" if ("Fed" in item.title.text or "inflation" in item.title.text) else "medium"
        })
    return events

# === Save to JSON ===
def save_events(events, path="news_events.json"):
    with open(path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"✅ Saved {len(events)} events to {path}")

# === Build LLM Prompt ===
def build_news_prompt(events):
    now = datetime.utcnow()
    filtered_events = []
    prompt = (
        "You're a macro risk analyst for crypto markets.\n"
        "Evaluate the following events for potential impact on crypto within the next 6 hours.\n"
        "Return a JSON array of objects like:\n"
        "[\n"
        "  {\"safe\": true, \"sensitivity\": 2, \"reason\": \"...\"},\n"
        "  {\"safe\": false, \"sensitivity\": 9, \"reason\": \"...\"}\n"
        "]\n\n"
        "Here is the list of events (in JSON):\n\n"
        "```json\n"
    )

    for event in events:
        try:
            event_time = datetime.fromisoformat(event["datetime"].replace("Z", ""))
            hours_until = (event_time - now).total_seconds() / 3600.0
            if hours_until >= -48:  # allow recent and future events
                filtered_events.append(event)
        except:
            continue

    prompt += json.dumps(filtered_events, indent=2)
    prompt += "\n```"
    return prompt.strip()

# === Analyze News ===
def analyze_news_with_llm(events):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = build_news_prompt(events)

    try:
        chat_completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a crypto macro risk analyst. Analyze impact of each event."},
                {"role": "user", "content": prompt}
            ]
        )
        raw_reply = chat_completion.choices[0].message.content
        print("\n🔍 Raw LLM Reply:\n", raw_reply)

        # Extract JSON array from response
        json_match = re.search(r"\[\s*{.*?}\s*\]", raw_reply, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            if not parsed:
                return {
                    "safe": True,
                    "sensitivity": 0,
                    "reason": "No actionable events detected."
                }

            high_risk = [e for e in parsed if not e["safe"] and e["sensitivity"] >= 7]
            return {
                "safe": len(high_risk) == 0,
                "sensitivity": max(e["sensitivity"] for e in parsed),
                "reason": high_risk[0]["reason"] if high_risk else "No major risk detected."
            }

        else:
            return {
                "safe": True,
                "sensitivity": 0,
                "reason": "Failed to extract JSON block from LLM reply."
            }

    except Exception as e:
        print("⚠️ Groq LLM analysis failed:", e)
        return {
            "safe": True,
            "sensitivity": 0,
            "reason": "Error during Groq analysis"
        }

# === Main Runner ===
def run_news_fetcher():
    print("🌐 Fetching Crypto and Macro News...")
    crypto_articles = fetch_crypto_news()
    macro_articles = fetch_macro_news()
    all_events = crypto_articles + macro_articles
    save_events(all_events)

    result = analyze_news_with_llm(all_events)
    print("\n🧠 Groq News Analysis:", result)

if __name__ == "__main__":
    run_news_fetcher()
