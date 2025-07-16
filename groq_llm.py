# groq_llm.py

import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set this in your environment
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def get_llm_judgment(prompt, temperature=0.4, max_tokens=500):
    """
    Sends a prompt to Groq LLM (LLaMA3) and returns the reasoning or decision.
    """
    try:
        data = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a highly experienced crypto trader assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(GROQ_API_URL, headers=HEADERS, json=data)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"❌ LLM request failed: {response.status_code}, {response.text}")
            return "LLM error: Unable to generate response."

    except Exception as e:
        print(f"❌ LLM Exception: {e}")
        return "LLM error: Exception occurred."
