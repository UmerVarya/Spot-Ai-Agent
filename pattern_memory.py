import json
import os

PATTERN_MEMORY_FILE = "pattern_memory.json"

# === Load memory file ===
def load_pattern_memory():
    if os.path.exists(PATTERN_MEMORY_FILE):
        try:
            with open(PATTERN_MEMORY_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    else:
        return {}

# === Save memory file ===
def save_pattern_memory(memory):
    with open(PATTERN_MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

# === Recall confidence from memory ===
def recall_pattern_confidence(symbol, pattern_name):
    memory = load_pattern_memory()
    if symbol in memory and pattern_name in memory[symbol]:
        return memory[symbol][pattern_name].get("confidence", 5.0)
    return 5.0  # Neutral default

# === Update memory after trade outcome ===
def update_pattern_memory(symbol, pattern_name, outcome):
    memory = load_pattern_memory()
    if symbol not in memory:
        memory[symbol] = {}
    if pattern_name not in memory[symbol]:
        memory[symbol][pattern_name] = {"wins": 0, "losses": 0, "confidence": 5.0}

    if outcome == "win":
        memory[symbol][pattern_name]["wins"] += 1
    else:
        memory[symbol][pattern_name]["losses"] += 1

    wins = memory[symbol][pattern_name]["wins"]
    losses = memory[symbol][pattern_name]["losses"]
    total = wins + losses

    if total > 0:
        confidence = 5.0 + (wins - losses) / total * 5.0  # Range 0 to 10
        memory[symbol][pattern_name]["confidence"] = round(min(max(confidence, 0), 10), 2)

    save_pattern_memory(memory)
