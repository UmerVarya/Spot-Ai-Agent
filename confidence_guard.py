import pandas as pd
import os

LEARNING_LOG = "trade_learning_log.csv"

def get_adaptive_conf_threshold():
    if not os.path.exists(LEARNING_LOG):
        return 6.0  # Default if no data

    df = pd.read_csv(LEARNING_LOG)
    if len(df) < 10:
        return 6.0  # Not enough history to adapt

    recent = df.tail(25)
    wins = recent[recent["outcome"] == "win"]
    win_rate = len(wins) / len(recent)

    avg_conf = recent["confidence"].mean()

    # Base threshold is 6.0 — adapt up or down
    if win_rate >= 0.7:
        return max(5.0, avg_conf - 0.5)
    elif win_rate <= 0.4:
        return min(7.5, avg_conf + 0.5)
    else:
        return round(avg_conf, 2)
