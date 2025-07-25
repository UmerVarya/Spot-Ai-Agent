"""
Machine learning utilities for the Spot AI Super Agent.

This module implements a simple logistic regression model using
gradient descent to predict the probability that a trade will be
successful (e.g., reaching TP levels instead of stop‑loss).  It
trains on past trade outcomes recorded in ``trade_learning_log.csv``
and saves model parameters and feature scaling values to disk.

Functions
---------
train_model()
    Train a logistic regression model on existing trade logs.
predict_success_probability(features)
    Given a feature vector for a potential trade, return the
    estimated probability of success using the trained model.

Note
----
This implementation is intentionally lightweight to avoid external
dependencies like scikit‑learn.  It uses numpy for matrix
operations and implements logistic regression from scratch.  If
scikit‑learn becomes available, you can replace the training and
prediction functions with the built‑in classifiers for better
performance and reliability.
"""

import os
import json
from typing import List, Tuple

import numpy as np
import pandas as pd

from .confidence import calculate_historical_confidence  # for potential feature expansion

# Path constants
ROOT_DIR = os.path.dirname(__file__)
LOG_FILE = os.path.join(ROOT_DIR, "trade_learning_log.csv")
MODEL_FILE = os.path.join(ROOT_DIR, "ml_model.json")


def _extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the learning log DataFrame into a feature matrix and labels.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of logged trades read from ``trade_learning_log.csv``.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Binary labels (1 for success, 0 for failure).
    """
    # Map session strings to integers
    session_map = {"Asia": 0, "Europe": 1, "US": 2, "New York": 2, "unknown": 3}
    # Determine outcomes considered successful.  You may adjust this list
    # depending on how you categorise early exits or TP4 modes.
    success_outcomes = {"tp1", "tp2", "tp3", "tp4", "tp4_sl", "win"}

    feature_list = []
    labels = []
    for _, row in df.iterrows():
        # Skip rows with missing mandatory fields
        try:
            score = float(row.get("score", 0))
            confidence = float(row.get("confidence", 0))
            session = row.get("session", "unknown")
            btc_dom = float(row.get("btc_dominance", 0))
            fg = float(row.get("fear_greed", 0))
            sentiment_conf = float(row.get("sentiment_confidence", 5)) if isinstance(row.get("sentiment_confidence"), (int, float)) else 5.0
            # Encode pattern type as simple length of pattern name (fallback)
            pattern = row.get("pattern", "none")
            pattern_len = len(str(pattern))
            session_id = session_map.get(str(session), 3)

            features = [score, confidence, session_id, btc_dom / 100.0, fg / 100.0, sentiment_conf / 10.0, pattern_len / 10.0]
            feature_list.append(features)
            outcome = str(row.get("outcome", "loss")).lower()
            labels.append(1 if outcome in success_outcomes else 0)
        except Exception:
            continue
    X = np.array(feature_list, dtype=float)
    y = np.array(labels, dtype=float)
    return X, y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def train_model(iterations: int = 200, learning_rate: float = 0.1) -> None:
    """Train a logistic regression model on the trade learning log.

    The resulting model parameters and feature scaling statistics are
    saved to ``ml_model.json`` in the project directory.  If the log
    file is missing or has insufficient data, no model will be saved.

    Parameters
    ----------
    iterations : int, optional
        Number of gradient descent iterations.  Default is 200.
    learning_rate : float, optional
        Gradient descent step size.  Default is 0.1.
    """
    if not os.path.exists(LOG_FILE):
        print("⚠️ No trade learning log found. Cannot train ML model.")
        return
    try:
        df = pd.read_csv(LOG_FILE, engine="python", on_bad_lines="skip")
    except Exception as e:
        print(f"⚠️ Failed to read learning log: {e}")
        return
    if len(df) < 20:
        print("⚠️ Not enough data to train ML model. Need at least 20 trades.")
        return
    X, y = _extract_features(df)
    if X.size == 0:
        print("⚠️ No valid training samples extracted.")
        return
    # Normalise features (standard score)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8  # avoid division by zero
    X_norm = (X - mu) / sigma
    # Add bias term
    X_aug = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])
    # Initialise weights randomly
    weights = np.zeros(X_aug.shape[1])
    # Gradient descent
    for _ in range(iterations):
        z = X_aug @ weights
        h = _sigmoid(z)
        gradient = (X_aug.T @ (h - y)) / len(y)
        weights -= learning_rate * gradient
    # Save model
    model_data = {
        "weights": weights.tolist(),
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "feature_names": [
            "score", "confidence", "session_id", "btc_dom", "fear_greed", "sentiment_conf", "pattern_len"
        ]
    }
    with open(MODEL_FILE, "w") as f:
        json.dump(model_data, f, indent=2)
    print("✅ ML model trained and saved.")


def _load_model() -> dict:
    if not os.path.exists(MODEL_FILE):
        return {}
    try:
        with open(MODEL_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _prepare_feature_vector(score: float, confidence: float, session: str, btc_d: float, fg: float, sentiment_conf: float, pattern: str) -> List[float]:
    session_map = {"Asia": 0, "Europe": 1, "US": 2, "New York": 2}
    session_id = session_map.get(session, 3)
    pattern_len = len(str(pattern))
    feat = [score, confidence, session_id, btc_d / 100.0, fg / 100.0, sentiment_conf / 10.0, pattern_len / 10.0]
    return feat


def predict_success_probability(score: float, confidence: float, session: str, btc_d: float, fg: float, sentiment_conf: float, pattern: str) -> float:
    """Predict the success probability for a potential trade.

    If no model is trained yet, returns 0.5 (neutral).  Otherwise the
    stored model parameters are used to compute the logistic function.

    Parameters
    ----------
    score : float
        Normalised technical score from the signal evaluation.
    confidence : float
        Blended confidence before ML adjustment.
    session : str
        Current market session ("Asia", "Europe", "US").
    btc_d : float
        Current BTC dominance percentage.
    fg : float
        Fear & Greed index.
    sentiment_conf : float
        Macro sentiment confidence (0–10 scale).
    pattern : str
        Pattern name or description.

    Returns
    -------
    float
        Probability of success in [0, 1].
    """
    model = _load_model()
    if not model:
        return 0.5  # neutral if model not trained
    weights = np.array(model.get("weights"))
    mu = np.array(model.get("mu"))
    sigma = np.array(model.get("sigma"))
    x = np.array(_prepare_feature_vector(score, confidence, session, btc_d, fg, sentiment_conf, pattern))
    # standardise
    x_norm = (x - mu) / (sigma + 1e-8)
    x_aug = np.hstack([1.0, x_norm])
    z = float(x_aug @ weights)
    prob = float(_sigmoid(np.array([z]))[0])
    return prob
