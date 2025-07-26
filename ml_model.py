"""
Advanced machine learning utilities for the Spot AI Super Agent.

This module extends the original custom logistic regression by adding
support for scikit‑learn's robust LogisticRegression classifier when
available.  It scales features using StandardScaler and optionally
performs simple train/test splits to evaluate accuracy.  Model
parameters are saved to a JSON file in a backward‑compatible way so
that older agents relying on the manual logistic regression can still
load and use the model.

Functions
---------
train_model(iterations=200, learning_rate=0.1)
    Train a logistic regression model on the historical trade log.
predict_success_probability(score, confidence, session, btc_d, fg, sentiment_conf, pattern)
    Estimate the probability of a trade succeeding based on current
    features using the saved model.

Notes
-----
If scikit‑learn is installed, the training routine will default to
using ``sklearn.linear_model.LogisticRegression`` with a built‑in
solver and L2 regularisation.  Otherwise it falls back to a manual
gradient descent implementation similar to the original code.  The
resulting model type is stored in the ``ml_model.json`` file under the
``model_type`` key along with the feature names, scaling parameters
and coefficients.
"""

from __future__ import annotations

import os
import json
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    # Try to import scikit‑learn components; if unavailable, fallback to manual
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    LogisticRegression = None  # type: ignore
    StandardScaler = None  # type: ignore
    train_test_split = None  # type: ignore
    SKLEARN_AVAILABLE = False

try:
    # Optionally import historical confidence calculator
    from confidence import calculate_historical_confidence  # noqa: F401
except Exception:
    calculate_historical_confidence = None  # type: ignore

# Path constants
ROOT_DIR = os.path.dirname(__file__)
LOG_FILE = os.path.join(ROOT_DIR, "trade_learning_log.csv")
MODEL_FILE = os.path.join(ROOT_DIR, "ml_model.json")


def _extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and label vector y from the learning log.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of logged trades read from ``trade_learning_log.csv``.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Binary labels (1 for success, 0 for failure).
    """
    session_map = {"Asia": 0, "Europe": 1, "US": 2, "New York": 2, "unknown": 3}
    success_outcomes = {"tp1", "tp2", "tp3", "tp4", "tp4_sl", "win"}
    feature_list: List[List[float]] = []
    labels: List[int] = []
    for _, row in df.iterrows():
        try:
            score = float(row.get("score", 0))
            conf = float(row.get("confidence", 0))
            session = row.get("session", "unknown")
            btc_dom = float(row.get("btc_dominance", 0))
            fg = float(row.get("fear_greed", 0))
            # Sentiment confidence may be missing or a string; normalise to 0–10
            sent_conf = row.get("sentiment_confidence", row.get("sentiment", 5))
            try:
                sent_conf_val = float(sent_conf)
            except Exception:
                sent_conf_val = 5.0
            pattern = row.get("pattern", "none")
            pattern_len = len(str(pattern))
            session_id = session_map.get(str(session), 3)
            features = [
                score,  # normalised technical score
                conf,   # final blended confidence
                session_id,
                btc_dom / 100.0,
                fg / 100.0,
                sent_conf_val / 10.0,
                pattern_len / 10.0,
            ]
            feature_list.append(features)
            outcome = str(row.get("outcome", "loss")).lower()
            labels.append(1 if outcome in success_outcomes else 0)
        except Exception:
            continue
    X = np.array(feature_list, dtype=float)
    y = np.array(labels, dtype=float)
    return X, y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def train_model(iterations: int = 200, learning_rate: float = 0.1) -> None:
    """Train a logistic regression model on the trade learning log.

    When scikit‑learn is available, use its ``LogisticRegression`` with
    L2 regularisation and balanced class weights to handle class
    imbalance.  Otherwise fall back to a manual gradient descent
    implementation similar to the original code.  After training, the
    model parameters and scaling statistics are saved to ``ml_model.json``.
    The function prints a brief summary of model accuracy if a train/test
    split is possible.
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
    model_data: dict = {}
    if SKLEARN_AVAILABLE:
        # Use scikit‑learn logistic regression
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        # Train/test split for simple performance check
        if train_test_split is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_norm, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X_norm, y  # type: ignore
            X_test, y_test = X_norm, y  # type: ignore
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
        clf.fit(X_train, y_train)
        # Evaluate accuracy if test split exists
        acc = None
        try:
            y_pred = clf.predict(X_test)
            acc = float((y_pred == y_test).mean())
        except Exception:
            pass
        if acc is not None:
            print(f"✅ ML model (sklearn) trained. Test accuracy: {acc:.2%}")
        else:
            print("✅ ML model (sklearn) trained.")
        # Prepare model data for saving
        model_data = {
            "model_type": "sklearn",
            "intercept": clf.intercept_.tolist(),
            "coefficients": clf.coef_.tolist()[0],
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "feature_names": [
                "score", "confidence", "session_id", "btc_dom",
                "fear_greed", "sent_conf", "pattern_len"
            ],
        }
    else:
        # Manual logistic regression training via gradient descent
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-8
        X_norm = (X - mu) / sigma
        X_aug = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])
        weights = np.zeros(X_aug.shape[1])
        for _ in range(iterations):
            z = X_aug @ weights
            h = _sigmoid(z)
            gradient = (X_aug.T @ (h - y)) / len(y)
            weights -= learning_rate * gradient
        print("✅ ML model (manual) trained.")
        model_data = {
            "model_type": "manual",
            "weights": weights.tolist(),
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
            "feature_names": [
                "score", "confidence", "session_id", "btc_dom",
                "fear_greed", "sent_conf", "pattern_len"
            ],
        }
    # Save model parameters to file
    try:
        with open(MODEL_FILE, "w") as f:
            json.dump(model_data, f, indent=2)
        print(f"✅ Model saved to {MODEL_FILE}")
    except Exception as e:
        print(f"⚠️ Failed to save ML model: {e}")


def _load_model() -> dict:
    """Load the trained model from disk.  Returns an empty dict on failure."""
    if not os.path.exists(MODEL_FILE):
        return {}
    try:
        with open(MODEL_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _prepare_feature_vector(score: float, confidence: float, session: str, btc_d: float, fg: float, sentiment_conf: float, pattern: str) -> List[float]:
    session_map = {"Asia": 0, "Europe": 1, "US": 2, "New York": 2, "unknown": 3}
    session_id = session_map.get(session, 3)
    pattern_len = len(str(pattern))
    return [
        score,
        confidence,
        session_id,
        btc_d / 100.0,
        fg / 100.0,
        sentiment_conf / 10.0,
        pattern_len / 10.0,
    ]


def predict_success_probability(
    score: float,
    confidence: float,
    session: str,
    btc_d: float,
    fg: float,
    sentiment_conf: float,
    pattern: str
) -> float:
    """Predict the success probability for a potential trade.

    Parameters
    ----------
    score : float
        Technical score from signal evaluation.
    confidence : float
        Blended confidence after LLM evaluation and adjustments.
    session : str
        Market session ("Asia", "Europe", "US", etc.).
    btc_d : float
        BTC dominance percentage.
    fg : float
        Fear & Greed index.
    sentiment_conf : float
        Macro sentiment confidence on a 0–10 scale.
    pattern : str
        Name of the detected pattern.

    Returns
    -------
    float
        Predicted probability in [0, 1].  Returns 0.5 if model is
        unavailable or cannot be evaluated.
    """
    model = _load_model()
    if not model:
        return 0.5
    x = np.array(_prepare_feature_vector(score, confidence, session, btc_d, fg, sentiment_conf, pattern))
    model_type = model.get("model_type", "manual")
    try:
        if model_type == "sklearn":
            # Apply scaling
            mean = np.array(model.get("scaler_mean"))
            scale = np.array(model.get("scaler_scale"))
            x_norm = (x - mean) / scale
            # Compute linear combination
            intercept = np.array(model.get("intercept"))
            coef = np.array(model.get("coefficients"))
            z = intercept + np.dot(coef, x_norm)
            prob = float(_sigmoid(np.array([z]))[0])
            return prob
        else:
            # Manual model structure: weights include bias as first element
            mu = np.array(model.get("mu"))
            sigma = np.array(model.get("sigma"))
            weights = np.array(model.get("weights"))
            x_norm = (x - mu) / (sigma + 1e-8)
            x_aug = np.hstack([1.0, x_norm])
            z = float(x_aug @ weights)
            prob = float(_sigmoid(np.array([z]))[0])
            return prob
    except Exception:
        # Fallback neutral probability if something goes wrong
        return 0.5
