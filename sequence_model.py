"""
Sequence‑based model utilities for the Spot‑AI Agent.

Deep learning libraries (TensorFlow/PyTorch) are not available in the
deployment environment, so this module provides a simple alternative
for modelling short‑term price dynamics.  It uses scikit‑learn's
``RandomForestRegressor`` to forecast the next‑minute return based on
a rolling window of recent features.  While not as expressive as
LSTM/GRU architectures, a tree‑based ensemble can still capture
non‑linear relationships within the window and offer a practical
approximation for intraday prediction.

The sequence model operates independently of the main classification
model in ``ml_model.py``.  After training, the agent can query
``predict_next_return`` to obtain a numeric expectation for the next
minute's percentage change, which can be used to refine entry timing
or adjust confidence.

Example usage::

    from sequence_model import train_sequence_model, predict_next_return
    # Train on historical OHLCV data with indicators
    train_sequence_model(df, window_size=10)
    # Predict the next return given the most recent window
    next_ret = predict_next_return(recent_df)

Note: The model is saved to ``sequence_model.pkl`` in the module
directory and automatically loaded when making predictions.
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:
    RandomForestRegressor = None  # type: ignore
    StandardScaler = None  # type: ignore
    joblib = None  # type: ignore
    SKLEARN_AVAILABLE = False

# File paths for persistence
ROOT_DIR = os.path.dirname(__file__)
SEQ_PKL = os.path.join(ROOT_DIR, "sequence_model.pkl")
SEQ_JSON = os.path.join(ROOT_DIR, "sequence_model_meta.json")


def _build_sequences(df: pd.DataFrame, window_size: int) -> (np.ndarray, np.ndarray):
    """Convert a DataFrame into overlapping sequences and target returns."""
    returns = df['close'].pct_change().fillna(0.0).to_numpy()
    feature_cols = [col for col in df.columns if col not in {'timestamp', 'close'}]
    feature_matrix = df[feature_cols].to_numpy(dtype=float)
    X: List[np.ndarray] = []
    y: List[float] = []
    for i in range(window_size, len(df) - 1):
        seq_features = feature_matrix[i - window_size:i].flatten()
        X.append(seq_features)
        y.append(returns[i + 1])
    return np.array(X), np.array(y)


def train_sequence_model(df: pd.DataFrame, window_size: int = 10) -> None:
    """
    Train a simple sequence‑to‑one regression model on historical data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least a 'close' column and any number
        of additional indicator columns.  Each row should represent a
        single bar (e.g. 1‑minute candle).
    window_size : int, optional
        Number of past bars to include in each input sequence.  The
        default of 10 uses the past 10 minutes of data.
    """
    if not SKLEARN_AVAILABLE:
        print("⚠️ scikit‑learn not available; cannot train sequence model.")
        return
    if df is None or len(df) < window_size + 2:
        print("⚠️ Not enough data to train sequence model.")
        return
    X, y = _build_sequences(df.reset_index(drop=True), window_size)
    if X.size == 0:
        print("⚠️ Failed to construct sequences from data.")
        return
    # Standardise features to zero mean and unit variance
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_norm, y)
    try:
        joblib.dump({'model': model, 'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_, 'window_size': window_size, 'feature_dim': X.shape[1]}, SEQ_PKL)
        print(f"✅ Sequence model trained and saved to {SEQ_PKL}")
    except Exception as e:
        print(f"⚠️ Failed to save sequence model: {e}")


def _load_sequence_model() -> Optional[dict]:
    if not os.path.exists(SEQ_PKL) or not SKLEARN_AVAILABLE:
        return None
    try:
        return joblib.load(SEQ_PKL)
    except Exception:
        return None


def predict_next_return(window_df: pd.DataFrame) -> float:
    """
    Predict the next‑bar return given a recent window of data.

    Parameters
    ----------
    window_df : pandas.DataFrame
        A DataFrame containing the most recent `window_size` bars used
        during training.  It must include the same columns (except the
        target) and be ordered chronologically.

    Returns
    -------
    float
        Predicted fractional return (e.g. 0.001 for 0.1 %).  If the
        model is unavailable or an error occurs, returns 0.0.
    """
    artefact = _load_sequence_model()
    if artefact is None:
        return 0.0
    model = artefact['model']
    mean = artefact['scaler_mean']
    scale = artefact['scaler_scale']
    window_size = artefact['window_size']
    # Ensure the window length matches the model
    if len(window_df) != window_size:
        return 0.0
    feature_cols = [col for col in window_df.columns if col not in {'timestamp', 'close'}]
    seq = window_df[feature_cols].to_numpy(dtype=float).flatten().reshape(1, -1)
    seq_norm = (seq - mean) / scale
    try:
        pred = model.predict(seq_norm)
        return float(pred[0])
    except Exception:
        return 0.0
