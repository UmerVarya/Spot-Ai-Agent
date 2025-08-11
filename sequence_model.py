"""
Sequence‑based model utilities for the Spot‑AI Agent.

This module implements a simple tree‑based sequence model using
scikit‑learn's ``RandomForestRegressor`` to predict the next‑bar return
from a rolling window of recent features. It provides lightweight
functionality compatible with environments lacking deep‑learning
libraries.
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
from log_utils import setup_logger

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    RandomForestRegressor = None  # type: ignore
    StandardScaler = None  # type: ignore
    joblib = None  # type: ignore
    SKLEARN_AVAILABLE = False

logger = setup_logger(__name__)

# File paths for persistence
ROOT_DIR = os.path.dirname(__file__)
SEQ_PKL = os.path.join(ROOT_DIR, "sequence_model.pkl")


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
    """Train a simple sequence‑to‑one regression model on historical data."""
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available; cannot train sequence model.")
        return
    if df is None or len(df) < window_size + 2:
        logger.warning("Not enough data to train sequence model.")
        return
    X, y = _build_sequences(df.reset_index(drop=True), window_size)
    if X.size == 0:
        logger.warning("Failed to construct sequences from data.")
        return
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_norm, y)
    try:
        joblib.dump({
            'model': model,
            'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_,
            'window_size': window_size,
            'feature_dim': X.shape[1]
        }, SEQ_PKL)
        logger.info("Sequence model trained and saved to %s", SEQ_PKL)
    except Exception as e:  # pragma: no cover - IO errors
        logger.warning("Failed to save sequence model: %s", e, exc_info=True)


def _load_sequence_model() -> Optional[dict]:
    if not os.path.exists(SEQ_PKL) or not SKLEARN_AVAILABLE:
        return None
    try:
        return joblib.load(SEQ_PKL)
    except Exception:  # pragma: no cover
        return None


def predict_next_return(window_df: pd.DataFrame) -> float:
    """Predict the next‑bar return given a recent window of data."""
    artefact = _load_sequence_model()
    if artefact is None:
        return 0.0
    model = artefact['model']
    mean = artefact['scaler_mean']
    scale = artefact['scaler_scale']
    window_size = artefact['window_size']
    if len(window_df) != window_size:
        return 0.0
    feature_cols = [col for col in window_df.columns if col not in {'timestamp', 'close'}]
    seq = window_df[feature_cols].to_numpy(dtype=float).flatten().reshape(1, -1)
    seq_norm = (seq - mean) / scale
    try:
        pred = model.predict(seq_norm)
        return float(pred[0])
    except Exception:  # pragma: no cover
        return 0.0
