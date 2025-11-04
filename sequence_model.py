"""Advanced sequence modelling utilities for the Spot‑AI Agent.

The module now supports multiple regressors, automated model selection,
validation metrics, and interpretability diagnostics so that the
time-series pipeline can be audited and tuned like the primary
classifier.
"""

import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from log_utils import setup_logger

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception:  # pragma: no cover - optional dependency
    HistGradientBoostingRegressor = None  # type: ignore

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance
    import joblib  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    RandomForestRegressor = None  # type: ignore
    GradientBoostingRegressor = None  # type: ignore
    HistGradientBoostingRegressor = None  # type: ignore
    StandardScaler = None  # type: ignore
    TimeSeriesSplit = None  # type: ignore
    mean_squared_error = None  # type: ignore
    mean_absolute_error = None  # type: ignore
    r2_score = None  # type: ignore
    clone = None  # type: ignore
    permutation_importance = None  # type: ignore
    joblib = None  # type: ignore
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None  # type: ignore

logger = setup_logger(__name__)

_training_lock = threading.Lock()
_training_thread: Optional[threading.Thread] = None

# File paths for persistence
ROOT_DIR = os.path.dirname(__file__)
SEQ_PKL = os.path.join(ROOT_DIR, "sequence_model.pkl")


@dataclass
class SequenceTrainingDiagnostics:
    """Diagnostics returned when training the sequence model."""

    model_type: str
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    window_size: int
    samples: int


def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if mean_squared_error is not None:
        try:
            mse = float(mean_squared_error(y_true, y_pred))
            metrics['mse'] = mse
            metrics['rmse'] = float(np.sqrt(mse))
        except Exception:
            metrics['mse'] = float('nan')
            metrics['rmse'] = float('nan')
    if mean_absolute_error is not None:
        try:
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        except Exception:
            metrics['mae'] = float('nan')
    if r2_score is not None:
        try:
            metrics['r2'] = float(r2_score(y_true, y_pred))
        except Exception:
            metrics['r2'] = float('nan')
    try:
        metrics['mape'] = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))))
    except Exception:
        metrics['mape'] = float('nan')
    return metrics


def _evaluate_candidate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    splitter: "TimeSeriesSplit",
) -> float:
    scores: List[float] = []
    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        try:
            estimator = clone(model) if clone is not None else model.__class__(**model.get_params())
        except Exception:
            estimator = model
        estimator.fit(X_train, y_train)
        preds = np.asarray(estimator.predict(X_test), dtype=float)
        rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
        scores.append(-rmse)
    return float(np.mean(scores)) if scores else float('-inf')


def _compute_feature_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    base_features: List[str],
    window_size: int,
) -> Dict[str, float]:
    if permutation_importance is None:
        return {}
    try:
        perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    except Exception:
        return {}
    try:
        importances = perm.importances_mean.reshape(window_size, len(base_features))
    except Exception:
        importances = np.zeros((window_size, len(base_features)))
    aggregated = np.abs(importances).mean(axis=0)
    total = float(np.sum(aggregated))
    if total <= 0:
        return {}
    importance_map = {
        feature: float(value / total) for feature, value in zip(base_features, aggregated)
    }
    return dict(sorted(importance_map.items(), key=lambda kv: kv[1], reverse=True))


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


def train_sequence_model(df: pd.DataFrame, window_size: int = 10) -> Optional[SequenceTrainingDiagnostics]:
    """Train an ensemble of sequence models and persist the best performer."""
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available; cannot train sequence model.")
        return None
    if df is None or len(df) < window_size + 2:
        logger.warning("Not enough data to train sequence model.")
        return None
    X, y = _build_sequences(df.reset_index(drop=True), window_size)
    if X.size == 0:
        logger.warning("Failed to construct sequences from data.")
        return None
    split_idx = max(window_size, int(len(X) * 0.8))
    split_idx = min(split_idx, len(X) - 1)
    if split_idx <= 0:
        logger.warning("Insufficient samples after sequence construction.")
        return None
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    if len(X_val) == 0:
        X_val = X_train[-window_size:]
        y_val = y_train[-window_size:]
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)

    models: Dict[str, Any] = {
        'random_forest': RandomForestRegressor(n_estimators=200, max_depth=7, random_state=42),
    }
    if GradientBoostingRegressor is not None:
        models['gradient_boosting'] = GradientBoostingRegressor(random_state=42)
    if HistGradientBoostingRegressor is not None:
        models['hist_gradient'] = HistGradientBoostingRegressor(random_state=42)
    if XGBRegressor is not None:
        models['xgboost'] = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            objective='reg:squarederror',
        )
    if LGBMRegressor is not None:
        models['lightgbm'] = LGBMRegressor(
            n_estimators=300,
            num_leaves=63,
            learning_rate=0.05,
            max_depth=-1,
        )

    splitter = None
    if TimeSeriesSplit is not None and len(X_train_norm) > 3:
        n_splits = min(5, max(2, len(X_train_norm) // max(1, window_size // 2 or 1)))
        if n_splits >= len(X_train_norm):
            n_splits = len(X_train_norm) - 1
        if n_splits >= 2:
            splitter = TimeSeriesSplit(n_splits=n_splits)

    best_model_name = ''
    best_estimator: Optional[Any] = None
    best_score = float('-inf')
    val_predictions: Optional[np.ndarray] = None
    for name, template in models.items():
        try:
            estimator = clone(template) if clone is not None else template.__class__(**template.get_params())
        except Exception:
            estimator = template
        estimator.fit(X_train_norm, y_train)
        val_pred = np.asarray(estimator.predict(X_val_norm), dtype=float)
        val_rmse = -float(np.sqrt(np.mean((val_pred - y_val) ** 2))) if len(y_val) else float('-inf')
        cv_score = val_rmse
        if splitter is not None:
            try:
                cv_score = _evaluate_candidate(template, X_train_norm, y_train, splitter)
            except Exception as exc:
                logger.warning("Time-series CV failed for %s: %s", name, exc, exc_info=True)
        combined_score = 0.5 * cv_score + 0.5 * val_rmse
        logger.info("Sequence model %s -> CV score %.6f | validation score %.6f", name, cv_score, val_rmse)
        if combined_score > best_score:
            best_score = combined_score
            best_model_name = name
            best_estimator = estimator
            val_predictions = val_pred

    if best_estimator is None:
        logger.warning("Failed to select a sequence model candidate.")
        return None

    metrics = _compute_regression_metrics(y_val, val_predictions if val_predictions is not None else y_val)
    scaler_full = StandardScaler()
    X_full_norm = scaler_full.fit_transform(X)
    try:
        final_model = clone(best_estimator) if clone is not None else best_estimator.__class__(**best_estimator.get_params())
    except Exception:
        final_model = best_estimator
    final_model.fit(X_full_norm, y)
    base_features = [col for col in df.columns if col not in {'timestamp', 'close'}]
    feature_importance = _compute_feature_importance(final_model, X_full_norm, y, base_features, window_size)

    artefact = {
        'model': final_model,
        'scaler_mean': scaler_full.mean_,
        'scaler_scale': scaler_full.scale_,
        'window_size': window_size,
        'feature_dim': X.shape[1],
        'model_type': best_model_name,
        'metrics': metrics,
        'feature_names': base_features,
        'feature_importance': feature_importance,
    }
    try:
        joblib.dump(artefact, SEQ_PKL)
        logger.info("Sequence model (%s) trained with RMSE %.6f and saved to %s", best_model_name, metrics.get('rmse', float('nan')), SEQ_PKL)
    except Exception as e:  # pragma: no cover - IO errors
        logger.warning("Failed to save sequence model: %s", e, exc_info=True)
    return SequenceTrainingDiagnostics(
        model_type=best_model_name,
        metrics=metrics,
        feature_importance=feature_importance,
        window_size=window_size,
        samples=len(X),
    )


def _load_sequence_model() -> Optional[dict]:
    if not os.path.exists(SEQ_PKL) or not SKLEARN_AVAILABLE:
        return None
    try:
        return joblib.load(SEQ_PKL)
    except Exception:  # pragma: no cover
        return None


def load_sequence_model_report() -> Dict[str, Any]:
    """Return persisted metadata for the trained sequence model."""

    artefact = _load_sequence_model()
    if not artefact:
        return {}
    report = {
        'model_type': artefact.get('model_type'),
        'metrics': artefact.get('metrics', {}),
        'window_size': artefact.get('window_size'),
        'feature_importance': artefact.get('feature_importance', {}),
    }
    return report


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


def schedule_sequence_model_training(
    df: pd.DataFrame,
    *,
    window_size: int = 10,
) -> bool:
    """Kick off background training for the sequence model if needed.

    Returns ``True`` when a new asynchronous training job was scheduled.
    If training is already running, the artefact exists, or prerequisites are
    missing (e.g., insufficient rows or scikit-learn unavailable), ``False`` is
    returned instead.
    """

    if not SKLEARN_AVAILABLE:
        return False
    if df is None or len(df) < window_size + 2:
        return False
    if os.path.exists(SEQ_PKL):
        return False

    data_copy = df.copy(deep=True)

    def _worker() -> None:
        try:
            train_sequence_model(data_copy, window_size=window_size)
        except Exception:
            logger.exception("Asynchronous sequence model training failed.")
        finally:
            with _training_lock:
                global _training_thread
                _training_thread = None

    with _training_lock:
        global _training_thread
        if _training_thread is not None and _training_thread.is_alive():
            return False
        _training_thread = threading.Thread(
            target=_worker,
            name="sequence-model-trainer",
            daemon=True,
        )
        _training_thread.start()

    logger.info(
        "Asynchronous sequence model training scheduled (window=%d, samples=%d).",
        window_size,
        len(data_copy),
    )
    return True


def is_sequence_model_training() -> bool:
    """Return ``True`` while an asynchronous training job is active."""

    with _training_lock:
        thread = _training_thread
    return bool(thread and thread.is_alive())
