"""
Advanced machine‑learning utilities for the Spot‑AI Super Agent.

This module supersedes the original logistic regression by introducing
an extensible training framework supporting **ensemble models**
(random forests, gradient boosting) and **cross‑validation**.  It
automatically selects the best performing classifier via grid
search and stores both the fitted model (via joblib) and the feature
scaling parameters.  If scikit‑learn is unavailable, it falls
back to a manual logistic regression implementation for
compatibility.

Key enhancements
----------------

* **Ensemble models** – The training routine can fit a random
  forest classifier and a gradient boosting classifier in addition to
  logistic regression.  It uses cross‑validation to select optimal
  hyper‑parameters and chooses the model with the highest mean
  validation accuracy.
* **Cross‑validation & hyper‑parameter tuning** – A grid search
  across model types and hyper‑parameters is performed using
  ``GridSearchCV`` with stratified splits to mitigate overfitting.
* **Saved artefacts** – The best model is saved to ``ml_model.pkl``
  via joblib.  Scaling statistics and the selected model type are
  stored in ``ml_model.json``.  Manual logistic regression
  coefficients remain supported for environments without scikit‑learn.
* **Predictive API** – ``predict_success_probability`` loads the
  appropriate artefacts and returns a probability between 0 and 1 for
  a prospective trade.  Tree‑based models automatically output a
  probability from their ``predict_proba`` method.
* **Non‑linear microstructure features** – The pipeline augments
  imbalance inputs with squared and extreme‑value flags so linear
  models can still react to threshold effects observed in order flow.
* **Balanced training & drift monitoring** – Sample weighting, feature
  selection (RFE/PCA), rolling time‑series validation and drift
  detection ensure the ensemble adapts to regime shifts while handling
  class imbalance.

These changes allow the agent to capture non‑linear interactions
between features and to adapt more flexibly to the historical trade
performance while remaining backward compatible with older models.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Mapping, Callable
import inspect

import numpy as np
import pandas as pd
from datetime import datetime
from log_utils import setup_logger
from trade_storage import TRADE_HISTORY_FILE, load_trade_history_df

try:
    # Core sklearn components used for modelling and preprocessing
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import (
        GridSearchCV,
        StratifiedKFold,
        StratifiedShuffleSplit,
        TimeSeriesSplit,
    )
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import RFECV
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
        brier_score_loss,
        log_loss,
    )
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance
    from sklearn.utils.class_weight import compute_class_weight
    import joblib  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:
    LogisticRegression = None  # type: ignore
    RandomForestClassifier = None  # type: ignore
    GradientBoostingClassifier = None  # type: ignore
    StandardScaler = None  # type: ignore
    GridSearchCV = None  # type: ignore
    StratifiedKFold = None  # type: ignore
    StratifiedShuffleSplit = None  # type: ignore
    TimeSeriesSplit = None  # type: ignore
    PCA = None  # type: ignore
    RFECV = None  # type: ignore
    CalibratedClassifierCV = None  # type: ignore
    balanced_accuracy_score = None  # type: ignore
    precision_recall_fscore_support = None  # type: ignore
    roc_auc_score = None  # type: ignore
    brier_score_loss = None  # type: ignore
    log_loss = None  # type: ignore
    clone = None  # type: ignore
    permutation_importance = None  # type: ignore
    compute_class_weight = None  # type: ignore
    joblib = None  # type: ignore
    accuracy_score = None  # type: ignore
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None  # type: ignore

try:
    from catboost import CatBoostClassifier  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None  # type: ignore

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
#
# ``ML_MODEL_JSON`` stores the model metadata, scaling parameters and manual
# coefficients.  ``ML_MODEL_PKL`` stores a pickled sklearn model when
# available.  Keeping both files allows backwards compatibility and
# seamless upgrades.
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(__file__)
LOG_FILE = TRADE_HISTORY_FILE
MODEL_JSON = os.path.join(ROOT_DIR, "ml_model.json")
MODEL_PKL = os.path.join(ROOT_DIR, "ml_model.pkl")


@dataclass
class TrainingDiagnostics:
    """Rich training diagnostics returned by :func:`train_model`."""

    model_type: str
    best_params: Dict[str, Any]
    validation_metrics: Dict[str, float]
    calibrated: bool
    calibration_method: Optional[str]
    feature_importance: Dict[str, float]
    samples: int


_FEATURE_FALLBACKS: Dict[str, float] = {
    'score': 0.0,
    'confidence': 0.0,
    'session_id': 3.0,
    'btc_dom': 0.0,
    'fear_greed': 0.5,
    'sent_conf': 0.5,
    'sent_bias': 0.0,
    'pattern_len': 0.0,
    'volatility': 0.0,
    'htf_trend': 0.0,
    'order_imbalance': 0.0,
    'order_imbalance_sq': 0.0,
    'extreme_imbalance_flag': 0.0,
    'order_flow_score': 0.0,
    'order_flow_flag': 0.0,
    'cvd': 0.0,
    'cvd_change': 0.0,
    'taker_buy_ratio': 0.0,
    'trade_imbalance': 0.0,
    'aggressive_trade_rate': 0.0,
    'spoofing_intensity': 0.0,
    'spoofing_alert': 0.0,
    'volume_ratio': 0.0,
    'price_change_pct': 0.0,
    'spread_bps': 0.0,
    'macro_indicator': 0.0,
    'macd': 0.0,
    'rsi': 0.5,
    'sma': 0.0,
    'atr': 0.0,
    'volume': 0.0,
    'macd_rsi': 0.0,
    'llm_decision': 1.0,
    'llm_confidence': 0.5,
    'time_since_last': 0.0,
    'recent_win_rate': 0.5,
}


_FEATURE_NORMALIZATION: Dict[str, Callable[[float], float]] = {
    'btc_dom': lambda v: v / 100.0,
    'fear_greed': lambda v: v / 100.0,
    'sent_conf': lambda v: v / 10.0,
    'pattern_len': lambda v: v / 10.0,
    'volatility': lambda v: v / 100.0,
    'htf_trend': lambda v: v / 100.0,
    'order_imbalance': lambda v: v / 100.0,
    'macro_indicator': lambda v: v / 100.0,
    'macd': lambda v: v / 100.0,
    'rsi': lambda v: v / 100.0,
    'sma': lambda v: v / 100.0,
    'atr': lambda v: v / 100.0,
    'volume': lambda v: v / 1_000_000.0,
    'llm_confidence': lambda v: v / 10.0,
    'time_since_last': lambda v: v / 24.0,
    'spread_bps': lambda v: v / 100.0,
}


def _normalise_feature(name: str, value: Any) -> float:
    """Convert raw feature values into the scale used during training."""

    if value is None:
        return 0.0
    if name == 'sent_bias':
        token = str(value).strip().lower()
        if token == 'bullish':
            return 1.0
        if token == 'bearish':
            return -1.0
        try:
            value = float(token)
        except Exception:
            return 0.0
    elif isinstance(value, bool):
        value = 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number):
        return 0.0
    normaliser = _FEATURE_NORMALIZATION.get(name)
    if normaliser is not None:
        try:
            return float(normaliser(number))
        except Exception:
            return 0.0
    return float(number)


def _normalise_feature_dict(data: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    if not data:
        return {}
    return {key: _normalise_feature(key, value) for key, value in data.items()}


def _augment_nonlinear_features(features: Mapping[str, float]) -> Dict[str, float]:
    """Enrich the feature dictionary with non-linear transforms."""

    enriched = dict(features)
    order_imbalance = enriched.get('order_imbalance', 0.0)
    enriched['order_imbalance_sq'] = order_imbalance ** 2
    enriched['extreme_imbalance_flag'] = 1.0 if order_imbalance >= 0.9 else 0.0
    return enriched


def _compute_sample_weights(y: np.ndarray) -> Optional[np.ndarray]:
    """Return class-balanced sample weights for the provided labels."""

    if compute_class_weight is None:
        return None
    try:
        classes = np.unique(y)
        if classes.size < 2:
            return None
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        weight_map = {cls: weight for cls, weight in zip(classes, weights)}
        return np.asarray([weight_map[label] for label in y], dtype=float)
    except Exception:
        return None


def _model_supports_sample_weight(model: Any) -> bool:
    """Check whether the estimator's fit method accepts sample weights."""

    try:
        signature = inspect.signature(model.fit)
    except (TypeError, ValueError):
        return False
    return 'sample_weight' in signature.parameters


def _perform_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cv_splits: int,
) -> Dict[str, Any]:
    """Run RFECV to identify a compact feature subset."""

    if RFECV is None or LogisticRegression is None or X.size == 0 or len(feature_names) == 0:
        return {}
    min_features = max(5, min(len(feature_names) - 1, len(feature_names) // 2))
    try:
        estimator = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')
        selector = RFECV(
            estimator=estimator,
            step=1,
            min_features_to_select=min_features,
            scoring='roc_auc',
            cv=max(2, min(cv_splits, len(y) - 1)),
            n_jobs=-1,
        )
        selector.fit(X, y)
        support = selector.support_
        if support.sum() == 0:
            return {}
        indices = np.where(support)[0]
        selected_features = [feature_names[idx] for idx in indices]
        ranking = selector.ranking_.tolist()
        grid_scores = getattr(selector, 'grid_scores_', None)
        if grid_scores is not None:
            scores = grid_scores.tolist()
        else:
            scores = []
        return {
            'indices': indices.tolist(),
            'support': support.astype(int).tolist(),
            'ranking': ranking,
            'selected_features': selected_features,
            'cv_scores': scores,
        }
    except Exception:
        return {}


def _fit_pca(X: np.ndarray, variance: float = 0.95) -> Optional[Any]:
    """Fit PCA on the training data to capture the desired variance."""

    if PCA is None or X.size == 0:
        return None
    try:
        n_components = min(X.shape[0], X.shape[1])
        if n_components <= 1:
            return None
        if 0 < variance < 1:
            n_comp = variance
        else:
            n_comp = max(1, min(n_components, int(variance)))
        pca = PCA(n_components=n_comp, svd_solver='full')
        pca.fit(X)
        return pca
    except Exception:
        return None


def _transform_feature_matrix(
    X: np.ndarray,
    view: str,
    selection_info: Optional[Dict[str, Any]] = None,
    pca_model: Optional[Any] = None,
) -> np.ndarray:
    """Transform features according to the chosen view."""

    if view == 'rfe' and selection_info:
        indices = selection_info.get('indices')
        if indices:
            return X[:, indices]
    if view == 'pca' and pca_model is not None:
        return pca_model.transform(X)
    return X


def _transform_feature_vector(
    x: np.ndarray,
    view: str,
    selection_info: Optional[Dict[str, Any]] = None,
    pca_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Transform a single feature vector using persisted metadata."""

    if view == 'rfe' and selection_info:
        indices = selection_info.get('indices') or []
        if indices:
            return x[indices]
    if view == 'pca' and pca_params:
        components = np.asarray(pca_params.get('components'))
        pca_mean = np.asarray(pca_params.get('mean'))
        if components.ndim == 2 and pca_mean.size:
            centered = x - pca_mean
            return centered @ components.T
    return x


def _calculate_distribution_summary(X: np.ndarray, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Summarise feature distributions for drift monitoring."""

    summary: Dict[str, Dict[str, float]] = {}
    if X.size == 0:
        return summary
    for idx, name in enumerate(feature_names):
        column = X[:, idx]
        try:
            summary[name] = {
                'mean': float(np.mean(column)),
                'std': float(np.std(column) + 1e-8),
                'p05': float(np.percentile(column, 5)),
                'p50': float(np.percentile(column, 50)),
                'p95': float(np.percentile(column, 95)),
            }
        except Exception:
            summary[name] = {
                'mean': 0.0,
                'std': 1.0,
                'p05': 0.0,
                'p50': 0.0,
                'p95': 0.0,
            }
    return summary


def _build_time_series_cv(n_samples: int, max_splits: int = 5) -> Optional[Any]:
    """Create a TimeSeriesSplit configuration for rolling validation."""

    if TimeSeriesSplit is None or n_samples <= 2:
        return None
    splits = min(max_splits, max(2, n_samples // 50))
    if splits >= n_samples:
        splits = max(2, min(max_splits, n_samples - 1))
    if splits < 2:
        return None
    try:
        return TimeSeriesSplit(n_splits=splits)
    except Exception:
        return None

def load_model_report() -> Dict[str, Any]:
    """Return the persisted model metadata if training artefacts exist."""

    metadata = _load_model_metadata()
    if metadata and os.path.exists(MODEL_PKL):
        metadata['model_path'] = MODEL_PKL
    return metadata


def _extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and label vector y from the learning log."""

    session_map = {"Asia": 0, "Europe": 1, "US": 2, "New York": 2, "unknown": 3}
    success_outcomes = {"tp1", "tp2", "tp3", "tp4", "tp4_sl", "win"}
    feature_order = [
        'score', 'confidence', 'session_id', 'btc_dom', 'fear_greed', 'sent_conf',
        'sent_bias', 'pattern_len', 'volatility', 'htf_trend', 'order_imbalance',
        'order_imbalance_sq', 'extreme_imbalance_flag',
        'macro_indicator', 'macd', 'rsi', 'sma', 'atr', 'volume', 'macd_rsi',
        'order_flow_score', 'order_flow_flag', 'cvd', 'cvd_change', 'taker_buy_ratio',
        'trade_imbalance', 'aggressive_trade_rate', 'spoofing_intensity',
        'spoofing_alert', 'volume_ratio', 'price_change_pct', 'spread_bps',
        'llm_decision', 'llm_confidence', 'time_since_last', 'recent_win_rate',
    ]
    feature_list: List[List[float]] = []
    labels: List[int] = []
    if "entry_time" in df.columns:
        df = df.sort_values(by="entry_time")
    elif "timestamp" in df.columns:
        df = df.sort_values(by="timestamp")
    last_exit: Optional[pd.Timestamp] = None
    recent_outcomes: List[str] = []
    window = 5
    for _, row in df.iterrows():
        try:
            session = row.get("session", "unknown")
            entry_dt = pd.to_datetime(row.get("entry_time", row.get("timestamp")), errors="coerce")
            exit_dt = pd.to_datetime(row.get("exit_time", row.get("timestamp")), errors="coerce")
            if last_exit is not None and entry_dt is not None:
                time_since_last = (entry_dt - last_exit).total_seconds() / 3600.0
            else:
                time_since_last = 0.0
            if recent_outcomes:
                win_count = sum(1 for o in recent_outcomes if o in success_outcomes)
                recent_win_rate = win_count / len(recent_outcomes)
            else:
                recent_win_rate = 0.5
            base_features = {
                'score': row.get('score', 0.0),
                'confidence': row.get('confidence', 0.0),
                'session_id': session_map.get(str(session), 3),
                'btc_dom': row.get('btc_dominance', 0.0),
                'fear_greed': row.get('fear_greed', 0.0),
                'sent_conf': row.get('sentiment_confidence', row.get('confidence', 5.0)),
                'sent_bias': row.get('sentiment_bias', 'neutral'),
                'pattern_len': len(str(row.get('pattern', 'none'))),
                'volatility': row.get('volatility', 0.0),
                'htf_trend': row.get('htf_trend', 0.0),
                'order_imbalance': row.get('order_imbalance', 0.0),
                'macro_indicator': row.get('macro_indicator', 0.0),
                'macd': row.get('macd', 0.0),
                'rsi': row.get('rsi', 50.0),
                'sma': row.get('sma', 0.0),
                'atr': row.get('atr', 0.0),
                'volume': row.get('volume', 0.0),
                'order_flow_score': row.get('order_flow_score', 0.0),
                'order_flow_flag': row.get('order_flow_flag', 0.0),
                'cvd': row.get('cvd', 0.0),
                'cvd_change': row.get('cvd_change', 0.0),
                'taker_buy_ratio': row.get('taker_buy_ratio', 0.0),
                'trade_imbalance': row.get('trade_imbalance', 0.0),
                'aggressive_trade_rate': row.get('aggressive_trade_rate', 0.0),
                'spoofing_intensity': row.get('spoofing_intensity', 0.0),
                'spoofing_alert': row.get('spoofing_alert', 0.0),
                'volume_ratio': row.get('volume_ratio', 0.0),
                'price_change_pct': row.get('price_change_pct', 0.0),
                'spread_bps': row.get('spread_bps', 0.0),
                'llm_confidence': row.get('llm_confidence', 5.0),
                'time_since_last': time_since_last,
                'recent_win_rate': recent_win_rate,
            }
            macd_norm = _normalise_feature('macd', base_features['macd'])
            rsi_norm = _normalise_feature('rsi', base_features['rsi'])
            base_features['macd_rsi'] = macd_norm * rsi_norm
            approval_raw = row.get("llm_approval", row.get("llm_decision"))
            llm_decision_bool = _coerce_bool(approval_raw)
            base_features['llm_decision'] = 1.0 if llm_decision_bool is not False else 0.0
            normalised = _normalise_feature_dict(base_features)
            normalised = _augment_nonlinear_features(normalised)
            features = [normalised.get(name, 0.0) for name in feature_order]
            feature_list.append(features)
            outcome = str(row.get("outcome", "loss")).lower()
            labels.append(1 if outcome in success_outcomes else 0)
            recent_outcomes.append(outcome)
            if len(recent_outcomes) > window:
                recent_outcomes.pop(0)
            if exit_dt is not None:
                last_exit = exit_dt
        except Exception:
            continue
    X = np.array(feature_list, dtype=float)
    y = np.array(labels, dtype=float)
    return X, y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _safe_predict_proba(model: Any, X: np.ndarray) -> Optional[np.ndarray]:
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            proba_arr = np.asarray(proba, dtype=float)
            if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                return proba_arr[:, 1]
        if hasattr(model, 'decision_function'):
            decision = np.asarray(model.decision_function(X), dtype=float)
            if decision.ndim == 1:
                return _sigmoid(decision)
    except Exception:
        return None
    return None


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if accuracy_score is not None:
        try:
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        except Exception:
            metrics['accuracy'] = float('nan')
    if balanced_accuracy_score is not None:
        try:
            metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
        except Exception:
            metrics['balanced_accuracy'] = float('nan')
    if precision_recall_fscore_support is not None:
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average='binary',
                zero_division=0,
            )
            metrics['precision'] = float(precision)
            metrics['recall'] = float(recall)
            metrics['f1'] = float(f1)
        except Exception:
            metrics.setdefault('precision', float('nan'))
            metrics.setdefault('recall', float('nan'))
            metrics.setdefault('f1', float('nan'))
    if y_prob is not None and roc_auc_score is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics['roc_auc'] = float('nan')
    if y_prob is not None and brier_score_loss is not None:
        try:
            metrics['brier'] = float(brier_score_loss(y_true, y_prob))
        except Exception:
            metrics['brier'] = float('nan')
    if y_prob is not None and log_loss is not None:
        try:
            metrics['log_loss'] = float(log_loss(y_true, np.vstack([1 - y_prob, y_prob]).T))
        except Exception:
            metrics['log_loss'] = float('nan')
    return metrics


def _extract_base_estimator(model: Any) -> Any:
    if hasattr(model, 'base_estimator_'):
        return getattr(model, 'base_estimator_')
    return model


def _compute_feature_attribution(
    model: Any,
    feature_names: List[str],
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    feature_scores: Dict[str, float] = {}
    estimator = _extract_base_estimator(model)
    try:
        if hasattr(estimator, 'feature_importances_'):
            importances = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, 'coef_'):
            coef = np.asarray(estimator.coef_, dtype=float)
            importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
        elif permutation_importance is not None and y is not None:
            perm = permutation_importance(estimator, X, y, n_repeats=10, random_state=42)
            importances = perm.importances_mean
        else:
            return feature_scores
        if importances.shape[0] != len(feature_names):
            return feature_scores
        total = float(np.sum(np.abs(importances)))
        if total <= 0:
            return feature_scores
        for name, value in zip(feature_names, importances):
            feature_scores[name] = float(value) / total
    except Exception:
        feature_scores = {}
    return dict(sorted(feature_scores.items(), key=lambda kv: kv[1], reverse=True))


def train_model(iterations: int = 200, learning_rate: float = 0.1) -> Optional[TrainingDiagnostics]:
    """
    Train and select the best classification model on the trade learning log.

    If scikit‑learn is available, this function performs the following:
    1. Extracts features and labels from the completed trades log
       (``TRADE_HISTORY_FILE``).
    2. Scales the features using ``StandardScaler``.
    3. Defines a grid of candidate models (logistic regression,
       random forest and gradient boosting) along with hyper‑parameter
       options.
    4. Runs ``GridSearchCV`` with stratified cross‑validation to find
       the best performing model.
    5. Saves the fitted model to ``ml_model.pkl`` via joblib and
       writes metadata (model type, scaler statistics) to
       ``ml_model.json``.

    When scikit‑learn is not available, the function falls back to a
    manual logistic regression implemented with gradient descent.
    The coefficients and scaling statistics are written to
    ``ml_model.json``.
    """
    # Ensure there is sufficient data
    if not os.path.exists(LOG_FILE):
        logger.warning("No trade learning log found. Cannot train ML model.")
        return None
    try:
        df = pd.read_csv(LOG_FILE, engine="python", on_bad_lines="skip", encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to read learning log: %s", e, exc_info=True)
        return None
    if len(df) < 20:
        logger.warning("Not enough data to train ML model. Need at least 20 trades.")
        return None
    X, y = _extract_features(df)
    if X.size == 0:
        logger.warning("No valid training samples extracted.")
        return None
    unique_classes = np.unique(y)
    if unique_classes.size < 2:
        logger.warning(
            "Training data must contain at least two outcome classes. Found classes: %s",
            unique_classes,
        )
        return None
    # Remove existing model artefacts before training a new one
    for artefact in (MODEL_PKL, MODEL_JSON):
        if os.path.exists(artefact):
            try:
                os.remove(artefact)
            except Exception:
                pass
    if SKLEARN_AVAILABLE:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        try:
            train_idx, val_idx = next(splitter.split(X, y))
        except Exception:
            split_point = max(1, int(len(X) * 0.8))
            train_idx = np.arange(0, split_point)
            val_idx = np.arange(split_point, len(X))
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_val_norm = scaler.transform(X_val)
        scaler_full = StandardScaler()
        X_full_norm = scaler_full.fit_transform(X)
        class_weight_metadata: Dict[str, float] = {}
        if compute_class_weight is not None:
            try:
                all_classes = np.unique(y)
                class_weights = compute_class_weight('balanced', classes=all_classes, y=y)
                class_weight_metadata = {str(int(cls)): float(weight) for cls, weight in zip(all_classes, class_weights)}
            except Exception:
                class_weight_metadata = {}
        sample_weights_train = _compute_sample_weights(y_train)
        sample_weights_full = _compute_sample_weights(y)
        feature_names = [
            'score', 'confidence', 'session_id', 'btc_dom',
            'fear_greed', 'sent_conf', 'sent_bias', 'pattern_len',
            'volatility', 'htf_trend', 'order_imbalance', 'order_imbalance_sq',
            'extreme_imbalance_flag', 'macro_indicator',
            'macd', 'rsi', 'sma', 'atr', 'volume', 'macd_rsi',
            'order_flow_score', 'order_flow_flag', 'cvd', 'cvd_change',
            'taker_buy_ratio', 'trade_imbalance', 'aggressive_trade_rate',
            'spoofing_intensity', 'spoofing_alert', 'volume_ratio',
            'price_change_pct', 'spread_bps',
            'llm_decision', 'llm_confidence', 'time_since_last', 'recent_win_rate'
        ]
        time_cv = _build_time_series_cv(len(X_train_norm))
        cv_for_selection = time_cv.get_n_splits() if time_cv is not None else 5
        selection_info = _perform_feature_selection(X_train_norm, y_train, feature_names, cv_for_selection)
        pca_model = _fit_pca(X_train_norm)
        feature_views: Dict[str, Dict[str, np.ndarray]] = {
            'standard': {'train': X_train_norm, 'val': X_val_norm},
        }
        if selection_info:
            feature_views['rfe'] = {
                'train': _transform_feature_matrix(X_train_norm, 'rfe', selection_info, None),
                'val': _transform_feature_matrix(X_val_norm, 'rfe', selection_info, None),
            }
        if pca_model is not None:
            feature_views['pca'] = {
                'train': _transform_feature_matrix(X_train_norm, 'pca', None, pca_model),
                'val': _transform_feature_matrix(X_val_norm, 'pca', None, pca_model),
            }

        def _logistic_factory() -> Any:
            return LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs')

        def _rf_factory() -> Any:
            return RandomForestClassifier(class_weight='balanced')

        def _gb_factory() -> Any:
            return GradientBoostingClassifier()

        def _mlp_factory() -> Any:
            return MLPClassifier(max_iter=500)

        model_candidates: List[Dict[str, Any]] = [
            {
                'name': 'logistic',
                'base_type': 'logistic',
                'factory': _logistic_factory,
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                },
                'view': 'standard',
            },
            {
                'name': 'logistic_rfe',
                'base_type': 'logistic',
                'factory': _logistic_factory,
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                },
                'view': 'rfe',
            },
            {
                'name': 'random_forest',
                'base_type': 'random_forest',
                'factory': _rf_factory,
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, None],
                    'min_samples_leaf': [1, 2, 4],
                },
                'view': 'standard',
            },
            {
                'name': 'gradient_boosting',
                'base_type': 'gradient_boosting',
                'factory': _gb_factory,
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5],
                },
                'view': 'standard',
            },
            {
                'name': 'mlp_pca',
                'base_type': 'mlp',
                'factory': _mlp_factory,
                'param_grid': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'alpha': [0.0001, 0.001],
                },
                'view': 'pca',
            },
        ]
        if XGBClassifier is not None:
            model_candidates.append({
                'name': 'xgboost',
                'base_type': 'xgboost',
                'factory': lambda: XGBClassifier(eval_metric='logloss', use_label_encoder=False),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                },
                'view': 'standard',
            })
        if LGBMClassifier is not None:
            model_candidates.append({
                'name': 'lightgbm',
                'base_type': 'lightgbm',
                'factory': lambda: LGBMClassifier(),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'num_leaves': [31, 63, 127],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, -1],
                },
                'view': 'standard',
            })
        if CatBoostClassifier is not None:
            model_candidates.append({
                'name': 'catboost',
                'base_type': 'catboost',
                'factory': lambda: CatBoostClassifier(verbose=False, loss_function='Logloss'),
                'param_grid': {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1],
                    'iterations': [100, 200],
                },
                'view': 'standard',
            })

        best_model: Optional[Any] = None
        best_type = 'logistic'
        best_params: Dict[str, Any] = {}
        best_score = -np.inf
        best_view = 'standard'
        metrics: Dict[str, float] = {}
        calibration_method: Optional[str] = None
        calibrated = False
        best_cv_score: Optional[float] = None
        best_candidate_name: Optional[str] = None
        scoring_metric = 'roc_auc' if roc_auc_score is not None else 'balanced_accuracy'
        for candidate in model_candidates:
            view = candidate['view']
            if view not in feature_views:
                continue
            try:
                estimator = candidate['factory']()
            except Exception:
                logger.warning("Failed to instantiate model %s", candidate['name'], exc_info=True)
                continue
            X_train_view = feature_views[view]['train']
            X_val_view = feature_views[view]['val']
            fit_kwargs: Dict[str, Any] = {}
            candidate_sample_weights = sample_weights_train
            if candidate_sample_weights is not None and not _model_supports_sample_weight(estimator):
                candidate_sample_weights = None
            if candidate_sample_weights is not None:
                fit_kwargs['sample_weight'] = candidate_sample_weights
            param_grid = candidate.get('param_grid') or {}
            cv = time_cv
            if cv is None:
                if StratifiedKFold is not None and len(X_train_view) >= 5:
                    try:
                        cv = StratifiedKFold(n_splits=min(5, max(2, len(X_train_view) // 4)), shuffle=True, random_state=42)
                    except Exception:
                        cv = 3
                else:
                    cv = 3
            try:
                if param_grid:
                    grid = GridSearchCV(
                        estimator,
                        param_grid,
                        scoring=scoring_metric,
                        cv=cv,
                        n_jobs=-1,
                    )
                    if fit_kwargs:
                        grid.fit(X_train_view, y_train, **fit_kwargs)
                    else:
                        grid.fit(X_train_view, y_train)
                    estimator = grid.best_estimator_
                    current_params = grid.best_params_
                    cv_score = float(grid.best_score_)
                else:
                    if fit_kwargs:
                        estimator.fit(X_train_view, y_train, **fit_kwargs)
                    else:
                        estimator.fit(X_train_view, y_train)
                    current_params = {}
                    cv_score = float('nan')
            except Exception:
                logger.warning("Training failed for model %s", candidate['name'], exc_info=True)
                continue
            try:
                y_val_pred = estimator.predict(X_val_view)
                y_val_prob = _safe_predict_proba(estimator, X_val_view)
                val_metrics = _compute_classification_metrics(y_val, y_val_pred, y_val_prob)
            except Exception:
                val_metrics = {}
            candidate_score = val_metrics.get('roc_auc')
            if candidate_score is None or np.isnan(candidate_score):
                candidate_score = val_metrics.get('balanced_accuracy')
            if candidate_score is None or np.isnan(candidate_score):
                candidate_score = val_metrics.get('accuracy')
            if candidate_score is None or np.isnan(candidate_score):
                candidate_score = -np.inf
            if candidate_score > best_score:
                best_score = float(candidate_score)
                best_model = estimator
                best_type = candidate['base_type']
                best_params = current_params
                metrics = val_metrics
                best_view = view
                best_cv_score = cv_score
                best_candidate_name = candidate['name']
        if best_model is None:
            logger.warning("No suitable model found during training.")
            return None

        if best_view == 'pca' and pca_model is not None:
            try:
                n_components = getattr(pca_model, 'n_components_', None) or getattr(pca_model, 'n_components', None)
                pca_full = PCA(n_components=n_components, svd_solver='full')
                pca_full.fit(X_full_norm)
            except Exception:
                pca_full = pca_model
        else:
            pca_full = pca_model if best_view == 'pca' else None

        X_val_best = _transform_feature_matrix(X_val_norm, best_view, selection_info, pca_model)
        X_full_transformed = _transform_feature_matrix(X_full_norm, best_view, selection_info, pca_full)
        try:
            y_val_pred = best_model.predict(X_val_best)
            y_val_prob = _safe_predict_proba(best_model, X_val_best)
            metrics = _compute_classification_metrics(y_val, y_val_pred, y_val_prob)
        except Exception:
            metrics = {}

        persisted_model: Any
        if hasattr(best_model, 'predict_proba') and CalibratedClassifierCV is not None:
            try:
                calibration_method = 'isotonic' if len(y) >= 200 else 'sigmoid'
                base_estimator = clone(best_model) if clone is not None else best_model
                persisted_model = CalibratedClassifierCV(
                    base_estimator=base_estimator,
                    method=calibration_method,
                    cv=3,
                )
                fit_kwargs_full: Dict[str, Any] = {}
                if sample_weights_full is not None and _model_supports_sample_weight(persisted_model):
                    fit_kwargs_full['sample_weight'] = sample_weights_full
                persisted_model.fit(X_full_transformed, y, **fit_kwargs_full)
                calibrated = True
            except Exception:
                persisted_model = clone(best_model) if clone is not None else best_model
                fit_kwargs_full = {}
                if sample_weights_full is not None and _model_supports_sample_weight(persisted_model):
                    fit_kwargs_full['sample_weight'] = sample_weights_full
                persisted_model.fit(X_full_transformed, y, **fit_kwargs_full)
                calibration_method = None
        else:
            persisted_model = clone(best_model) if clone is not None else best_model
            fit_kwargs_full = {}
            if sample_weights_full is not None and _model_supports_sample_weight(persisted_model):
                fit_kwargs_full['sample_weight'] = sample_weights_full
            persisted_model.fit(X_full_transformed, y, **fit_kwargs_full)

        if best_view == 'rfe' and selection_info:
            model_feature_names = [feature_names[idx] for idx in selection_info.get('indices', [])]
        elif best_view == 'pca' and pca_full is not None:
            n_components = getattr(pca_full, 'n_components_', None)
            if n_components is None:
                n_components = X_full_transformed.shape[1]
            model_feature_names = [f'pca_component_{i}' for i in range(int(n_components))]
        else:
            model_feature_names = feature_names
        feature_info = _compute_feature_attribution(persisted_model, model_feature_names, X_full_transformed, y)
        class_counts = {
            str(int(cls)): int(np.sum(y == cls)) for cls in np.unique(y)
        }
        feature_distribution = _calculate_distribution_summary(X_full_norm, feature_names)
        cv_score_serialisable: Optional[float] = (
            float(best_cv_score) if best_cv_score is not None and np.isfinite(best_cv_score) else None
        )
        diagnostics = TrainingDiagnostics(
            model_type=best_type,
            best_params=best_params,
            validation_metrics=metrics,
            calibrated=calibrated,
            calibration_method=calibration_method,
            feature_importance=feature_info,
            samples=len(y),
        )
        try:
            joblib.dump(persisted_model, MODEL_PKL)
            model_metadata = {
                'model_type': best_type,
                'model_name': best_candidate_name,
                'feature_view': best_view,
                'scaler_mean': scaler_full.mean_.tolist(),
                'scaler_scale': scaler_full.scale_.tolist(),
                'feature_names': feature_names,
                'model_feature_names': model_feature_names,
                'validation_metrics': metrics,
                'calibration': {
                    'applied': calibrated,
                    'method': calibration_method,
                },
                'best_params': best_params,
                'class_distribution': class_counts,
                'class_weights': class_weight_metadata,
                'validation_samples': int(len(y_val)),
                'training_timestamp': datetime.utcnow().isoformat(),
                'feature_importance': feature_info,
                'feature_distribution': feature_distribution,
                'cv_best_score': cv_score_serialisable,
                'selection_info': selection_info,
                'pca': {
                    'components': getattr(pca_full, 'components_', None).tolist() if best_view == 'pca' and hasattr(pca_full, 'components_') else None,
                    'mean': getattr(pca_full, 'mean_', None).tolist() if best_view == 'pca' and hasattr(pca_full, 'mean_') else None,
                    'explained_variance_ratio': getattr(pca_full, 'explained_variance_ratio_', None).tolist() if best_view == 'pca' and hasattr(pca_full, 'explained_variance_ratio_') else None,
                },
            }
            with open(MODEL_JSON, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            logger.info(
                "Best model (%s/%s) saved to %s with metadata %s",
                best_type,
                best_view,
                MODEL_PKL,
                MODEL_JSON,
            )
        except Exception as e:
            logger.warning("Failed to save model: %s", e, exc_info=True)
        return diagnostics

    else:
        # ------------------------------------------------------------------
        # Fallback: manual logistic regression without sklearn
        # ------------------------------------------------------------------
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
        logger.info("Manual logistic regression trained.")
        model_data = {
            'model_type': 'manual',
            'weights': weights.tolist(),
            'mu': mu.tolist(),
            'sigma': sigma.tolist(),
            'feature_names': [
                'score', 'confidence', 'session_id', 'btc_dom',
                'fear_greed', 'sent_conf', 'sent_bias', 'pattern_len',
                'volatility', 'htf_trend', 'order_imbalance', 'order_imbalance_sq',
                'extreme_imbalance_flag', 'macro_indicator',
                'macd', 'rsi', 'sma', 'atr', 'volume', 'macd_rsi',
                'order_flow_score', 'order_flow_flag', 'cvd', 'cvd_change',
                'taker_buy_ratio', 'trade_imbalance', 'aggressive_trade_rate',
                'spoofing_intensity', 'spoofing_alert', 'volume_ratio',
                'price_change_pct', 'spread_bps',
                'llm_decision', 'llm_confidence', 'time_since_last', 'recent_win_rate'
            ],
        }
        try:
            with open(MODEL_JSON, 'w') as f:
                json.dump(model_data, f, indent=2)
            logger.info("Manual model saved to %s", MODEL_JSON)
        except Exception as e:
            logger.warning("Failed to save manual model: %s", e, exc_info=True)
        metrics = {}
        try:
            logits = X_aug @ weights
            preds = (logits > 0).astype(float)
            probs = _sigmoid(logits)
            metrics = _compute_classification_metrics(y, preds, probs)
        except Exception:
            metrics = {}
        return TrainingDiagnostics(
            model_type='manual',
            best_params={},
            validation_metrics=metrics,
            calibrated=False,
            calibration_method=None,
            feature_importance={},
            samples=len(y),
        )


def _load_model_metadata() -> Dict[str, Any]:
    """Load model metadata from the JSON file, if present."""
    if not os.path.exists(MODEL_JSON):
        return {}
    try:
        with open(MODEL_JSON, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def detect_feature_drift(
    recent_data: pd.DataFrame,
    feature_shift_threshold: float = 0.3,
    label_shift_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Detect distribution drift between recent samples and the training data."""

    metadata = _load_model_metadata()
    if not metadata:
        return {'status': 'unavailable', 'reason': 'no_metadata'}
    feature_names: List[str] = metadata.get('feature_names') or []
    if not feature_names:
        return {'status': 'unavailable', 'reason': 'missing_feature_names'}
    reference_summary: Dict[str, Dict[str, float]] = metadata.get('feature_distribution') or {}
    if not reference_summary:
        return {'status': 'unavailable', 'reason': 'missing_reference_distribution'}
    scaler_mean = np.asarray(metadata.get('scaler_mean', []), dtype=float)
    scaler_scale = np.asarray(metadata.get('scaler_scale', []), dtype=float)
    if scaler_mean.size == 0 or scaler_scale.size == 0:
        return {'status': 'unavailable', 'reason': 'missing_scaler'}
    X_recent, y_recent = _extract_features(recent_data)
    if X_recent.size == 0:
        return {'status': 'unavailable', 'reason': 'no_recent_samples'}
    try:
        X_recent_norm = (X_recent - scaler_mean) / scaler_scale
    except Exception:
        return {'status': 'error', 'reason': 'normalisation_failed'}
    recent_summary = _calculate_distribution_summary(X_recent_norm, feature_names)
    drift_features: Dict[str, float] = {}
    for name in feature_names:
        ref_stats = reference_summary.get(name)
        recent_stats = recent_summary.get(name)
        if not ref_stats or not recent_stats:
            continue
        ref_std = abs(ref_stats.get('std', 0.0)) + 1e-6
        mean_shift = abs(recent_stats.get('mean', 0.0) - ref_stats.get('mean', 0.0)) / ref_std
        std_shift = abs(recent_stats.get('std', 0.0) - ref_stats.get('std', 0.0)) / ref_std
        drift_score = float(max(mean_shift, std_shift))
        if drift_score > feature_shift_threshold:
            drift_features[name] = drift_score
    label_drift: Dict[str, float] = {}
    reference_classes = metadata.get('class_distribution', {})
    if reference_classes and y_recent.size > 0:
        recent_counts = {str(int(cls)): int(np.sum(y_recent == cls)) for cls in np.unique(y_recent)}
        total_ref = float(sum(reference_classes.values()) or 1.0)
        total_recent = float(sum(recent_counts.values()) or 1.0)
        for cls in set(reference_classes.keys()) | set(recent_counts.keys()):
            ref_ratio = reference_classes.get(cls, 0) / total_ref
            recent_ratio = recent_counts.get(cls, 0) / total_recent
            diff = abs(recent_ratio - ref_ratio)
            if diff > label_shift_threshold:
                label_drift[cls] = diff
    should_retrain = bool(drift_features or label_drift)
    return {
        'status': 'ok',
        'should_retrain': should_retrain,
        'feature_drift': drift_features,
        'label_drift': label_drift,
        'recent_samples': int(len(X_recent)),
    }


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        if isinstance(value, (int, float)) and value == value:
            if value in (0, 1):
                return bool(value)
    except Exception:
        return None
    token = str(value).strip().lower()
    if not token:
        return None
    if token in {"true", "1", "yes", "approved", "y"}:
        return True
    if token in {"false", "0", "no", "vetoed", "n"}:
        return False
    return None


def _prepare_feature_vector(
    score: float,
    confidence: float,
    session: str,
    btc_d: float,
    fg: float,
    sentiment_conf: float,
    pattern: str,
    llm_approval: bool,
    llm_confidence: float,
    time_since_last: float,
    recent_win_rate: float,
) -> Dict[str, float]:
    session_map = {"Asia": 0, "Europe": 1, "US": 2, "New York": 2, "unknown": 3}
    session_id = session_map.get(session, 3)
    base = {
        'score': score,
        'confidence': confidence,
        'session_id': session_id,
        'btc_dom': btc_d,
        'fear_greed': fg,
        'sent_conf': sentiment_conf,
        'pattern_len': len(str(pattern)),
        'llm_decision': 1.0 if llm_approval else 0.0,
        'llm_confidence': llm_confidence,
        'time_since_last': time_since_last,
        'recent_win_rate': recent_win_rate,
    }
    normalised = _normalise_feature_dict(base)
    return _augment_nonlinear_features(normalised)


def _assemble_feature_vector(
    feature_names: List[str],
    provided: Mapping[str, float],
    overrides: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    values: Dict[str, float] = dict(_FEATURE_FALLBACKS)
    for key, value in provided.items():
        values[key] = float(value)
    if overrides:
        for key, value in overrides.items():
            values[key] = float(value)
    return np.array([values.get(name, 0.0) for name in feature_names], dtype=float)


def predict_success_probability(
    score: float,
    confidence: float,
    session: str,
    btc_d: float,
    fg: float,
    sentiment_conf: float,
    pattern: str,
    llm_approval: bool = True,
    llm_confidence: float = 5.0,
    feature_overrides: Optional[Mapping[str, float]] = None,
    micro_features: Optional[Mapping[str, Any]] = None,
) -> float:
    """
    Predict the probability of a trade succeeding based on current features.

    Parameters
    ----------
    llm_approval : bool, optional
        Whether the LLM advisor approved the setup. Defaults to ``True`` when
        the decision is unavailable so historical models remain compatible.

    The function loads the previously trained model.  If a pickled
    sklearn model exists (``ml_model.pkl``) it is used; otherwise it
    falls back to the manual logistic regression defined in
    ``ml_model.json``.  In case of any loading errors, a neutral
    probability of 0.5 is returned.
    """
    metadata = _load_model_metadata()
    if not metadata:
        return 0.5
    history = load_trade_history_df()
    time_since_last = 0.0
    recent_win_rate = 0.5
    success_outcomes = {"tp1", "tp2", "tp3", "tp4", "tp4_sl", "win"}
    if not history.empty:
        try:
            if "exit_time" in history.columns:
                history = history.sort_values(by="exit_time")
                last_time = pd.to_datetime(history["exit_time"].iloc[-1], errors="coerce")
            else:
                history = history.sort_values(by="timestamp")
                last_time = pd.to_datetime(history["timestamp"].iloc[-1], errors="coerce")
            if last_time is not None and not pd.isna(last_time):
                time_since_last = (datetime.utcnow() - last_time).total_seconds() / 3600.0
        except Exception:
            pass
        try:
            recent = history.tail(5)
            outcomes = [str(o).lower() for o in recent.get("outcome", [])]
            wins = [o for o in outcomes if o in success_outcomes]
            if outcomes:
                recent_win_rate = len(wins) / len(outcomes)
        except Exception:
            pass
    base_features = _prepare_feature_vector(
        score,
        confidence,
        session,
        btc_d,
        fg,
        sentiment_conf,
        pattern,
        llm_approval,
        llm_confidence,
        time_since_last,
        recent_win_rate,
    )
    overrides: Dict[str, float] = {}
    if feature_overrides:
        overrides.update({k: float(v) for k, v in feature_overrides.items()})
    micro_norm = _normalise_feature_dict(micro_features)
    overrides.update(_augment_nonlinear_features(micro_norm))
    feature_names = metadata.get('feature_names') or list(_FEATURE_FALLBACKS.keys())
    missing_features = [name for name in _FEATURE_FALLBACKS.keys() if name not in feature_names]
    if missing_features:
        feature_names = feature_names + missing_features
    x = _assemble_feature_vector(feature_names, base_features, overrides)
    model_type = metadata.get('model_type', 'manual')
    try:
        # Use sklearn model if available
        if model_type in {'logistic', 'random_forest', 'gradient_boosting', 'mlp', 'xgboost', 'lightgbm', 'catboost'} and SKLEARN_AVAILABLE and os.path.exists(MODEL_PKL):
            mean = np.array(metadata.get('scaler_mean'), dtype=float)
            scale = np.array(metadata.get('scaler_scale'), dtype=float)
            scale = np.where(scale == 0, 1.0, scale)
            x_norm = (x - mean) / scale
            view = metadata.get('feature_view', 'standard')
            selection_info = metadata.get('selection_info') if view == 'rfe' else None
            pca_params = metadata.get('pca') if view == 'pca' else None
            x_view = _transform_feature_vector(x_norm, view, selection_info, pca_params)
            clf = joblib.load(MODEL_PKL)
            x_input = np.asarray(x_view, dtype=float).reshape(1, -1)
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(x_input)
                prob = float(proba[0][1])
                return prob
            if hasattr(clf, 'predict'):
                pred = clf.predict(x_input)
                return float(pred[0])
        elif model_type == 'manual':
            # Manual logistic regression
            mu = np.array(metadata.get('mu'))
            sigma = np.array(metadata.get('sigma'))
            weights = np.array(metadata.get('weights'))
            x_norm = (x - mu) / (sigma + 1e-8)
            x_aug = np.hstack([1.0, x_norm])
            z = float(x_aug @ weights)
            prob = float(_sigmoid(np.array([z]))[0])
            return prob
        else:
            return 0.5
    except Exception:
        return 0.5
