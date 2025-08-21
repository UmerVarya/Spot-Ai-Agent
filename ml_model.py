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

These changes allow the agent to capture non‑linear interactions
between features and to adapt more flexibly to the historical trade
performance while remaining backward compatible with older models.
"""

from __future__ import annotations

import os
import json
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from log_utils import setup_logger

try:
    # Core sklearn components used for modelling and preprocessing
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import make_scorer, accuracy_score
    import joblib  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:
    LogisticRegression = None  # type: ignore
    RandomForestClassifier = None  # type: ignore
    GradientBoostingClassifier = None  # type: ignore
    StandardScaler = None  # type: ignore
    GridSearchCV = None  # type: ignore
    StratifiedKFold = None  # type: ignore
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
LOG_FILE = os.path.join(ROOT_DIR, "trade_learning_log.csv")
MODEL_JSON = os.path.join(ROOT_DIR, "ml_model.json")
MODEL_PKL = os.path.join(ROOT_DIR, "ml_model.pkl")


def _extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and label vector y from the learning log."""
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
            sent_conf = row.get("sentiment_confidence", row.get("sentiment", 5))
            try:
                sent_conf_val = float(sent_conf)
            except Exception:
                sent_conf_val = 5.0
            pattern = row.get("pattern", "none")
            pattern_len = len(str(pattern))
            session_id = session_map.get(str(session), 3)
            volatility = float(row.get("volatility", 0.0)) / 100.0
            htf_trend = float(row.get("htf_trend", 0.0)) / 100.0
            order_imbalance = float(row.get("order_imbalance", 0.0)) / 100.0
            macro_indicator = float(row.get("macro_indicator", 0.0)) / 100.0
            # Additional technical indicators from the learning log.  These
            # columns may not always be present so we fall back to sensible
            # defaults when missing.
            macd_val = float(row.get("macd", 0.0)) / 100.0
            rsi_val = float(row.get("rsi", 50.0)) / 100.0
            sma_val = float(row.get("sma", 0.0)) / 100.0
            atr_val = float(row.get("atr", 0.0)) / 100.0
            vol_val = float(row.get("volume", 0.0)) / 1_000_000.0
            macd_rsi = macd_val * rsi_val  # interaction feature to filter noise
            features = [
                score,
                conf,
                session_id,
                btc_dom / 100.0,
                fg / 100.0,
                sent_conf_val / 10.0,
                pattern_len / 10.0,
                volatility,
                htf_trend,
                order_imbalance,
                macro_indicator,
                macd_val,
                rsi_val,
                sma_val,
                atr_val,
                vol_val,
                macd_rsi,
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
    """
    Train and select the best classification model on the trade learning log.

    If scikit‑learn is available, this function performs the following:
    1. Extracts features and labels from ``trade_learning_log.csv``.
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
        return
    try:
        df = pd.read_csv(LOG_FILE, engine="python", on_bad_lines="skip", encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to read learning log: %s", e, exc_info=True)
        return
    if len(df) < 20:
        logger.warning("Not enough data to train ML model. Need at least 20 trades.")
        return
    X, y = _extract_features(df)
    if X.size == 0:
        logger.warning("No valid training samples extracted.")
        return
    # Remove existing model artefacts before training a new one
    for artefact in (MODEL_PKL, MODEL_JSON):
        if os.path.exists(artefact):
            try:
                os.remove(artefact)
            except Exception:
                pass
    if SKLEARN_AVAILABLE:
        # ------------------------------------------------------------------
        # Scikit‑learn pipeline: scale features and evaluate multiple models
        # ------------------------------------------------------------------
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        # Define model grid with reasonable hyper‑parameters
        models: Dict[str, Any] = {
            'logistic': LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'),
            'random_forest': RandomForestClassifier(class_weight='balanced'),
            'gradient_boosting': GradientBoostingClassifier(),
            'mlp': MLPClassifier(max_iter=500),
        }
        if XGBClassifier is not None:
            models['xgboost'] = XGBClassifier(
                eval_metric='logloss',
                use_label_encoder=False,
            )
        if LGBMClassifier is not None:
            models['lightgbm'] = LGBMClassifier()
        param_grid: Dict[str, Dict[str, List[Any]]] = {
            'logistic': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_leaf': [1, 2, 4],
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5],
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'alpha': [0.0001, 0.001],
            },
        }
        if XGBClassifier is not None:
            param_grid['xgboost'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
            }
        if LGBMClassifier is not None:
            param_grid['lightgbm'] = {
                'n_estimators': [50, 100, 200],
                'num_leaves': [31, 63, 127],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, -1],
            }
        # Use stratified K‑fold to preserve class distribution
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        best_model: Optional[Any] = None
        best_score: float = -np.inf
        best_type: str = 'logistic'
        # Train each model type separately
        for model_name, model in models.items():
            grid_params = param_grid.get(model_name, {})
            if not grid_params:
                # If no grid specified (should not happen), skip grid search
                estimator = model.fit(X_norm, y)
                score = float(estimator.score(X_norm, y))
                candidate_model = estimator
            else:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=grid_params,
                    cv=cv,
                    scoring=make_scorer(accuracy_score),
                    n_jobs=-1,
                )
                gs.fit(X_norm, y)
                candidate_model = gs.best_estimator_
                score = float(gs.best_score_)
            logger.info("%s trained. CV accuracy: %.2f%%", model_name, score * 100)
            if score > best_score:
                best_score = score
                best_model = candidate_model
                best_type = model_name
        if best_model is None:
            logger.warning("Failed to select a best model. Falling back to logistic regression.")
            best_model = models['logistic'].fit(X_norm, y)
            best_type = 'logistic'
        # Persist the chosen model and scaling parameters
        try:
            joblib.dump(best_model, MODEL_PKL)
            model_metadata = {
                'model_type': best_type,
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(),
                'feature_names': [
                    'score', 'confidence', 'session_id', 'btc_dom',
                    'fear_greed', 'sent_conf', 'pattern_len'
                ],
            }
            with open(MODEL_JSON, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            logger.info("Best model (%s) saved to %s with metadata %s", best_type, MODEL_PKL, MODEL_JSON)
        except Exception as e:
            logger.warning("Failed to save model: %s", e, exc_info=True)
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
                'fear_greed', 'sent_conf', 'pattern_len'
            ],
        }
        try:
            with open(MODEL_JSON, 'w') as f:
                json.dump(model_data, f, indent=2)
            logger.info("Manual model saved to %s", MODEL_JSON)
        except Exception as e:
            logger.warning("Failed to save manual model: %s", e, exc_info=True)


def _load_model_metadata() -> Dict[str, Any]:
    """Load model metadata from the JSON file, if present."""
    if not os.path.exists(MODEL_JSON):
        return {}
    try:
        with open(MODEL_JSON, 'r') as f:
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
    """
    Predict the probability of a trade succeeding based on current features.

    The function loads the previously trained model.  If a pickled
    sklearn model exists (``ml_model.pkl``) it is used; otherwise it
    falls back to the manual logistic regression defined in
    ``ml_model.json``.  In case of any loading errors, a neutral
    probability of 0.5 is returned.
    """
    metadata = _load_model_metadata()
    if not metadata:
        return 0.5
    x = np.array(_prepare_feature_vector(score, confidence, session, btc_d, fg, sentiment_conf, pattern))
    model_type = metadata.get('model_type', 'manual')
    try:
        # Use sklearn model if available
        if model_type in {'logistic', 'random_forest', 'gradient_boosting', 'mlp', 'xgboost', 'lightgbm'} and SKLEARN_AVAILABLE and os.path.exists(MODEL_PKL):
            # Load scaler statistics
            mean = np.array(metadata.get('scaler_mean'))
            scale = np.array(metadata.get('scaler_scale'))
            x_norm = (x - mean) / scale
            # Load model
            clf = joblib.load(MODEL_PKL)
            # Some classifiers (e.g., GradientBoosting) expect 2D array
            x_norm_2d = x_norm.reshape(1, -1)
            proba = clf.predict_proba(x_norm_2d)
            # Class 1 is success (win); index may vary but binary classification returns shape (n,2)
            prob = float(proba[0][1])
            return prob
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
