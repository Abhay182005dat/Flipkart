"""
ECIP Duration Predictor
=======================
CatBoost regressor that predicts event duration in minutes.

Design choices
--------------
- **CatBoost** over XGBoost/LightGBM: native categorical handling for
  high-cardinality features (corridor, event_cause, veh_type) without
  one-hot explosion on 8 K records.
- **Log-transform** the target: duration is right-skewed
  (median 53 min, mean 516 min, max 9920 min).
- **Ordered boosting**: CatBoost's default reduces over-fitting on
  small datasets compared to standard GBDT.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from catboost import CatBoostRegressor, Pool

from ecip.config import ALL_FEATURES, CATEGORICAL_FEATURES, MODELS_DIR

logger = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "duration_model.cbm"
META_PATH = MODELS_DIR / "duration_meta.pkl"


def _get_cat_indices(feature_list: list[str]) -> list[int]:
    """Return indices of categorical features within the feature list."""
    return [i for i, f in enumerate(feature_list) if f in CATEGORICAL_FEATURES]


def create_model(**kwargs) -> CatBoostRegressor:
    """Create a CatBoost regressor with sensible defaults."""
    defaults = dict(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        eval_metric="RMSE",
        early_stopping_rounds=50,
        verbose=100,
        random_seed=42,
    )
    defaults.update(kwargs)
    return CatBoostRegressor(**defaults)


def train(
    X_train: np.ndarray | "pd.DataFrame",
    y_train: np.ndarray,
    X_val: Optional[np.ndarray | "pd.DataFrame"] = None,
    y_val: Optional[np.ndarray] = None,
    feature_names: Optional[list[str]] = None,
    **model_kwargs,
) -> CatBoostRegressor:
    """
    Train the duration model.

    Parameters
    ----------
    X_train, y_train : Training data.  ``y_train`` should be raw minutes;
        log-transform is applied internally.
    X_val, y_val : Optional validation set.
    feature_names : Column names (defaults to ALL_FEATURES).

    Returns
    -------
    Fitted CatBoostRegressor.
    """
    feature_names = feature_names or ALL_FEATURES
    cat_idx = _get_cat_indices(feature_names)

    model = create_model(**model_kwargs)

    # Log-transform target
    y_train_log = np.log1p(y_train)

    train_pool = Pool(X_train, label=y_train_log, cat_features=cat_idx,
                      feature_names=feature_names)

    eval_pool = None
    if X_val is not None and y_val is not None:
        y_val_log = np.log1p(y_val)
        eval_pool = Pool(X_val, label=y_val_log, cat_features=cat_idx,
                         feature_names=feature_names)

    model.fit(train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None)

    logger.info(
        "Duration model trained — %d iterations, best RMSE (log): %.4f",
        model.tree_count_,
        model.get_best_score().get("validation", {}).get("RMSE", float("nan")),
    )
    return model


def predict(model: CatBoostRegressor, X: np.ndarray | "pd.DataFrame") -> np.ndarray:
    """Predict duration in minutes (inverse log-transform)."""
    log_pred = model.predict(X)
    return np.expm1(log_pred).clip(min=1.0)


def save(model: CatBoostRegressor, path: Path | None = None):
    """Save model to disk."""
    path = path or MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info("Duration model saved to %s", path)


def load(path: Path | None = None) -> CatBoostRegressor:
    """Load model from disk."""
    path = path or MODEL_PATH
    model = CatBoostRegressor()
    model.load_model(str(path))
    logger.info("Duration model loaded from %s", path)
    return model
