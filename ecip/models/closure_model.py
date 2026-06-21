"""
ECIP Closure Predictor
======================
CatBoost classifier that predicts road-closure probability.

Design choices
--------------
- Binary classification on ``requires_road_closure`` — a real column
  in the dataset (not synthetic).
- **Class imbalance**: True=676, False=7497 (~8.3 % positive).
  Handled via ``auto_class_weights='Balanced'``.
- Outputs calibrated probabilities (CatBoost internal Platt scaling).
- Uses the same feature set as the duration model for consistency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from catboost import CatBoostClassifier, Pool

from ecip.config import ALL_FEATURES, CATEGORICAL_FEATURES, MODELS_DIR

logger = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "closure_model.cbm"


def _get_cat_indices(feature_list: list[str]) -> list[int]:
    """Return indices of categorical features within the feature list."""
    return [i for i, f in enumerate(feature_list) if f in CATEGORICAL_FEATURES]


def create_model(**kwargs) -> CatBoostClassifier:
    """Create a CatBoost classifier with sensible defaults."""
    defaults = dict(
        iterations=800,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=3,
        eval_metric="AUC",
        auto_class_weights="Balanced",
        early_stopping_rounds=50,
        verbose=100,
        random_seed=42,
    )
    defaults.update(kwargs)
    return CatBoostClassifier(**defaults)


def train(
    X_train: np.ndarray | "pd.DataFrame",
    y_train: np.ndarray,
    X_val: Optional[np.ndarray | "pd.DataFrame"] = None,
    y_val: Optional[np.ndarray] = None,
    feature_names: Optional[list[str]] = None,
    **model_kwargs,
) -> CatBoostClassifier:
    """
    Train the closure classifier.

    Parameters
    ----------
    X_train, y_train : Training data.  ``y_train`` is binary (0/1).
    X_val, y_val : Optional validation set.
    feature_names : Column names.

    Returns
    -------
    Fitted CatBoostClassifier.
    """
    feature_names = feature_names or ALL_FEATURES
    cat_idx = _get_cat_indices(feature_names)

    model = create_model(**model_kwargs)

    train_pool = Pool(X_train, label=y_train.astype(int), cat_features=cat_idx,
                      feature_names=feature_names)

    eval_pool = None
    if X_val is not None and y_val is not None:
        eval_pool = Pool(X_val, label=y_val.astype(int), cat_features=cat_idx,
                         feature_names=feature_names)

    model.fit(train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None)

    best_auc = model.get_best_score().get("validation", {}).get("AUC", float("nan"))
    logger.info(
        "Closure model trained — %d iterations, best AUC: %.4f",
        model.tree_count_, best_auc,
    )
    return model


def predict_proba(model: CatBoostClassifier, X: np.ndarray | "pd.DataFrame") -> np.ndarray:
    """Predict closure probability (returns the positive-class probability)."""
    probas = model.predict_proba(X)
    if probas.ndim == 2:
        return probas[:, 1]
    return probas


def save(model: CatBoostClassifier, path: Path | None = None):
    """Save model to disk."""
    path = path or MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info("Closure model saved to %s", path)


def load(path: Path | None = None) -> CatBoostClassifier:
    """Load model from disk."""
    path = path or MODEL_PATH
    model = CatBoostClassifier()
    model.load_model(str(path))
    logger.info("Closure model loaded from %s", path)
    return model
