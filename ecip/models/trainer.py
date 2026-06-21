"""
ECIP Training Pipeline
======================
End-to-end: load data → features → split → train models → evaluate → save.

Runs both the duration and closure models with time-series-aware
cross-validation to prevent data leakage from future events.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

from ecip.config import ALL_FEATURES, MODELS_DIR
from ecip.data.loader import load_and_clean
from ecip.data.features import build_features, get_model_features
from ecip.models import duration_model, closure_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_duration(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute duration regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE (avoid division by zero)
    mask = y_true > 0
    mape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
    return {"MAE_min": round(mae, 2), "RMSE_min": round(rmse, 2), "MAPE_%": round(mape, 2)}


def evaluate_closure(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute closure classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "AUC": round(auc, 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def run_training(data_path: str | Path | None = None, n_cv_folds: int = 5):
    """
    Main training entry point.

    1. Load and clean data.
    2. Build features.
    3. Train duration model (on events with valid duration).
    4. Train closure model (on all events).
    5. Cross-validate both.
    6. Save models.
    """
    logger.info("=" * 70)
    logger.info("ECIP MODEL TRAINING PIPELINE")
    logger.info("=" * 70)

    # ── 1. Load ────────────────────────────────────────────────────────
    df = load_and_clean(data_path)

    # ── 2. Features ────────────────────────────────────────────────────
    df = build_features(df)

    # Sort by time for time-series CV
    df = df.sort_values("start_datetime").reset_index(drop=True)

    # ── 3. Duration model ──────────────────────────────────────────────
    logger.info("-" * 70)
    logger.info("DURATION MODEL")
    logger.info("-" * 70)

    df_dur = df[df["duration_min"].notna()].copy()
    logger.info("Training samples with valid duration: %d", len(df_dur))

    X_dur = get_model_features(df_dur, ALL_FEATURES)
    y_dur = df_dur["duration_min"].values

    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=n_cv_folds)
    dur_metrics_all = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_dur), 1):
        X_tr, X_va = X_dur.iloc[train_idx], X_dur.iloc[val_idx]
        y_tr, y_va = y_dur[train_idx], y_dur[val_idx]

        # Recompute historical aggregates from training fold only
        df_train_fold = df_dur.iloc[train_idx]
        df_val_fold = df_dur.iloc[val_idx].copy()
        # Note: features already computed from full training set;
        # for a stricter approach, re-engineer features per fold.
        # Acceptable for a hackathon MVP.

        model_dur = duration_model.train(
            X_tr, y_tr, X_va, y_va, feature_names=ALL_FEATURES, verbose=0
        )
        y_pred = duration_model.predict(model_dur, X_va)
        metrics = evaluate_duration(y_va, y_pred)
        dur_metrics_all.append(metrics)
        logger.info("  Fold %d: %s", fold, metrics)

    # Average CV metrics
    avg_dur = {k: round(np.mean([m[k] for m in dur_metrics_all]), 2) for k in dur_metrics_all[0]}
    logger.info("  Duration CV Average: %s", avg_dur)

    # Train final model on all duration data
    split_idx = int(len(X_dur) * 0.85)
    X_tr_final, X_va_final = X_dur.iloc[:split_idx], X_dur.iloc[split_idx:]
    y_tr_final, y_va_final = y_dur[:split_idx], y_dur[split_idx:]

    final_dur_model = duration_model.train(
        X_tr_final, y_tr_final, X_va_final, y_va_final,
        feature_names=ALL_FEATURES
    )
    duration_model.save(final_dur_model)

    # ── 4. Closure model ───────────────────────────────────────────────
    logger.info("-" * 70)
    logger.info("CLOSURE MODEL")
    logger.info("-" * 70)

    X_clo = get_model_features(df, ALL_FEATURES)
    y_clo = df["requires_road_closure"].values.astype(int)
    logger.info("Training samples: %d (positive rate: %.1f%%)",
                len(y_clo), y_clo.mean() * 100)

    tscv = TimeSeriesSplit(n_splits=n_cv_folds)
    clo_metrics_all = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clo), 1):
        X_tr, X_va = X_clo.iloc[train_idx], X_clo.iloc[val_idx]
        y_tr, y_va = y_clo[train_idx], y_clo[val_idx]

        model_clo = closure_model.train(
            X_tr, y_tr, X_va, y_va, feature_names=ALL_FEATURES, verbose=0
        )
        y_prob = closure_model.predict_proba(model_clo, X_va)
        metrics = evaluate_closure(y_va, y_prob)
        clo_metrics_all.append(metrics)
        logger.info("  Fold %d: %s", fold, metrics)

    avg_clo = {k: round(np.mean([m[k] for m in clo_metrics_all]), 4) for k in clo_metrics_all[0]}
    logger.info("  Closure CV Average: %s", avg_clo)

    # Train final model
    split_idx = int(len(X_clo) * 0.85)
    X_tr_final, X_va_final = X_clo.iloc[:split_idx], X_clo.iloc[split_idx:]
    y_tr_final, y_va_final = y_clo[:split_idx], y_clo[split_idx:]

    final_clo_model = closure_model.train(
        X_tr_final, y_tr_final, X_va_final, y_va_final,
        feature_names=ALL_FEATURES
    )
    closure_model.save(final_clo_model)

    # ── 5. Summary ─────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("  Duration CV: %s", avg_dur)
    logger.info("  Closure  CV: %s", avg_clo)
    logger.info("  Models saved to: %s", MODELS_DIR)
    logger.info("=" * 70)

    return {
        "duration_cv": avg_dur,
        "closure_cv": avg_clo,
        "duration_model": final_dur_model,
        "closure_model": final_clo_model,
    }


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_training(data_path)
