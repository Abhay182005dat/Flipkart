"""
ECIP Application State
======================
Holds loaded models, feature pipeline, similar-event index, EII engine,
and all core components.  Created once at startup and shared across
all API requests via ``app.state.ecip``.
"""

from __future__ import annotations

import logging

import pandas as pd
import numpy as np

from ecip.config import ALL_FEATURES, MODELS_DIR
from ecip.data.loader import load_and_clean
from ecip.data.features import build_features, get_model_features
from ecip.models import duration_model, closure_model
from ecip.core.eii import EventImpactIndex
from ecip.core.priority import ResponsePriorityEngine
from ecip.core.similar_events import SimilarEventRetriever
from ecip.core.scenario_planner import ScenarioPlanningEngine
from ecip.core.resource_optimizer import ResourceOptimizer

logger = logging.getLogger(__name__)


class AppState:
    """Singleton-style container for all ECIP runtime state."""

    def __init__(self):
        self.df: pd.DataFrame | None = None         # Cleaned + featured data
        self.dur_model = None                         # CatBoost duration
        self.clo_model = None                         # CatBoost closure
        self.eii = EventImpactIndex()
        self.priority_engine = ResponsePriorityEngine()
        self.similar_retriever = SimilarEventRetriever()
        self.scenario_planner = ScenarioPlanningEngine()
        self.optimizer = ResourceOptimizer()
        self.feature_names = ALL_FEATURES
        self._is_loaded = False

    def load_all(self):
        """Load models + data + build indices."""
        # 1. Load and feature-engineer the dataset
        logger.info("Loading and processing dataset…")
        self.df = build_features(load_and_clean())

        # 2. Load trained models
        logger.info("Loading models from %s", MODELS_DIR)
        self.dur_model = duration_model.load()
        self.clo_model = closure_model.load()

        # 3. Build similarity index
        logger.info("Building similar-event index…")
        self.similar_retriever.fit(self.df)

        self._is_loaded = True
        logger.info("All components loaded successfully.")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def predict_event(self, event_data: dict) -> dict:
        """
        Full prediction pipeline for a single event.

        Parameters
        ----------
        event_data : dict with raw event fields (event_cause, corridor,
            latitude, longitude, priority, veh_type, event_type,
            start_datetime, requires_road_closure).

        Returns
        -------
        Complete decision bundle: similar events, predictions, EII,
        priority, scenario plans, and explanations.
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded — call load_all() first")

        # ── Step 1: Build a 1-row DataFrame and compute features ──────
        event_df = pd.DataFrame([event_data])
        event_df = build_features(event_df, reference_df=self.df)
        X = get_model_features(event_df, self.feature_names)

        # ── Step 2: Similar Events (FIRST — before predictions) ───────
        similar = self.similar_retriever.find_similar(event_data)

        # ── Step 3: ML Predictions ────────────────────────────────────
        pred_duration = float(duration_model.predict(self.dur_model, X)[0])
        pred_closure = float(closure_model.predict_proba(self.clo_model, X)[0])

        # ── Step 4: EII Computation ───────────────────────────────────
        priority_is_high = str(event_data.get("priority", "low")).lower() == "high"
        location_risk = float(event_df["location_risk"].iloc[0])

        eii_result = self.eii.compute(
            predicted_duration_min=pred_duration,
            closure_probability=pred_closure,
            priority_is_high=priority_is_high,
            location_risk=location_risk,
        )

        # ── Step 5: Response Priority ─────────────────────────────────
        priority_result = self.priority_engine.compute(
            eii_score=eii_result["eii_score"],
            eii_level=eii_result["eii_level"],
            event_type=event_data.get("event_type", "unplanned"),
            closure_prob=pred_closure,
        )

        # ── Step 6: Scenario Planning (pre-compute standard what-ifs) ─
        baseline = {
            "duration_min": pred_duration,
            "closure_prob": pred_closure,
            "priority_is_high": priority_is_high,
            "location_risk": location_risk,
            "current_personnel": 0,
            "current_barricades": 0,
        }
        scenarios = self.scenario_planner.generate_standard_scenarios(baseline)

        # ── Assemble decision bundle ──────────────────────────────────
        return {
            "event_input": event_data,
            "similar_events": similar,
            "predictions": {
                "duration_min": round(pred_duration, 1),
                "closure_probability": round(pred_closure, 3),
            },
            "eii": eii_result,
            "response_priority": priority_result,
            "scenarios": scenarios,
        }

    def get_shap_explanation(self, event_data: dict) -> dict:
        """Generate SHAP explanations for a single event."""
        try:
            import shap

            event_df = pd.DataFrame([event_data])
            event_df = build_features(event_df, reference_df=self.df)
            X = get_model_features(event_df, self.feature_names)

            explanations = {}

            # Duration SHAP
            dur_explainer = shap.TreeExplainer(self.dur_model)
            dur_shap = dur_explainer.shap_values(X)
            dur_base = float(dur_explainer.expected_value)

            dur_impacts = sorted(
                zip(self.feature_names, dur_shap[0].tolist()),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:7]

            explanations["duration"] = {
                "base_value": round(np.expm1(dur_base), 1),
                "prediction": round(float(duration_model.predict(self.dur_model, X)[0]), 1),
                "top_features": [
                    {"feature": f, "impact": round(v, 3), "direction": "increases" if v > 0 else "decreases"}
                    for f, v in dur_impacts
                ],
            }

            # Closure SHAP
            clo_explainer = shap.TreeExplainer(self.clo_model)
            clo_shap = clo_explainer.shap_values(X)
            # Handle multi-output SHAP for classifiers
            if isinstance(clo_shap, list) and len(clo_shap) == 2:
                clo_shap_pos = clo_shap[1]
                clo_base = float(clo_explainer.expected_value[1])
            else:
                clo_shap_pos = clo_shap
                clo_base = float(clo_explainer.expected_value) if not hasattr(clo_explainer.expected_value, '__len__') else float(clo_explainer.expected_value[0])

            clo_impacts = sorted(
                zip(self.feature_names, clo_shap_pos[0].tolist()),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:7]

            explanations["closure"] = {
                "base_value": round(clo_base, 3),
                "prediction": round(float(closure_model.predict_proba(self.clo_model, X)[0]), 3),
                "top_features": [
                    {"feature": f, "impact": round(v, 3), "direction": "increases" if v > 0 else "decreases"}
                    for f, v in clo_impacts
                ],
            }

            return explanations

        except ImportError:
            return {"error": "SHAP not installed"}
        except Exception as e:
            logger.exception("SHAP explanation failed")
            return {"error": str(e)}
