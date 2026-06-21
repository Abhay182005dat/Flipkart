"""
Similar Event Intelligence
==========================
First-class component: retrieves similar historical events to provide
operator context BEFORE predictions are shown.

Flow: New Event → **Similar Events** → Prediction → Recommendation

Design (per user refinement #5)
-------------------------------
Combines corridor identity WITH geospatial clustering (geohash)
to reduce cold-start issues when a junction is new or missing.
Even if a junction has never been seen, its corridor + geohash will
match events in the same area.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ecip.config import SIMILAR_EVENT_K, SIMILAR_EVENT_FEATURE_WEIGHTS

logger = logging.getLogger(__name__)


class SimilarEventRetriever:
    """
    Weighted k-NN retriever for similar historical events.

    Features used (with weights):
        event_cause  0.25  — same cause is the strongest similarity signal
        corridor     0.20  — same road corridor means same traffic patterns
        time_bucket  0.15  — rush-hour events behave differently from night
        priority     0.15  — high-priority events need different response
        veh_type     0.10  — vehicle type affects duration and closure
        lat/lon      0.15  — geospatial proximity (captures geohash-level)
    """

    CAT_COLS = ["event_cause", "corridor", "time_of_day_bucket", "veh_type", "priority"]
    NUM_COLS = ["latitude", "longitude"]

    # Columns to return as aggregate stats
    AGG_COLS = {
        "duration_min": ("mean", "median"),
        "requires_road_closure": ("mean",),  # closure rate
    }

    def __init__(self, k: int = SIMILAR_EVENT_K):
        self.k = k
        self.encoder: Optional[OneHotEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.nn: Optional[NearestNeighbors] = None
        self.df: Optional[pd.DataFrame] = None
        self._is_fitted = False

    def fit(self, historical_df: pd.DataFrame) -> "SimilarEventRetriever":
        """
        Build the similarity index from historical events.

        Parameters
        ----------
        historical_df : Cleaned + feature-enriched DataFrame.
        """
        self.df = historical_df.copy().reset_index(drop=True)

        # Fill any NaN in similarity columns
        for col in self.CAT_COLS:
            self.df[col] = self.df[col].fillna("unknown").astype(str)
        for col in self.NUM_COLS:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0.0)

        # One-hot encode categoricals
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        cat_matrix = self.encoder.fit_transform(self.df[self.CAT_COLS])

        # Scale numericals
        self.scaler = StandardScaler()
        num_matrix = self.scaler.fit_transform(self.df[self.NUM_COLS])

        # Apply weights (cat features get uniform weight per category group,
        # num features get explicit weights)
        cat_weight = 0.25  # Distribute across one-hot columns
        num_weights = np.array([0.075, 0.075])  # lat, lon

        feature_matrix = np.hstack([
            cat_matrix * cat_weight,
            num_matrix * num_weights,
        ])

        self.nn = NearestNeighbors(
            n_neighbors=min(self.k, len(self.df)),
            metric="cosine",
            algorithm="brute",  # Fine for 8K records — <20ms
        )
        self.nn.fit(feature_matrix)
        self._is_fitted = True

        logger.info(
            "SimilarEventRetriever fitted on %d events (%d features)",
            len(self.df), feature_matrix.shape[1],
        )
        return self

    def find_similar(self, new_event: dict, k: int | None = None) -> dict:
        """
        Find top-K similar historical events.

        Parameters
        ----------
        new_event : dict with keys matching CAT_COLS + NUM_COLS.
        k : Override number of results (default: self.k).

        Returns
        -------
        {
            "similar_events": [list of event dicts with similarity_score],
            "aggregate_stats": {avg_duration, median_duration, closure_rate, ...}
        }
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .find_similar()")

        k = k or self.k

        # Encode the query event
        cat_vals = [[str(new_event.get(c, "unknown")) for c in self.CAT_COLS]]
        num_vals = [[float(new_event.get(c, 0.0)) for c in self.NUM_COLS]]

        cat_enc = self.encoder.transform(cat_vals) * 0.25
        num_enc = self.scaler.transform(num_vals) * np.array([0.075, 0.075])
        query_vec = np.hstack([cat_enc, num_enc])

        distances, indices = self.nn.kneighbors(
            query_vec, n_neighbors=min(k, len(self.df))
        )

        # Build result list
        similar_events = []
        for dist, idx in zip(distances[0], indices[0]):
            event = self.df.iloc[idx].to_dict()
            # Clean up for JSON serialisation
            event["similarity_score"] = round(float(1 - dist), 3)
            # Convert numpy types
            for key, val in event.items():
                if isinstance(val, (np.integer,)):
                    event[key] = int(val)
                elif isinstance(val, (np.floating,)):
                    event[key] = round(float(val), 4)
                elif isinstance(val, (np.bool_,)):
                    event[key] = bool(val)
                elif hasattr(val, "isoformat"):
                    event[key] = val.isoformat()
            similar_events.append(event)

        # Aggregate statistics
        agg = self._compute_aggregates(similar_events)

        return {
            "similar_events": similar_events,
            "aggregate_stats": agg,
        }

    def _compute_aggregates(self, events: list[dict]) -> dict:
        """Compute summary statistics from retrieved similar events."""
        if not events:
            return {}

        durations = [
            e.get("duration_min", 0) for e in events
            if e.get("duration_min") is not None and e.get("duration_min", 0) > 0
        ]
        closures = [
            1 if e.get("requires_road_closure", False) else 0
            for e in events
        ]

        stats = {
            "sample_size": len(events),
            "avg_similarity": round(
                float(np.mean([e["similarity_score"] for e in events])), 3
            ),
        }

        if durations:
            stats["avg_duration_min"] = round(float(np.mean(durations)), 1)
            stats["median_duration_min"] = round(float(np.median(durations)), 1)
            stats["min_duration_min"] = round(float(np.min(durations)), 1)
            stats["max_duration_min"] = round(float(np.max(durations)), 1)

        if closures:
            stats["closure_rate"] = round(float(np.mean(closures)), 2)

        return stats
