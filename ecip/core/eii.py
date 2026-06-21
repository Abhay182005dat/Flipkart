"""
Event Impact Index (EII)
========================
Deterministic, auditable composite metric derived from real model outputs.

Revised formula (adapted to actual dataset — no severity column):

    EII = 0.40 × Duration Risk
        + 0.35 × Closure Risk
        + 0.10 × Priority Risk          (binary: High=1, Low=0)
        + 0.15 × Location Risk          (corridor density + closure rate)

    Normalised to 0–100  →  Low / Medium / High / Critical

Design principles
-----------------
- **Never trained**: always derived from real predictions.
- **Decomposable**: every component can be inspected.
- **Tunable**: weights live in config.py and can be adjusted per city.

Weight calibration strategy (per user refinement #4)
----------------------------------------------------
1. Initial weights are domain-expert estimates (above).
2. After 200+ post-event outcomes are collected, optimise weights via
   linear regression: actual_outcome_severity ≈ Σ(w_i × component_i).
3. Quarterly review with traffic-authority stakeholders.
4. A/B test weight changes on a subset of events before full rollout.
"""

from __future__ import annotations

from ecip.config import EII_LEVELS, EII_MAX_DURATION_MIN, EII_WEIGHTS


class EventImpactIndex:
    """
    Compute the Event Impact Index from model outputs.

    >>> eii = EventImpactIndex()
    >>> result = eii.compute(
    ...     predicted_duration_min=145,
    ...     closure_probability=0.82,
    ...     priority_is_high=True,
    ...     location_risk=0.6,
    ... )
    >>> result["eii_score"]
    65.2
    """

    def __init__(
        self,
        weights: dict | None = None,
        max_duration_min: float = EII_MAX_DURATION_MIN,
        levels: list | None = None,
    ):
        self.weights = weights or dict(EII_WEIGHTS)
        self.max_duration = max_duration_min
        self.levels = levels or list(EII_LEVELS)

    def compute(
        self,
        predicted_duration_min: float,
        closure_probability: float,
        priority_is_high: bool,
        location_risk: float,
    ) -> dict:
        """
        Compute EII from model outputs.

        Parameters
        ----------
        predicted_duration_min : Predicted event duration (minutes).
        closure_probability : Model-predicted road-closure probability (0–1).
        priority_is_high : True if the event priority is "high".
        location_risk : Pre-computed corridor risk score (0–1).

        Returns
        -------
        dict with keys:
            eii_score (float 0–100), eii_level (str), eii_color (str),
            components (dict), explanation (str).
        """
        # ── Normalise components to 0-1 ───────────────────────────────
        duration_risk = min(predicted_duration_min / self.max_duration, 1.0)
        closure_risk = max(0.0, min(closure_probability, 1.0))
        priority_risk = 1.0 if priority_is_high else 0.0
        loc_risk = max(0.0, min(location_risk, 1.0))

        # ── Weighted composite ────────────────────────────────────────
        raw_eii = (
            self.weights["duration"] * duration_risk
            + self.weights["closure"] * closure_risk
            + self.weights["priority"] * priority_risk
            + self.weights["location"] * loc_risk
        )

        eii_score = round(raw_eii * 100, 1)

        # ── Classify ──────────────────────────────────────────────────
        eii_level, eii_color = "Critical", "red"
        for threshold, level, color in self.levels:
            if eii_score <= threshold:
                eii_level, eii_color = level, color
                break

        # ── Components dict ───────────────────────────────────────────
        components = {
            "duration_risk": round(duration_risk, 3),
            "closure_risk": round(closure_risk, 3),
            "priority_risk": round(priority_risk, 3),
            "location_risk": round(loc_risk, 3),
        }

        # ── Human-readable explanation ────────────────────────────────
        contributions = []
        for name, value in components.items():
            weight_key = name.replace("_risk", "")
            weight = self.weights.get(weight_key, 0)
            contribution = weight * value * 100
            contributions.append((name, contribution, value))

        contributions.sort(key=lambda x: -x[1])

        explanation_parts = []
        for name, contribution, value in contributions[:3]:
            readable = name.replace("_", " ").title()
            explanation_parts.append(
                f"{readable} contributes {contribution:.1f} pts (value: {value:.2f})"
            )

        explanation = (
            f"EII is {eii_score} ({eii_level}). "
            + "; ".join(explanation_parts) + "."
        )

        return {
            "eii_score": eii_score,
            "eii_level": eii_level,
            "eii_color": eii_color,
            "components": components,
            "explanation": explanation,
        }

    def compute_from_actuals(
        self,
        actual_duration_min: float,
        actual_closure: bool,
        priority_is_high: bool,
        location_risk: float,
    ) -> dict:
        """
        Recompute EII from actual post-event outcomes.
        Used for feedback-loop accuracy tracking.
        """
        return self.compute(
            predicted_duration_min=actual_duration_min,
            closure_probability=1.0 if actual_closure else 0.0,
            priority_is_high=priority_is_high,
            location_risk=location_risk,
        )
