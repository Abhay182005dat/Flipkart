"""
Scenario Planning Engine (formerly "Counterfactual Planning Engine")
====================================================================
Renamed per user refinement #2 — "Scenario Planning" is more
appropriate unless causal inference can be rigorously justified.

Allows operators to explore what-if questions:
  - "What if we deploy 2 more officers?"
  - "What if we add 3 barricades?"
  - "What if we close Road X proactively?"

Re-runs the EII pipeline under modified parameters and shows
projected outcome changes.
"""

from __future__ import annotations

from ecip.config import DEFAULT_ELASTICITIES
from ecip.core.eii import EventImpactIndex


class ScenarioPlanningEngine:
    """
    Explore resource-allocation scenarios and project EII changes.

    Uses learned resource-effectiveness elasticities (updated from
    post-event data) to estimate how additional resources reduce
    duration and closure risk.
    """

    def __init__(
        self,
        eii_calculator: EventImpactIndex | None = None,
        elasticities: dict | None = None,
    ):
        self.eii = eii_calculator or EventImpactIndex()
        self.eff = elasticities or dict(DEFAULT_ELASTICITIES)

    def simulate(self, baseline: dict, scenario: dict) -> dict:
        """
        Project outcomes under a hypothetical scenario.

        Parameters
        ----------
        baseline : Current prediction state.
            Required keys: duration_min, closure_prob, priority_is_high,
            location_risk, current_personnel, current_barricades
        scenario : Proposed changes.
            Keys: delta_personnel, delta_barricades, close_road (bool)

        Returns
        -------
        dict with baseline, projected, delta, and totals.
        """
        dp = scenario.get("delta_personnel", 0)
        db = scenario.get("delta_barricades", 0)
        close_road = scenario.get("close_road", False)

        # ── Duration adjustment ───────────────────────────────────────
        duration_factor = 1.0
        if dp != 0:
            duration_factor *= max(
                1 + self.eff["personnel_duration_elasticity"] * dp, 0.4
            )
        if db != 0:
            duration_factor *= max(
                1 + self.eff["barricade_duration_elasticity"] * db, 0.8
            )

        new_duration = baseline["duration_min"] * max(duration_factor, 0.3)

        # ── Closure probability adjustment ────────────────────────────
        if close_road:
            new_closure_prob = 1.0
            new_duration *= 0.75  # Controlled closure is shorter
        else:
            closure_delta = (
                self.eff["personnel_closure_elasticity"] * dp
                + self.eff["barricade_closure_elasticity"] * db
            )
            new_closure_prob = max(0.0, min(1.0, baseline["closure_prob"] + closure_delta))

        # ── Recompute EII ─────────────────────────────────────────────
        baseline_eii = self.eii.compute(
            predicted_duration_min=baseline["duration_min"],
            closure_probability=baseline["closure_prob"],
            priority_is_high=baseline["priority_is_high"],
            location_risk=baseline["location_risk"],
        )

        projected_eii = self.eii.compute(
            predicted_duration_min=new_duration,
            closure_probability=new_closure_prob,
            priority_is_high=baseline["priority_is_high"],
            location_risk=baseline["location_risk"],
        )

        base_dur = baseline["duration_min"]
        return {
            "scenario": scenario,
            "baseline": {
                "duration_min": round(base_dur, 1),
                "closure_prob": round(baseline["closure_prob"], 3),
                "eii_score": baseline_eii["eii_score"],
                "eii_level": baseline_eii["eii_level"],
            },
            "projected": {
                "duration_min": round(new_duration, 1),
                "closure_prob": round(new_closure_prob, 3),
                "eii_score": projected_eii["eii_score"],
                "eii_level": projected_eii["eii_level"],
            },
            "delta": {
                "duration_change_min": round(new_duration - base_dur, 1),
                "duration_change_pct": round(
                    (new_duration - base_dur) / max(base_dur, 1) * 100, 1
                ),
                "closure_change": round(new_closure_prob - baseline["closure_prob"], 3),
                "eii_change": round(
                    projected_eii["eii_score"] - baseline_eii["eii_score"], 1
                ),
                "eii_level_change": (
                    f"{baseline_eii['eii_level']} → {projected_eii['eii_level']}"
                    if baseline_eii["eii_level"] != projected_eii["eii_level"]
                    else "No level change"
                ),
            },
            "total_personnel": baseline.get("current_personnel", 0) + dp,
            "total_barricades": baseline.get("current_barricades", 0) + db,
        }

    def generate_standard_scenarios(self, baseline: dict) -> list[dict]:
        """Pre-compute a set of common what-if scenarios for dashboard display."""
        scenarios = [
            {"label": "Add 2 officers", "delta_personnel": 2, "delta_barricades": 0},
            {"label": "Add 4 officers", "delta_personnel": 4, "delta_barricades": 0},
            {"label": "Add 2 officers + 3 barricades", "delta_personnel": 2, "delta_barricades": 3},
            {"label": "Add 4 officers + 4 barricades", "delta_personnel": 4, "delta_barricades": 4},
            {"label": "Maximum deployment", "delta_personnel": 8, "delta_barricades": 10},
            {"label": "Proactive road closure", "delta_personnel": 0, "delta_barricades": 0, "close_road": True},
        ]

        results = []
        for s in scenarios:
            label = s.pop("label")
            result = self.simulate(baseline, s)
            result["label"] = label
            s["label"] = label  # Restore for reuse
            results.append(result)

        return results
