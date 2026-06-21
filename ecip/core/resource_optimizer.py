"""
Resource Optimizer
==================
Multi-event, resource-constrained allocation using Integer Linear
Programming (ILP) via Google OR-Tools.

Allocates limited personnel and barricades across simultaneous events
to maximise total risk mitigation, weighted by EII and response priority.
"""

from __future__ import annotations

import logging
from typing import Optional

from ecip.config import EII_RESOURCE_MAP

logger = logging.getLogger(__name__)


class ResourceOptimizer:
    """
    ILP-based resource optimizer.

    Parameters
    ----------
    total_personnel : Total available officers.
    total_barricades : Total available barricades.
    """

    ALPHA = 0.6   # Personnel effectiveness weight
    BETA = 0.4    # Barricade effectiveness weight
    ESCALATION_PENALTY = 50

    def __init__(self, total_personnel: int = 50, total_barricades: int = 80):
        self.P_max = total_personnel
        self.B_max = total_barricades

    def optimize(self, events: list[dict]) -> dict:
        """
        Allocate resources across simultaneous events.

        Parameters
        ----------
        events : list of dicts, each with:
            event_id, eii_score, eii_level, closure_prob,
            duration_hours, response_priority

        Returns
        -------
        dict with "allocations" (list) and "summary" (dict).
        """
        if not events:
            return {"allocations": [], "summary": self._empty_summary()}

        # Try ILP solver first; fall back to greedy
        try:
            from ortools.linear_solver import pywraplp
            return self._solve_ilp(events, pywraplp)
        except Exception as e:
            logger.warning("ILP solver failed (%s), falling back to greedy", e)
            return self._greedy_allocation(events)

    def _solve_ilp(self, events: list[dict], pywraplp) -> dict:
        """Solve via ILP using OR-Tools SCIP solver."""
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            return self._greedy_allocation(events)

        n = len(events)

        # Derive resource bounds from EII level
        for ev in events:
            bounds = EII_RESOURCE_MAP.get(
                ev.get("eii_level", "Medium"), EII_RESOURCE_MAP["Medium"]
            )
            ev["p_min"] = bounds["p_min"]
            ev["p_max"] = bounds["p_max"]
            ev["b_min"] = bounds["b_min"]
            ev["b_max"] = bounds["b_max"]

        # Decision variables
        p = [solver.IntVar(events[i]["p_min"], events[i]["p_max"], f"p_{i}") for i in range(n)]
        b = [solver.IntVar(events[i]["b_min"], events[i]["b_max"], f"b_{i}") for i in range(n)]
        e = [solver.BoolVar(f"e_{i}") for i in range(n)]

        # C1: Total personnel
        solver.Add(sum(p[i] for i in range(n)) <= self.P_max)

        # C2: Total barricades
        solver.Add(sum(b[i] for i in range(n)) <= self.B_max)

        # C7: Closure → extra barricades
        for i in range(n):
            if events[i].get("closure_prob", 0) > 0.5:
                solver.Add(b[i] >= events[i]["b_min"] + 2)

        # C9: P1 events get guaranteed extra personnel
        for i in range(n):
            if events[i].get("response_priority", 4) == 1:
                solver.Add(p[i] >= events[i]["p_min"] + 2)

        # Objective: maximise risk mitigation
        objective = solver.Objective()
        for i in range(n):
            prio = events[i].get("response_priority", 4)
            priority_boost = 5 - prio + 1  # P1=5, P4=2
            w_i = (
                events[i].get("eii_score", 50)
                * events[i].get("duration_hours", 1)
                * priority_boost
            )

            p_max = max(events[i]["p_max"], 1)
            b_max = max(events[i]["b_max"], 1)

            objective.SetCoefficient(p[i], w_i * self.ALPHA / p_max)
            objective.SetCoefficient(b[i], w_i * self.BETA / b_max)
            objective.SetCoefficient(e[i], -self.ESCALATION_PENALTY)

        objective.SetMaximization()
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            allocations = []
            total_p, total_b = 0, 0
            for i in range(n):
                alloc_p = int(p[i].solution_value())
                alloc_b = int(b[i].solution_value())
                total_p += alloc_p
                total_b += alloc_b
                allocations.append({
                    "event_id": events[i].get("event_id", f"event_{i}"),
                    "personnel": alloc_p,
                    "barricades": alloc_b,
                    "escalated": bool(e[i].solution_value()),
                    "response_priority": events[i].get("response_priority", 4),
                    "eii_score": events[i].get("eii_score", 0),
                    "eii_level": events[i].get("eii_level", "Medium"),
                    "explanation": self._explain_allocation(events[i], alloc_p, alloc_b),
                })

            allocations.sort(
                key=lambda x: (x["response_priority"], -x["eii_score"])
            )

            return {
                "allocations": allocations,
                "summary": {
                    "total_events": n,
                    "total_personnel_used": total_p,
                    "total_barricades_used": total_b,
                    "personnel_remaining": self.P_max - total_p,
                    "barricades_remaining": self.B_max - total_b,
                    "personnel_utilization_pct": round(total_p / self.P_max * 100, 1),
                    "barricades_utilization_pct": round(total_b / self.B_max * 100, 1),
                    "solver": "ILP (SCIP)",
                },
            }
        else:
            logger.warning("ILP solver returned non-optimal status: %s", status)
            return self._greedy_allocation(events)

    def _greedy_allocation(self, events: list[dict]) -> dict:
        """Greedy fallback: prioritise by response priority then EII."""
        indexed = list(enumerate(events))
        indexed.sort(
            key=lambda x: (x[1].get("response_priority", 4), -x[1].get("eii_score", 0))
        )

        remaining_p = self.P_max
        remaining_b = self.B_max
        allocations = [None] * len(events)

        for idx, ev in indexed:
            bounds = EII_RESOURCE_MAP.get(
                ev.get("eii_level", "Medium"), EII_RESOURCE_MAP["Medium"]
            )
            alloc_p = min(bounds["p_min"], remaining_p)
            alloc_b = min(bounds["b_min"], remaining_b)
            remaining_p -= alloc_p
            remaining_b -= alloc_b

            allocations[idx] = {
                "event_id": ev.get("event_id", f"event_{idx}"),
                "personnel": alloc_p,
                "barricades": alloc_b,
                "escalated": remaining_p == 0,
                "response_priority": ev.get("response_priority", 4),
                "eii_score": ev.get("eii_score", 0),
                "eii_level": ev.get("eii_level", "Medium"),
                "explanation": self._explain_allocation(ev, alloc_p, alloc_b),
            }

        total_p = self.P_max - remaining_p
        total_b = self.B_max - remaining_b

        return {
            "allocations": [a for a in allocations if a is not None],
            "summary": {
                "total_events": len(events),
                "total_personnel_used": total_p,
                "total_barricades_used": total_b,
                "personnel_remaining": remaining_p,
                "barricades_remaining": remaining_b,
                "personnel_utilization_pct": round(total_p / max(self.P_max, 1) * 100, 1),
                "barricades_utilization_pct": round(total_b / max(self.B_max, 1) * 100, 1),
                "solver": "Greedy (fallback)",
            },
        }

    def _explain_allocation(self, event: dict, personnel: int, barricades: int) -> str:
        """Generate human-readable explanation for an allocation."""
        eii = event.get("eii_score", 0)
        level = event.get("eii_level", "Medium")
        prio = event.get("response_priority", 4)
        closure = event.get("closure_prob", 0)

        parts = [f"EII {eii} ({level}), Priority P{prio}."]
        parts.append(f"Assigned {personnel} officers and {barricades} barricades.")

        if closure > 0.5:
            parts.append(f"High closure probability ({closure:.0%}) → extra barricades.")
        if prio == 1:
            parts.append("P1 event → guaranteed minimum + 2 extra officers.")

        return " ".join(parts)

    def _empty_summary(self) -> dict:
        return {
            "total_events": 0,
            "total_personnel_used": 0,
            "total_barricades_used": 0,
            "personnel_remaining": self.P_max,
            "barricades_remaining": self.B_max,
            "personnel_utilization_pct": 0.0,
            "barricades_utilization_pct": 0.0,
            "solver": "N/A",
        }
