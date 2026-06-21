"""
Response Priority Engine
========================
Maps EII + event characteristics to a 4-tier operational priority system.

    P1 🔴 DEPLOY IMMEDIATELY      — EII ≥ 75 or emergency override
    P2 🟠 DEPLOY WITHIN 15 MIN    — EII 50-74 or elevated unplanned
    P3 🟡 MONITOR                  — EII 25-49
    P4 🟢 OBSERVE ONLY            — EII < 25

Answers: "Which event gets attention first?"
"""

from __future__ import annotations

from ecip.config import PRIORITY_TIERS


class ResponsePriorityEngine:
    """
    Assigns an operational response-priority tier (P1–P4) to each event
    based on its EII score, event type, and closure probability.
    """

    def __init__(self):
        self.tiers = dict(PRIORITY_TIERS)

    def compute(
        self,
        eii_score: float,
        eii_level: str,
        event_type: str,
        closure_prob: float,
        simultaneous_events: int = 0,
    ) -> dict:
        """
        Determine response priority tier.

        Rules (evaluated in order — first match wins):
        1. P1 if EII ≥ 75 (Critical)
        2. P1 if unplanned AND closure_prob > 0.8 (emergency override)
        3. P2 if EII ≥ 50 (High)
        4. P2 if unplanned AND closure_prob > 0.5 (elevated unplanned)
        5. P3 if EII ≥ 25 (Medium)
        6. P4 otherwise (Low)

        Modifier: >5 simultaneous events → escalate P3 to P2.
        """
        if eii_score >= 75:
            priority = 1
            reason = f"EII score {eii_score} is Critical (≥75)"
        elif event_type == "unplanned" and closure_prob > 0.8:
            priority = 1
            reason = (
                f"Unplanned event with {closure_prob:.0%} closure "
                f"probability — emergency override"
            )
        elif eii_score >= 50:
            priority = 2
            reason = f"EII score {eii_score} is High (50–74)"
        elif event_type == "unplanned" and closure_prob > 0.5:
            priority = 2
            reason = (
                f"Unplanned event with {closure_prob:.0%} closure "
                f"probability — elevated response"
            )
        elif eii_score >= 25:
            priority = 3
            reason = f"EII score {eii_score} is Medium (25–49)"
        else:
            priority = 4
            reason = f"EII score {eii_score} is Low (<25)"

        # Busy-period escalation
        if priority == 3 and simultaneous_events > 5:
            priority = 2
            reason += (
                f" — escalated P3→P2 due to {simultaneous_events} "
                f"simultaneous events"
            )

        tier = self.tiers[priority]

        return {
            "priority": priority,
            "label": tier["label"],
            "color": tier["color"],
            "icon": tier["icon"],
            "max_response_min": tier["max_response_min"],
            "reason": reason,
            "eii_score": eii_score,
            "eii_level": eii_level,
        }
