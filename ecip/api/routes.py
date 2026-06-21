"""
ECIP API Routes
===============
REST endpoints for the Event-Driven Congestion Intelligence Platform.

Endpoints
---------
POST /api/v1/events/predict        → Full prediction pipeline
POST /api/v1/events/explain        → SHAP explanations
POST /api/v1/events/similar        → Similar historical events
POST /api/v1/events/scenario       → Scenario planning (what-if)
POST /api/v1/optimize/resources    → Multi-event resource allocation
GET  /api/v1/stats                 → Dataset statistics
GET  /api/v1/events/list           → List recent events
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Request / Response Models ─────────────────────────────────────────

class EventInput(BaseModel):
    """Input for a single event prediction."""
    event_type: str = Field("unplanned", description="'planned' or 'unplanned'")
    event_cause: str = Field(..., description="e.g. 'vehicle_breakdown', 'accident', 'political_rally'")
    latitude: float = Field(..., description="Latitude (Bengaluru range: 12.5–13.5)")
    longitude: float = Field(..., description="Longitude (Bengaluru range: 77.0–78.0)")
    corridor: str = Field("Non-corridor", description="Road corridor name")
    priority: str = Field("high", description="'high' or 'low'")
    veh_type: str = Field("unknown", description="Vehicle type involved")
    start_datetime: str = Field(..., description="ISO datetime string")
    requires_road_closure: bool = Field(False, description="Whether road closure is needed")

    model_config = {"json_schema_extra": {
        "examples": [{
            "event_type": "unplanned",
            "event_cause": "accident",
            "latitude": 12.9716,
            "longitude": 77.5946,
            "corridor": "Mysore Road",
            "priority": "high",
            "veh_type": "heavy_vehicle",
            "start_datetime": "2024-03-15T17:30:00+05:30",
            "requires_road_closure": True,
        }]
    }}


class ScenarioInput(BaseModel):
    """Input for scenario planning (what-if)."""
    # Baseline state
    duration_min: float = Field(..., description="Predicted duration (minutes)")
    closure_prob: float = Field(..., description="Predicted closure probability (0–1)")
    priority_is_high: bool = Field(True)
    location_risk: float = Field(0.5, description="Location risk score (0–1)")
    current_personnel: int = Field(0)
    current_barricades: int = Field(0)
    # Scenario
    delta_personnel: int = Field(0, description="Officers to add/remove")
    delta_barricades: int = Field(0, description="Barricades to add/remove")
    close_road: bool = Field(False, description="Proactively close road?")


class ResourceEvent(BaseModel):
    """Single event for multi-event resource optimization."""
    event_id: str = Field(...)
    eii_score: float = Field(...)
    eii_level: str = Field(...)
    closure_prob: float = Field(0.5)
    duration_hours: float = Field(1.0)
    response_priority: int = Field(3, ge=1, le=4)


class ResourceRequest(BaseModel):
    """Multi-event resource optimization request."""
    events: list[ResourceEvent]
    total_personnel: int = Field(50, ge=1)
    total_barricades: int = Field(80, ge=1)


# ── Helper ────────────────────────────────────────────────────────────

def _get_state(request: Request):
    """Retrieve AppState from the app."""
    state = getattr(request.app.state, "ecip", None)
    if state is None or not state.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    return state


# ── Endpoints ─────────────────────────────────────────────────────────
import math
import numpy as np

def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        obj = float(obj)

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None

    return obj

@router.post("/events/predict", summary="Full event prediction pipeline")
async def predict_event(event: EventInput, request: Request):
    """
    Complete decision bundle for a single event.

    Flow: Similar Events → ML Predictions → EII → Priority → Scenarios

    Returns predictions, EII score/level, response priority (P1–P4),
    top-5 similar historical events, and pre-computed what-if scenarios.
    """
    state = _get_state(request)
    try:
        result = state.predict_event(event.model_dump())
        result = sanitize_json(result)
        return result
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events/explain", summary="SHAP explanations for an event")
async def explain_event(event: EventInput, request: Request):
    """
    Generate per-model SHAP explanations showing which features
    drove the duration and closure predictions.
    """
    state = _get_state(request)
    try:
        result = state.get_shap_explanation(event.model_dump())
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Explanation failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events/similar", summary="Find similar historical events")
async def find_similar(event: EventInput, request: Request, k: int = 5):
    """
    Retrieve top-K similar historical events with aggregate statistics
    (average duration, closure rate, manpower used).
    """
    state = _get_state(request)
    try:
        result = state.similar_retriever.find_similar(event.model_dump(), k=k)
        return result
    except Exception as e:
        logger.exception("Similar event search failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events/scenario", summary="Scenario planning (what-if)")
async def run_scenario(scenario: ScenarioInput, request: Request):
    """
    Simulate a resource-change scenario and project the EII impact.

    Example: "What if we deploy 2 more officers?"
    """
    state = _get_state(request)
    try:
        baseline = {
            "duration_min": scenario.duration_min,
            "closure_prob": scenario.closure_prob,
            "priority_is_high": scenario.priority_is_high,
            "location_risk": scenario.location_risk,
            "current_personnel": scenario.current_personnel,
            "current_barricades": scenario.current_barricades,
        }
        change = {
            "delta_personnel": scenario.delta_personnel,
            "delta_barricades": scenario.delta_barricades,
            "close_road": scenario.close_road,
        }
        result = state.scenario_planner.simulate(baseline, change)
        return result
    except Exception as e:
        logger.exception("Scenario simulation failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/resources", summary="Multi-event resource allocation")
async def optimize_resources(req: ResourceRequest, request: Request):
    """
    Optimally allocate personnel and barricades across multiple
    simultaneous events using Integer Linear Programming.
    """
    state = _get_state(request)
    try:
        optimizer = state.optimizer
        optimizer.P_max = req.total_personnel
        optimizer.B_max = req.total_barricades

        events = [e.model_dump() for e in req.events]
        result = optimizer.optimize(events)
        return result
    except Exception as e:
        logger.exception("Resource optimization failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", summary="Dataset statistics")
async def get_stats(request: Request):
    """Return high-level statistics about the loaded dataset."""
    state = _get_state(request)
    df = state.df

    has_dur = df["duration_min"].notna()
    return {
        "total_events": len(df),
        "events_with_duration": int(has_dur.sum()),
        "event_type_distribution": df["event_type"].value_counts().to_dict(),
        "event_cause_distribution": df["event_cause"].value_counts().head(10).to_dict(),
        "corridor_distribution": df["corridor"].value_counts().head(10).to_dict(),
        "priority_distribution": df["priority"].value_counts().to_dict(),
        "closure_rate": round(float(df["requires_road_closure"].mean()), 3),
        "avg_duration_min": round(float(df.loc[has_dur, "duration_min"].mean()), 1),
        "median_duration_min": round(float(df.loc[has_dur, "duration_min"].median()), 1),
    }


@router.get("/events/list", summary="List recent events from dataset")
async def list_events(
    request: Request,
    limit: int = 20,
    offset: int = 0,
    event_cause: Optional[str] = None,
    corridor: Optional[str] = None,
):
    """Return paginated events from the dataset, optionally filtered."""
    state = _get_state(request)
    df = state.df.copy()

    if event_cause:
        df = df[df["event_cause"] == event_cause.lower()]
    if corridor:
        df = df[df["corridor"] == corridor]

    total = len(df)
    df = df.sort_values("start_datetime", ascending=False).iloc[offset:offset + limit]

    events = []
    for _, row in df.iterrows():
        events.append({
            "id": row.get("id", ""),
            "event_type": row.get("event_type", ""),
            "event_cause": row.get("event_cause", ""),
            "corridor": row.get("corridor", ""),
            "priority": row.get("priority", ""),
            "latitude": float(row["latitude"]) if row.get("latitude") else None,
            "longitude": float(row["longitude"]) if row.get("longitude") else None,
            "start_datetime": str(row.get("start_datetime", "")),
            "duration_min": round(float(row["duration_min"]), 1) if row.get("duration_min") and row["duration_min"] == row["duration_min"] else None,
            "requires_road_closure": bool(row.get("requires_road_closure", False)),
        })

    return {"total": total, "limit": limit, "offset": offset, "events": events}
