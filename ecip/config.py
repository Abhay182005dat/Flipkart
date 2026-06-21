"""ECIP Central Configuration."""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / "ecip" / "saved_models"
DATASET_FILENAME = "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME

# Ensure model directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature Engineering ────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    "event_type",
    "event_cause",
    "corridor",
    "priority",
    "veh_type",
    "geohash",
    "time_of_day_bucket",
    "day_of_week_name",
]

NUMERICAL_FEATURES = [
    "latitude",
    "longitude",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
    "corridor_avg_duration",
    "corridor_closure_rate",
    "corridor_event_density",
    "cause_avg_duration",
    "cause_closure_rate",
]

# Features actually used by models (combined)
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# ── EII Configuration ─────────────────────────────────────────────────
EII_WEIGHTS = {
    "duration": 0.40,
    "closure": 0.35,
    "priority": 0.10,
    "location": 0.15,
}

EII_MAX_DURATION_MIN = 480  # 8 hours normalisation cap

EII_LEVELS = [
    (25, "Low", "green"),
    (50, "Medium", "yellow"),
    (75, "High", "orange"),
    (100, "Critical", "red"),
]

# ── Response Priority ─────────────────────────────────────────────────
PRIORITY_TIERS = {
    1: {"label": "DEPLOY IMMEDIATELY", "color": "red", "icon": "", "max_response_min": 0},
    2: {"label": "DEPLOY WITHIN 15 MIN", "color": "orange", "icon": "", "max_response_min": 15},
    3: {"label": "MONITOR", "color": "yellow", "icon": "", "max_response_min": 60},
    4: {"label": "OBSERVE ONLY", "color": "green", "icon": "", "max_response_min": None},
}

# ── Resource Allocation ────────────────────────────────────────────────
EII_RESOURCE_MAP = {
    "Low":      {"p_min": 1, "p_max": 3,  "b_min": 0, "b_max": 2},
    "Medium":   {"p_min": 2, "p_max": 5,  "b_min": 1, "b_max": 4},
    "High":     {"p_min": 3, "p_max": 8,  "b_min": 2, "b_max": 8},
    "Critical": {"p_min": 5, "p_max": 15, "b_min": 4, "b_max": 20},
}

# ── Similar Event Retrieval ────────────────────────────────────────────
SIMILAR_EVENT_K = 5
SIMILAR_EVENT_FEATURE_WEIGHTS = {
    "event_cause": 0.25,
    "corridor": 0.20,
    "time_of_day_bucket": 0.15,
    "priority": 0.15,
    "veh_type": 0.10,
    "latitude": 0.075,
    "longitude": 0.075,
}

# ── Scenario Planning ─────────────────────────────────────────────────
DEFAULT_ELASTICITIES = {
    "personnel_duration_elasticity": -0.08,
    "personnel_closure_elasticity": -0.05,
    "barricade_closure_elasticity": -0.03,
    "barricade_duration_elasticity": -0.02,
}

# ── Indian Holidays (approximate – Bengaluru-relevant) ─────────────────
INDIAN_HOLIDAYS_2023_2024 = {
    # 2023
    "2023-01-26", "2023-03-08", "2023-03-22", "2023-03-30",
    "2023-04-07", "2023-04-14", "2023-04-22", "2023-05-01",
    "2023-06-29", "2023-08-15", "2023-08-29", "2023-09-07",
    "2023-09-19", "2023-09-28", "2023-10-02", "2023-10-24",
    "2023-11-01", "2023-11-13", "2023-11-14", "2023-11-27",
    "2023-12-25",
    # 2024
    "2024-01-26", "2024-03-08", "2024-03-25", "2024-03-29",
    "2024-04-11", "2024-04-14", "2024-04-17", "2024-04-21",
    "2024-05-01", "2024-05-23", "2024-06-17", "2024-07-17",
    "2024-08-15", "2024-08-26", "2024-09-07", "2024-09-16",
    "2024-10-02", "2024-10-12", "2024-10-31", "2024-11-01",
    "2024-11-15", "2024-12-25",
}
