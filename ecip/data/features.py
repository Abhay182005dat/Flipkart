"""
ECIP Feature Engineering
========================
Transforms cleaned event data into ML-ready feature vectors.

Feature groups
--------------
A. **Temporal** — cyclical hour/month encoding, day-of-week, weekend, holiday
B. **Geospatial** — geohash (precision 6), lat/lon (numerical)
C. **Historical aggregates** — corridor-level and cause-level averages
   computed from the training set only (to prevent data leakage)
D. **Event-specific** — event_type, event_cause, priority, veh_type, closure flag

Design notes
------------
- Junction fill rate is 31 %, so we use **corridor** as the primary
  location grouping (99.8 % fill).  Geohash provides finer-grained
  spatial signal without cold-start issues.
- Per user refinement #5: similar-event retrieval combines corridor
  identity with geohash to mitigate cold-start on new junctions.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ecip.config import (
    CATEGORICAL_FEATURES,
    INDIAN_HOLIDAYS_2023_2024,
    NUMERICAL_FEATURES,
)

logger = logging.getLogger(__name__)


# ── Geohash (lightweight, no external dependency) ──────────────────────
_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def _encode_geohash(lat: float, lon: float, precision: int = 6) -> str:
    """Encode lat/lon to a geohash string of given precision."""
    lat_range = (-90.0, 90.0)
    lon_range = (-180.0, 180.0)
    bits = [16, 8, 4, 2, 1]
    geohash: list[str] = []
    ch = 0
    bit = 0
    even = True

    while len(geohash) < precision:
        if even:
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon > mid:
                ch |= bits[bit]
                lon_range = (mid, lon_range[1])
            else:
                lon_range = (lon_range[0], mid)
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat > mid:
                ch |= bits[bit]
                lat_range = (mid, lat_range[1])
            else:
                lat_range = (lat_range[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash.append(_BASE32[ch])
            bit = 0
            ch = 0

    return "".join(geohash)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time encodings and calendar flags."""
    df = df.copy()

    # Ensure start_datetime is a datetime type
    if not pd.api.types.is_datetime64_any_dtype(df["start_datetime"]):
        df["start_datetime"] = pd.to_datetime(
            df["start_datetime"], format="mixed", utc=True, errors="coerce"
        )

    hour = df["start_datetime"].dt.hour
    month = df["start_datetime"].dt.month
    dow = df["start_datetime"].dt.dayofweek  # 0=Mon

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Calendar
    df["day_of_week_name"] = df["start_datetime"].dt.day_name().str.lower()
    df["is_weekend"] = (dow >= 5).astype(int)

    # Time-of-day bucket
    df["time_of_day_bucket"] = pd.cut(
        hour,
        bins=[-1, 6, 12, 17, 21, 24],
        labels=["night", "morning", "afternoon", "evening", "late_night"],
    ).astype(str)

    # Holidays
    date_str = df["start_datetime"].dt.strftime("%Y-%m-%d")
    df["is_holiday"] = date_str.isin(INDIAN_HOLIDAYS_2023_2024).astype(int)

    logger.info("Added temporal features")
    return df


def add_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add geohash encoding from lat/lon."""
    df = df.copy()

    # Geohash (precision 6 → ~1.2 km × 0.6 km tiles)
    has_geo = df["latitude"].notna() & df["longitude"].notna()
    df["geohash"] = "unknown"
    df.loc[has_geo, "geohash"] = df.loc[has_geo].apply(
        lambda row: _encode_geohash(row["latitude"], row["longitude"], precision=6),
        axis=1,
    )

    logger.info(
        "Added geospatial features (%d/%d have valid geohash)",
        has_geo.sum(),
        len(df),
    )
    return df


def add_historical_aggregates(
    df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add corridor-level and cause-level historical aggregates.

    Parameters
    ----------
    df : DataFrame to add features to.
    reference_df : DataFrame to compute aggregates FROM.
        If None, uses ``df`` itself (appropriate for training set).
        During inference, pass the training set here to prevent leakage.
    """
    df = df.copy()
    ref = reference_df if reference_df is not None else df

    # ── Corridor-level aggregates ──────────────────────────────────────
    has_dur = ref["duration_min"].notna()

    corridor_stats = (
        ref.loc[has_dur]
        .groupby("corridor")
        .agg(
            corridor_avg_duration=("duration_min", "mean"),
            corridor_closure_rate=("requires_road_closure", "mean"),
            corridor_event_density=("id", "count"),
        )
        .reset_index()
    )
    # Normalise density to 0-1
    max_density = corridor_stats["corridor_event_density"].max()
    if max_density > 0:
        corridor_stats["corridor_event_density"] /= max_density

    df = df.merge(corridor_stats, on="corridor", how="left")

    # ── Cause-level aggregates ─────────────────────────────────────────
    cause_stats = (
        ref.loc[has_dur]
        .groupby("event_cause")
        .agg(
            cause_avg_duration=("duration_min", "mean"),
            cause_closure_rate=("requires_road_closure", "mean"),
        )
        .reset_index()
    )
    df = df.merge(cause_stats, on="event_cause", how="left")

    # Fill NaN with global means
    global_dur = ref.loc[has_dur, "duration_min"].mean() if has_dur.any() else 60.0
    global_closure = ref["requires_road_closure"].mean()

    for col in ("corridor_avg_duration", "cause_avg_duration"):
        df[col] = df[col].fillna(global_dur)
    for col in ("corridor_closure_rate", "cause_closure_rate"):
        df[col] = df[col].fillna(global_closure)
    df["corridor_event_density"] = df["corridor_event_density"].fillna(0.0)

    logger.info("Added historical aggregates (corridor & cause level)")
    return df


def compute_location_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a normalised location risk score per event.
    Combines corridor event density and closure rate.
    Used as an input to EII.
    """
    df = df.copy()
    df["location_risk"] = (
        0.5 * df["corridor_closure_rate"] +
        0.5 * df["corridor_event_density"]
    ).clip(0, 1)
    return df


def build_features(
    df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df : DataFrame (already cleaned via data.loader.clean).
    reference_df : Training set for computing historical aggregates
        (prevents leakage during inference).  None = use df itself.

    Returns
    -------
    DataFrame with all features added.
    """
    df = add_temporal_features(df)
    df = add_geospatial_features(df)
    df = add_historical_aggregates(df, reference_df=reference_df)
    df = compute_location_risk(df)

    # Ensure categoricals are strings (CatBoost needs this)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("unknown")

    # Ensure numericals are float
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    logger.info(
        "Feature engineering complete: %d rows × %d total cols",
        len(df), len(df.columns),
    )
    return df


def get_model_features(df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """Extract the feature matrix for model training/inference."""
    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return df[feature_list].copy()
