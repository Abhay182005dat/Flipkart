"""
ECIP Data Loader
================
Loads the Astram traffic-event CSV, cleans it, and computes derived
columns (duration, resolved timestamps, etc.).

Designed for the real dataset:
  - 8,173 records, 46 columns
  - `end_datetime` is 94 % null → we derive from closed/resolved datetimes
  - Severity column does NOT exist; priority is binary High/Low
  - Junction fill rate is 31 %; corridor fill is 99.8 %
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ecip.config import DATASET_PATH

logger = logging.getLogger(__name__)


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw CSV and return an unmodified DataFrame."""
    path = Path(path) if path else DATASET_PATH
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows × %d cols", *df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the raw DataFrame.

    Steps
    -----
    1. Parse datetime columns (mixed ISO formats, UTC).
    2. Derive a single ``end_dt`` from whichever of
       ``closed_datetime`` / ``resolved_datetime`` is available.
    3. Compute ``duration_min``.
    4. Normalise categorical values (lowercase, strip, merge near-duplicates).
    5. Drop rows with negative or absurdly large durations.
    6. Fill sparse columns with sensible defaults.
    """
    df = df.copy()

    # ── 1. Datetime parsing ────────────────────────────────────────────
    for col in ("start_datetime", "end_datetime", "closed_datetime",
                "resolved_datetime", "created_date", "modified_datetime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed",
                                     utc=True, errors="coerce")

    # ── 2. Derive end timestamp ────────────────────────────────────────
    df["end_dt"] = df["closed_datetime"].fillna(df["resolved_datetime"])

    # ── 3. Duration ────────────────────────────────────────────────────
    mask_has_end = df["end_dt"].notna() & df["start_datetime"].notna()
    df["duration_min"] = np.nan
    df.loc[mask_has_end, "duration_min"] = (
        (df.loc[mask_has_end, "end_dt"] - df.loc[mask_has_end, "start_datetime"])
        .dt.total_seconds() / 60.0
    )

    # ── 4. Normalise categoricals ──────────────────────────────────────
    # event_cause: lowercase, merge 'Debris' → 'debris', 'Fog / Low Visibility' → 'fog'
    df["event_cause"] = (
        df["event_cause"]
        .str.strip()
        .str.lower()
        .replace({
            "fog / low visibility": "fog",
            "test_demo": "others",
        })
    )

    # priority: lowercase
    df["priority"] = df["priority"].str.strip().str.lower().fillna("low")

    # event_type: should already be clean
    df["event_type"] = df["event_type"].str.strip().str.lower()

    # veh_type: fill missing
    df["veh_type"] = df["veh_type"].fillna("unknown").str.strip().str.lower()

    # corridor: fill missing
    df["corridor"] = df["corridor"].fillna("non-corridor").str.strip()

    # junction: keep as-is (31 % fill)
    # zone: keep as-is

    # requires_road_closure: ensure boolean
    df["requires_road_closure"] = df["requires_road_closure"].astype(str).str.lower()
    df["requires_road_closure"] = df["requires_road_closure"].map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False).astype(bool)

    # ── 5. Filter bad durations ────────────────────────────────────────
    bad_duration = (
        df["duration_min"].notna()
        & ((df["duration_min"] <= 0) | (df["duration_min"] > 10_000))
    )
    n_bad = bad_duration.sum()
    if n_bad > 0:
        logger.info("Dropping %d rows with invalid duration (≤0 or >10000 min)", n_bad)
        df.loc[bad_duration, "duration_min"] = np.nan

    # ── 6. Latitude / longitude sanity ─────────────────────────────────
    # Bengaluru bounding box approx: 12.7–13.2 lat, 77.3–77.8 lon
    geo_valid = (
        df["latitude"].between(12.5, 13.5) & df["longitude"].between(77.0, 78.0)
    )
    df.loc[~geo_valid, ["latitude", "longitude"]] = np.nan

    logger.info(
        "Cleaned dataset: %d rows, %d with valid duration, %d with valid geo",
        len(df),
        df["duration_min"].notna().sum(),
        geo_valid.sum(),
    )
    return df


def load_and_clean(path: Path | str | None = None) -> pd.DataFrame:
    """Convenience: load + clean in one call."""
    return clean(load_raw(path))
