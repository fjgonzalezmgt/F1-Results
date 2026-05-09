"""Data loading, validation, cleaning and update helpers.

Provides functions to read driver and calendar seed CSVs, validate table
schemas, coerce column types, clip rating values and merge external updates
(from public APIs or LLM web-search) into the seed tables.
"""

from __future__ import annotations

from io import StringIO
from typing import Iterable

import pandas as pd

from f1predictor.config import (
    CALENDAR_PATH,
    CALENDAR_REQUIRED_COLUMNS,
    DRIVER_RATING_COLUMNS,
    DRIVER_REQUIRED_COLUMNS,
    DRIVERS_PATH,
)
from f1predictor.logging_utils import instrument_module_functions


def load_drivers(path=DRIVERS_PATH) -> pd.DataFrame:
    """Load driver seed data from a CSV file.

    Parameters
    ----------
    path : Path, optional
        Filesystem path to the driver CSV seed file.
        Defaults to ``DRIVERS_PATH`` defined in ``config``.

    Returns
    -------
    pd.DataFrame
        Raw driver table as read from disk; no cleaning applied.
    """
    return pd.read_csv(path)


def load_calendar(path=CALENDAR_PATH) -> pd.DataFrame:
    """Load race calendar seed data from a CSV file.

    Parameters
    ----------
    path : Path, optional
        Filesystem path to the calendar CSV seed file.
        Defaults to ``CALENDAR_PATH`` defined in ``config``.

    Returns
    -------
    pd.DataFrame
        Raw calendar table as read from disk; no cleaning applied.
    """
    return pd.read_csv(path)


def dataframe_to_csv_text(df: pd.DataFrame) -> str:
    """Serialize a DataFrame to a CSV string for Streamlit cache hashing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to serialize.

    Returns
    -------
    str
        CSV representation of ``df`` without the row index.
    """
    return df.to_csv(index=False)


def dataframe_from_csv_text(csv_text: str) -> pd.DataFrame:
    """Deserialize a DataFrame from a CSV string.

    Parameters
    ----------
    csv_text : str
        CSV-formatted string, typically produced by ``dataframe_to_csv_text``.

    Returns
    -------
    pd.DataFrame
        Reconstructed DataFrame.
    """
    return pd.read_csv(StringIO(csv_text))


def _require_columns(df: pd.DataFrame, columns: Iterable[str], kind: str) -> None:
    """Raise ValueError if required columns are missing from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    columns : Iterable[str]
        Column names that must be present in ``df``.
    kind : str
        Human-readable label used in the error message to identify the table.

    Raises
    ------
    ValueError
        If any of ``columns`` are absent from ``df.columns``.
    """
    missing = set(columns).difference(df.columns)
    if missing:
        raise ValueError(f"{kind}: faltan columnas: {', '.join(sorted(missing))}")


def validate_drivers(df: pd.DataFrame) -> None:
    """Validate the driver table used by the simulation.

    Parameters
    ----------
    df : pd.DataFrame
        Driver table to validate.

    Raises
    ------
    ValueError
        If required columns are missing, driver codes are duplicated,
        or the table contains fewer than 10 rows.
    """
    _require_columns(df, DRIVER_REQUIRED_COLUMNS, "drivers")
    if df["code"].duplicated().any():
        duplicated = df.loc[df["code"].duplicated(), "code"].tolist()
        raise ValueError(f"Codigos de piloto duplicados: {', '.join(duplicated)}")
    if len(df) < 10:
        raise ValueError("La tabla de pilotos parece incompleta.")


def validate_calendar(df: pd.DataFrame) -> None:
    """Validate the calendar table used by the simulation.

    Parameters
    ----------
    df : pd.DataFrame
        Calendar table to validate.

    Raises
    ------
    ValueError
        If required columns are missing or round numbers are duplicated.
    """
    _require_columns(df, CALENDAR_REQUIRED_COLUMNS, "calendar")
    if df["round"].duplicated().any():
        duplicated = df.loc[df["round"].duplicated(), "round"].astype(str).tolist()
        raise ValueError(f"Rondas duplicadas: {', '.join(duplicated)}")


def clean_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce driver table to stable types and bounded ratings.

    Validates the input, strips whitespace from text columns, converts
    ``current_points`` to numeric, and clips all rating columns to [1, 100].

    Parameters
    ----------
    df : pd.DataFrame
        Driver table to clean; must pass ``validate_drivers``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with standardised types and bounded values.

    Raises
    ------
    ValueError
        If ``df`` fails validation (see ``validate_drivers``).
    """
    validate_drivers(df)
    clean = df.copy()
    clean["driver"] = clean["driver"].astype(str).str.strip()
    clean["code"] = clean["code"].astype(str).str.upper().str.strip()
    clean["team"] = clean["team"].astype(str).str.strip()
    clean["nationality"] = clean["nationality"].astype(str).str.strip()
    clean["current_points"] = pd.to_numeric(clean["current_points"], errors="coerce").fillna(0)

    for column in DRIVER_RATING_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean[column] = clean[column].fillna(clean[column].median()).clip(1, 100)
    return clean


def clean_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce calendar table to stable types and bounded event features.

    Validates the input, converts numeric columns, fills missing values
    with column medians, clips feature columns to [0, 100] and sorts by
    round number.

    Parameters
    ----------
    df : pd.DataFrame
        Calendar table to clean; must pass ``validate_calendar``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with standardised types, sorted by ``round``.

    Raises
    ------
    ValueError
        If ``df`` fails validation (see ``validate_calendar``).
    """
    validate_calendar(df)
    clean = df.copy()
    clean["round"] = pd.to_numeric(clean["round"], errors="coerce").astype(int)
    clean["sprint_remaining"] = pd.to_numeric(clean["sprint_remaining"], errors="coerce").fillna(0).astype(int)
    clean["completed"] = pd.to_numeric(clean["completed"], errors="coerce").fillna(0).astype(int)
    for column in [
        "downforce",
        "power",
        "tyre_stress",
        "overtake_difficulty",
        "weather_risk",
        "safety_car_risk",
        "qualifying_importance",
    ]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean[column] = clean[column].fillna(clean[column].median()).clip(0, 100)
    return clean.sort_values("round").reset_index(drop=True)


def apply_driver_update(base_df: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    """Merge public or LLM-updated driver data into the driver seed table.

    Only rows whose driver code (or name) matches an existing entry in
    ``base_df`` are updated; unmatched rows in ``updates`` are ignored.
    Text columns are updated when the incoming value is non-empty; numeric
    columns replace the base value unconditionally when present.

    Parameters
    ----------
    base_df : pd.DataFrame
        Existing driver seed table (will not be mutated).
    updates : pd.DataFrame
        Partial driver table from an external source; must contain either
        a ``code`` or ``driver`` column for row matching.

    Returns
    -------
    pd.DataFrame
        Merged and cleaned copy of ``base_df`` incorporating ``updates``.
    """
    if updates.empty:
        return base_df.copy()

    key = "code" if "code" in updates.columns else "driver"
    if key not in base_df.columns:
        return base_df.copy()

    text_columns = ["driver", "team", "nationality"]
    numeric_columns = ["current_points", *DRIVER_RATING_COLUMNS]
    available_text = [c for c in text_columns if c in updates.columns and c in base_df.columns and c != key]
    available_numeric = [c for c in numeric_columns if c in updates.columns and c in base_df.columns]
    if not available_text and not available_numeric:
        return base_df.copy()

    merged = base_df.copy()
    updates_indexed = updates.copy()
    updates_indexed[key] = updates_indexed[key].astype(str).str.upper().str.strip()
    updates_indexed = updates_indexed.loc[updates_indexed[key] != ""]
    updates_indexed = updates_indexed.drop_duplicates(key, keep="last").set_index(key)

    base_key = merged[key].astype(str).str.upper().str.strip()
    mask = base_key.isin(updates_indexed.index)

    for column in available_text:
        mapped = base_key.map(updates_indexed[column])
        text = mapped.astype(str).str.strip()
        valid = mask & mapped.notna() & text.ne("") & text.str.lower().ne("nan")
        merged.loc[valid, column] = text.loc[valid]

    for column in available_numeric:
        mapping = pd.to_numeric(updates_indexed[column], errors="coerce")
        mapped = base_key.map(mapping)
        valid = mask & mapped.notna()
        merged.loc[valid, column] = mapped.loc[valid]
    return clean_drivers(merged)


def apply_calendar_update(base_df: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    """Merge public or LLM calendar updates into the calendar seed table.

    Rows are matched on ``round`` number.  Text columns are updated when the
    incoming value is non-empty; numeric columns replace the base value when
    present and parseable.

    Parameters
    ----------
    base_df : pd.DataFrame
        Existing calendar seed table (will not be mutated).
    updates : pd.DataFrame
        Partial calendar table from an external source; must contain a
        ``round`` column for row matching.

    Returns
    -------
    pd.DataFrame
        Merged and cleaned copy of ``base_df`` incorporating ``updates``.
    """
    if updates.empty or "round" not in updates.columns:
        return clean_calendar(base_df)

    updatable = [
        "grand_prix",
        "country",
        "circuit",
        "race_date",
        "sprint_remaining",
        "completed",
        "track_type",
        "downforce",
        "power",
        "tyre_stress",
        "overtake_difficulty",
        "weather_risk",
        "safety_car_risk",
        "qualifying_importance",
    ]
    available = [column for column in updatable if column in updates.columns and column in base_df.columns]
    if not available:
        return clean_calendar(base_df)

    merged = base_df.copy()
    merged["round"] = pd.to_numeric(merged["round"], errors="coerce").astype(int)
    updates_indexed = updates.copy()
    updates_indexed["round"] = pd.to_numeric(updates_indexed["round"], errors="coerce").astype(int)
    updates_indexed = updates_indexed.drop_duplicates("round").set_index("round")

    numeric_columns = {
        "sprint_remaining",
        "completed",
        "downforce",
        "power",
        "tyre_stress",
        "overtake_difficulty",
        "weather_risk",
        "safety_car_risk",
        "qualifying_importance",
    }
    for column in available:
        mapping = updates_indexed[column]
        mask = merged["round"].isin(mapping.index)
        if column in numeric_columns:
            mapped = pd.to_numeric(merged.loc[mask, "round"].map(mapping), errors="coerce")
            merged.loc[mask, column] = mapped.fillna(merged.loc[mask, column])
        else:
            mapped = merged.loc[mask, "round"].map(mapping)
            merged.loc[mask, column] = mapped.where(
                mapped.notna() & (mapped.astype(str) != ""),
                merged.loc[mask, column],
            )
    return clean_calendar(merged)


def apply_official_formula1_update(
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    payload: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a Formula1.com LLM web-search payload to driver and calendar seeds.

    Parameters
    ----------
    drivers : pd.DataFrame
        Current driver seed table.
    calendar : pd.DataFrame
        Current calendar seed table.
    payload : dict[str, pd.DataFrame]
        Dictionary with optional keys ``"drivers"`` and ``"calendar"``
        containing DataFrames returned by the LLM web-search call.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Cleaned ``(drivers, calendar)`` tuple with LLM updates applied.
    """
    updated_drivers = drivers.copy()
    if "drivers" in payload:
        updated_drivers = apply_driver_update(updated_drivers, payload["drivers"])

    updated_calendar = calendar.copy()
    if "calendar" in payload:
        updated_calendar = apply_calendar_update(updated_calendar, payload["calendar"])

    return clean_drivers(updated_drivers), clean_calendar(updated_calendar)


instrument_module_functions(__name__)
