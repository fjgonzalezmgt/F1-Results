"""Public F1 API ingestion for model input data.

This module intentionally keeps network calls outside the simulator. It pulls
structured public data from Jolpica-F1/Ergast-compatible endpoints and OpenF1,
then translates it into the editable model inputs used by the app.
"""

from __future__ import annotations

import json
import math
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from f1predictor.config import (
    CALENDAR_REQUIRED_COLUMNS,
    CACHE_DIR,
    DRIVER_RATING_COLUMNS,
    JOLPICA_BASE_URL,
    OPENF1_BASE_URL,
)
from f1predictor.data import clean_calendar, clean_drivers


REQUEST_TIMEOUT_SECONDS = 20
HTTP_HEADERS = {"User-Agent": "f1-championship-lab/1.0"}


@dataclass
class PublicApiResult:
    """Normalized output from the public API refresh flow."""

    drivers: pd.DataFrame
    calendar: pd.DataFrame
    summary: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class PublicApiError(RuntimeError):
    """Raised when an upstream public API response cannot be used."""


def _normalize_text(value: Any) -> str:
    """Normalise text to lowercase ASCII alphanumeric for fuzzy matching.

    Applies NFKD decomposition, removes combining characters (accents),
    lowercases and strips all non-alphanumeric characters.

    Parameters
    ----------
    value : Any
        Input value; converted to string before processing.

    Returns
    -------
    str
        Normalised string containing only lowercase ASCII letters and digits.
    """
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return "".join(char.lower() for char in text if char.isalnum())


def _bounded(value: float, low: float = 1.0, high: float = 100.0) -> float:
    """Clip a float to a bounded range, replacing NaN with the midpoint.

    Parameters
    ----------
    value : float
        Value to bound.
    low : float, optional
        Lower bound, inclusive. Default is 1.0.
    high : float, optional
        Upper bound, inclusive. Default is 100.0.

    Returns
    -------
    float
        ``value`` clipped to [``low``, ``high``], or the midpoint if NaN.
    """
    if math.isnan(value):
        return float((low + high) / 2)
    return float(max(low, min(high, value)))


def _score_from_position(position: float, field_size: int) -> float:
    """Convert a finishing or qualifying position to a 0-100 score.

    Position 1 maps to 100; position ``field_size`` maps to approximately 45.

    Parameters
    ----------
    position : float
        Finishing or qualifying position (1-indexed).
    field_size : int
        Total number of classified drivers used to scale the range.

    Returns
    -------
    float
        Score in [1, 100].
    """
    if not position or math.isnan(position):
        return 55.0
    denominator = max(1, field_size - 1)
    return _bounded(100.0 - 55.0 * (position - 1.0) / denominator)


def _request_json(
    base_url: str,
    path: str,
    params: dict[str, Any] | None = None,
    cache_name: str | None = None,
    ttl_seconds: int = 900,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Make an HTTP GET request and return the parsed JSON payload.

    Optionally caches responses to ``CACHE_DIR`` and serves from cache
    when the file is younger than ``ttl_seconds``.

    Parameters
    ----------
    base_url : str
        Root URL of the API (e.g. ``"https://api.jolpi.ca"``).
    path : str
        URL path appended to ``base_url``; a leading ``/`` is added if absent.
    params : dict[str, Any] or None, optional
        Query-string parameters to encode in the URL.
    cache_name : str or None, optional
        Filename for the on-disk JSON cache; caching is skipped if ``None``.
    ttl_seconds : int, optional
        Maximum age (seconds) of a valid cached response. Default is 900.

    Returns
    -------
    dict[str, Any] or list[dict[str, Any]]
        Decoded JSON payload.

    Raises
    ------
    PublicApiError
        On HTTP errors, connection failures or timeouts.
    """
    if not path.startswith("/"):
        path = f"/{path}"
    query = urlencode(params or {}, doseq=True)
    url = f"{base_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{query}"

    cache_path: Path | None = None
    if cache_name:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / cache_name
        if cache_path.exists() and time.time() - cache_path.stat().st_mtime <= ttl_seconds:
            return json.loads(cache_path.read_text(encoding="utf-8"))

    request = Request(url, headers=HTTP_HEADERS)
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise PublicApiError(f"HTTP {exc.code} al consultar {url}") from exc
    except URLError as exc:
        raise PublicApiError(f"No se pudo conectar a {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise PublicApiError(f"Timeout consultando {url}") from exc

    if cache_path:
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _jolpica_json(
    path: str,
    params: dict[str, Any] | None = None,
    cache_name: str | None = None,
    ttl_seconds: int = 900,
) -> dict[str, Any]:
    """Fetch a Jolpica-F1 endpoint and validate the MRData envelope.

    Parameters
    ----------
    path : str
        API path relative to ``JOLPICA_BASE_URL``.
    params : dict[str, Any] or None, optional
        Additional query parameters.
    cache_name : str or None, optional
        Filename for the on-disk cache.
    ttl_seconds : int, optional
        Cache TTL in seconds. Default is 900.

    Returns
    -------
    dict[str, Any]
        Full response dictionary containing the ``MRData`` key.

    Raises
    ------
    PublicApiError
        If the response is not a dict or does not contain ``MRData``.
    """
    payload = _request_json(JOLPICA_BASE_URL, path, params=params, cache_name=cache_name, ttl_seconds=ttl_seconds)
    if not isinstance(payload, dict) or "MRData" not in payload:
        raise PublicApiError(f"Respuesta Jolpica inesperada para {path}")
    return payload


def _openf1_json(
    path: str,
    params: dict[str, Any] | None = None,
    cache_name: str | None = None,
    ttl_seconds: int = 900,
) -> list[dict[str, Any]]:
    """Fetch an OpenF1 endpoint and validate the list response.

    Parameters
    ----------
    path : str
        API path relative to ``OPENF1_BASE_URL``.
    params : dict[str, Any] or None, optional
        Additional query parameters.
    cache_name : str or None, optional
        Filename for the on-disk cache.
    ttl_seconds : int, optional
        Cache TTL in seconds. Default is 900.

    Returns
    -------
    list[dict[str, Any]]
        List of record dicts from the OpenF1 response.

    Raises
    ------
    PublicApiError
        If the response is not a list.
    """
    payload = _request_json(OPENF1_BASE_URL, path, params=params, cache_name=cache_name, ttl_seconds=ttl_seconds)
    if isinstance(payload, list):
        return payload
    raise PublicApiError(f"Respuesta OpenF1 inesperada para {path}")


def _race_table(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the ``Races`` list from a Jolpica MRData payload.

    Parameters
    ----------
    payload : dict[str, Any]
        Full Jolpica JSON response containing a ``RaceTable`` key.

    Returns
    -------
    list[dict[str, Any]]
        List of race objects, or an empty list if missing.
    """
    return payload.get("MRData", {}).get("RaceTable", {}).get("Races", []) or []


def _standings_lists(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the ``StandingsLists`` from a Jolpica MRData payload.

    Parameters
    ----------
    payload : dict[str, Any]
        Full Jolpica JSON response containing a ``StandingsTable`` key.

    Returns
    -------
    list[dict[str, Any]]
        List of standings-list objects, or an empty list if missing.
    """
    return payload.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", []) or []


def fetch_jolpica_schedule(season: int) -> pd.DataFrame:
    """Fetch a full season race schedule from Jolpica-F1.

    Falls back to the legacy ``/{season}.json`` endpoint if the primary
    ``/races.json`` endpoint fails.

    Parameters
    ----------
    season : int
        F1 season year (e.g. ``2026``).

    Returns
    -------
    pd.DataFrame
        Schedule DataFrame with columns ``round``, ``grand_prix``,
        ``country``, ``circuit``, ``race_date`` and ``has_sprint``,
        sorted by ``round``.

    Raises
    ------
    PublicApiError
        If both primary and fallback endpoints fail.
    """
    try:
        payload = _jolpica_json(
            f"/ergast/f1/{season}/races.json",
            params={"limit": 100},
            cache_name=f"jolpica_schedule_{season}.json",
        )
    except PublicApiError:
        payload = _jolpica_json(
            f"/ergast/f1/{season}.json",
            params={"limit": 100},
            cache_name=f"jolpica_schedule_legacy_{season}.json",
        )

    rows: list[dict[str, Any]] = []
    for race in _race_table(payload):
        circuit = race.get("Circuit", {}) or {}
        location = circuit.get("Location", {}) or {}
        rows.append(
            {
                "round": int(race.get("round", 0)),
                "grand_prix": race.get("raceName", ""),
                "country": location.get("country", ""),
                "circuit": circuit.get("circuitName", ""),
                "race_date": race.get("date", ""),
                "has_sprint": int("Sprint" in race or "SprintQualifying" in race),
            }
        )
    return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)


def fetch_jolpica_driver_standings(season: int) -> pd.DataFrame:
    """Fetch current driver standings from Jolpica-F1.

    Parameters
    ----------
    season : int
        F1 season year.

    Returns
    -------
    pd.DataFrame
        Driver standings with columns ``driver``, ``code``, ``team``,
        ``nationality``, ``current_points``, ``standing_position`` and
        ``wins``.  Returns an empty DataFrame if no standings are found.
    """
    payload = _jolpica_json(
        f"/ergast/f1/{season}/driverstandings.json",
        params={"limit": 100},
        cache_name=f"jolpica_driver_standings_{season}.json",
    )
    lists = _standings_lists(payload)
    if not lists:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for item in lists[0].get("DriverStandings", []) or []:
        driver = item.get("Driver", {}) or {}
        constructors = item.get("Constructors", []) or []
        constructor = constructors[0] if constructors else {}
        given = driver.get("givenName", "")
        family = driver.get("familyName", "")
        rows.append(
            {
                "driver": f"{given} {family}".strip(),
                "code": driver.get("code") or _fallback_code(family),
                "team": constructor.get("name", ""),
                "nationality": driver.get("nationality", ""),
                "current_points": float(item.get("points", 0) or 0),
                "standing_position": int(item.get("position", 0) or 0),
                "wins": int(item.get("wins", 0) or 0),
            }
        )
    return pd.DataFrame(rows)


def fetch_jolpica_constructor_standings(season: int) -> pd.DataFrame:
    """Fetch current constructor standings from Jolpica-F1.

    Parameters
    ----------
    season : int
        F1 season year.

    Returns
    -------
    pd.DataFrame
        Constructor standings with columns ``team``, ``constructor_points``,
        ``constructor_position`` and ``constructor_wins``.  Returns an empty
        DataFrame if no standings are found.
    """
    payload = _jolpica_json(
        f"/ergast/f1/{season}/constructorstandings.json",
        params={"limit": 100},
        cache_name=f"jolpica_constructor_standings_{season}.json",
    )
    lists = _standings_lists(payload)
    if not lists:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for item in lists[0].get("ConstructorStandings", []) or []:
        constructor = item.get("Constructor", {}) or {}
        rows.append(
            {
                "team": constructor.get("name", ""),
                "constructor_points": float(item.get("points", 0) or 0),
                "constructor_position": int(item.get("position", 0) or 0),
                "constructor_wins": int(item.get("wins", 0) or 0),
            }
        )
    return pd.DataFrame(rows)


def fetch_jolpica_round_results(season: int, round_no: int) -> pd.DataFrame:
    """Fetch classified race results for a single round.

    Parameters
    ----------
    season : int
        F1 season year.
    round_no : int
        Round number within the season (1-indexed).

    Returns
    -------
    pd.DataFrame
        Results DataFrame with columns ``round``, ``driver``, ``code``,
        ``team``, ``grid``, ``position``, ``points``, ``status`` and ``dnf``.
        Returns an empty DataFrame if no results are published.
    """
    payload = _jolpica_json(
        f"/ergast/f1/{season}/{round_no}/results.json",
        params={"limit": 100},
        cache_name=f"jolpica_results_{season}_{round_no}.json",
    )
    rows: list[dict[str, Any]] = []
    for race in _race_table(payload):
        for result in race.get("Results", []) or []:
            driver = result.get("Driver", {}) or {}
            constructor = result.get("Constructor", {}) or {}
            family = driver.get("familyName", "")
            status = str(result.get("status", ""))
            rows.append(
                {
                    "round": int(race.get("round", round_no)),
                    "driver": f"{driver.get('givenName', '')} {family}".strip(),
                    "code": driver.get("code") or _fallback_code(family),
                    "team": constructor.get("name", ""),
                    "grid": int(result.get("grid", 0) or 0),
                    "position": int(result.get("positionOrder", result.get("position", 0)) or 0),
                    "points": float(result.get("points", 0) or 0),
                    "status": status,
                    "dnf": int(_is_dnf_status(status)),
                }
            )
    return pd.DataFrame(rows)


def fetch_jolpica_round_qualifying(season: int, round_no: int) -> pd.DataFrame:
    """Fetch qualifying results for a single round.

    Parameters
    ----------
    season : int
        F1 season year.
    round_no : int
        Round number within the season (1-indexed).

    Returns
    -------
    pd.DataFrame
        Qualifying DataFrame with columns ``round``, ``driver``, ``code``,
        ``team`` and ``qualifying_position``.
        Returns an empty DataFrame if not yet published.
    """
    payload = _jolpica_json(
        f"/ergast/f1/{season}/{round_no}/qualifying.json",
        params={"limit": 100},
        cache_name=f"jolpica_qualifying_{season}_{round_no}.json",
    )
    rows: list[dict[str, Any]] = []
    for race in _race_table(payload):
        for result in race.get("QualifyingResults", []) or []:
            driver = result.get("Driver", {}) or {}
            constructor = result.get("Constructor", {}) or {}
            family = driver.get("familyName", "")
            rows.append(
                {
                    "round": int(race.get("round", round_no)),
                    "driver": f"{driver.get('givenName', '')} {family}".strip(),
                    "code": driver.get("code") or _fallback_code(family),
                    "team": constructor.get("name", ""),
                    "qualifying_position": int(result.get("position", 0) or 0),
                }
            )
    return pd.DataFrame(rows)


def fetch_jolpica_round_sprint_results(season: int, round_no: int) -> pd.DataFrame:
    """Fetch sprint results for a single round when a sprint was held.

    Parameters
    ----------
    season : int
        F1 season year.
    round_no : int
        Round number; the endpoint returns HTTP 404 when no sprint exists.

    Returns
    -------
    pd.DataFrame
        Sprint-results DataFrame with columns ``round``, ``driver``,
        ``code``, ``position`` and ``points``.
        Returns an empty DataFrame when no sprint results are found.
    """
    payload = _jolpica_json(
        f"/ergast/f1/{season}/{round_no}/sprint.json",
        params={"limit": 100},
        cache_name=f"jolpica_sprint_{season}_{round_no}.json",
    )
    rows: list[dict[str, Any]] = []
    for race in _race_table(payload):
        for result in race.get("SprintResults", []) or []:
            driver = result.get("Driver", {}) or {}
            family = driver.get("familyName", "")
            rows.append(
                {
                    "round": int(race.get("round", round_no)),
                    "driver": f"{driver.get('givenName', '')} {family}".strip(),
                    "code": driver.get("code") or _fallback_code(family),
                    "position": int(result.get("positionOrder", result.get("position", 0)) or 0),
                    "points": float(result.get("points", 0) or 0),
                }
            )
    return pd.DataFrame(rows)


def fetch_completed_jolpica_results(season: int, schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Fetch completed race and qualifying result tables round by round.

    Iterates over every round in ``schedule``, silently skipping rounds
    whose results are not yet published.

    Parameters
    ----------
    season : int
        F1 season year.
    schedule : pd.DataFrame
        Schedule DataFrame from ``fetch_jolpica_schedule`` containing a
        ``round`` column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, list[str]]
        * **results** – Concatenated race-results DataFrame.
        * **qualifying** – Concatenated qualifying-results DataFrame.
        * **errors** – List of error messages for failed rounds.
    """
    result_frames: list[pd.DataFrame] = []
    qualifying_frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for round_no in schedule["round"].astype(int).tolist():
        try:
            round_results = fetch_jolpica_round_results(season, round_no)
        except PublicApiError as exc:
            errors.append(str(exc))
            continue
        if round_results.empty:
            continue

        result_frames.append(round_results)
        try:
            qualifying = fetch_jolpica_round_qualifying(season, round_no)
            if not qualifying.empty:
                qualifying_frames.append(qualifying)
        except PublicApiError as exc:
            errors.append(str(exc))

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    qualifying = pd.concat(qualifying_frames, ignore_index=True) if qualifying_frames else pd.DataFrame()
    return results, qualifying, errors


def fetch_completed_jolpica_sprints(season: int, schedule: pd.DataFrame) -> tuple[set[int], list[str]]:
    """Return the set of sprint rounds that already have published results.

    Parameters
    ----------
    season : int
        F1 season year.
    schedule : pd.DataFrame
        Schedule DataFrame; must have a ``has_sprint`` column.

    Returns
    -------
    tuple[set[int], list[str]]
        * **completed** – Set of round numbers with published sprint results.
        * **errors** – List of non-404 error messages encountered.
    """
    completed: set[int] = set()
    errors: list[str] = []
    if schedule.empty or "has_sprint" not in schedule.columns:
        return completed, errors

    sprint_rounds = schedule.loc[
        pd.to_numeric(schedule["has_sprint"], errors="coerce").fillna(0).astype(int).gt(0),
        "round",
    ].astype(int)
    for round_no in sprint_rounds.tolist():
        try:
            sprint = fetch_jolpica_round_sprint_results(season, round_no)
        except PublicApiError as exc:
            message = str(exc)
            if "HTTP 404" not in message:
                errors.append(message)
            continue
        if not sprint.empty:
            completed.add(round_no)
    return completed, errors


def _fallback_code(family_name: str) -> str:
    """Generate a three-letter driver code from a family name.

    Parameters
    ----------
    family_name : str
        Driver's family (last) name.

    Returns
    -------
    str
        Uppercase three-letter code derived by normalising and truncating
        the family name.
    """
    return _normalize_text(family_name).upper()[:3]


def _is_dnf_status(status: str) -> bool:
    """Return True when a Jolpica status string indicates a retirement.

    Parameters
    ----------
    status : str
        Status string from a Jolpica race result (e.g. ``"+1 Lap"``,
        ``"Engine"``, ``"Accident"``).

    Returns
    -------
    bool
        ``True`` if the status indicates a DNF; ``False`` for classified
        finishes and empty strings.
    """
    lower = status.lower()
    if not lower:
        return False
    if "finished" in lower or "lap" in lower:
        return False
    return True


def _make_driver_lookup(base: pd.DataFrame) -> dict[str, str]:
    """Build a normalised-text to driver-code lookup dictionary.

    Creates mappings for the full driver name, the three-letter code and
    the family name, all normalised via ``_normalize_text``.

    Parameters
    ----------
    base : pd.DataFrame
        Cleaned driver table with ``code`` and ``driver`` columns.

    Returns
    -------
    dict[str, str]
        Mapping from normalised text keys to uppercase driver codes.
    """
    lookup: dict[str, str] = {}
    for _, row in base.iterrows():
        code = str(row["code"]).upper()
        lookup[_normalize_text(row["driver"])] = code
        lookup[_normalize_text(row["code"])] = code
        family = str(row["driver"]).split()[-1]
        lookup[_normalize_text(family)] = code
    return lookup


def _make_team_lookup(base: pd.DataFrame) -> dict[str, str]:
    """Build a normalised-text to canonical team-name lookup dictionary.

    Includes common abbreviations and alternative spellings as aliases
    (e.g. ``"redbull"`` → ``"Red Bull Racing"``, ``"kicksauber"`` → ``"Audi"``).

    Parameters
    ----------
    base : pd.DataFrame
        Cleaned driver table with a ``team`` column.

    Returns
    -------
    dict[str, str]
        Mapping from normalised text keys to canonical team name strings.
    """
    lookup: dict[str, str] = {}
    for team in base["team"].dropna().unique():
        lookup[_normalize_text(team)] = str(team)
    aliases = {
        "redbull": "Red Bull Racing",
        "rb": "Racing Bulls",
        "visacashapprb": "Racing Bulls",
        "sauber": "Audi",
        "kicksauber": "Audi",
        "astonmartin": "Aston Martin",
        "haas": "Haas F1 Team",
    }
    for alias, team in aliases.items():
        if team in set(base["team"]):
            lookup[alias] = team
    return lookup


def _map_api_driver_codes(api_df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Remap driver codes in an API DataFrame to match the seed table.

    Parameters
    ----------
    api_df : pd.DataFrame
        DataFrame from an external API containing ``code`` and optionally
        ``driver`` columns.
    base : pd.DataFrame
        Cleaned driver seed table used to build the lookup.

    Returns
    -------
    pd.DataFrame
        Copy of ``api_df`` with the ``code`` column remapped to seed codes.
    """
    if api_df.empty:
        return api_df.copy()
    lookup = _make_driver_lookup(base)
    mapped = api_df.copy()
    mapped["api_code"] = mapped.get("code", "").astype(str).str.upper().str.strip()
    driver_values = (
        mapped["driver"]
        if "driver" in mapped.columns
        else pd.Series([""] * len(mapped), index=mapped.index)
    )
    mapped["code"] = [
        lookup.get(_normalize_text(code), lookup.get(_normalize_text(driver), code))
        for code, driver in zip(mapped["api_code"], driver_values)
    ]
    return mapped


def _map_api_teams(api_df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Remap team names in an API DataFrame to match the seed table.

    Parameters
    ----------
    api_df : pd.DataFrame
        DataFrame from an external API containing a ``team`` column.
    base : pd.DataFrame
        Cleaned driver seed table used to build the lookup.

    Returns
    -------
    pd.DataFrame
        Copy of ``api_df`` with the ``team`` column remapped to canonical names.
    """
    if api_df.empty or "team" not in api_df.columns:
        return api_df.copy()
    lookup = _make_team_lookup(base)
    mapped = api_df.copy()
    mapped["team"] = [lookup.get(_normalize_text(team), str(team)) for team in mapped["team"]]
    return mapped


def derive_driver_inputs_from_public_data(
    base_drivers: pd.DataFrame,
    standings: pd.DataFrame,
    constructor_standings: pd.DataFrame,
    results: pd.DataFrame,
    qualifying: pd.DataFrame,
    lookback: int = 5,
) -> pd.DataFrame:
    """Translate standings and results into the model's driver rating columns.

    Blends API-derived metrics (race pace, consistency, qualifying, form,
    reliability, racecraft and team pace) with existing seed values using a
    fixed API weight of 0.38, so manual edits are partially preserved.

    Parameters
    ----------
    base_drivers : pd.DataFrame
        Current cleaned driver seed table.
    standings : pd.DataFrame
        Driver standings from ``fetch_jolpica_driver_standings``.
    constructor_standings : pd.DataFrame
        Constructor standings from ``fetch_jolpica_constructor_standings``.
    results : pd.DataFrame
        Concatenated race results from ``fetch_completed_jolpica_results``.
    qualifying : pd.DataFrame
        Concatenated qualifying results from ``fetch_completed_jolpica_results``.
    lookback : int, optional
        Number of most recent rounds used when computing rolling metrics.
        Default is 5.

    Returns
    -------
    pd.DataFrame
        Cleaned driver table with API-updated rating columns and points.
    """
    base = clean_drivers(base_drivers)
    output = base.copy()
    output["current_points"] = output["current_points"].astype(float)
    for column in DRIVER_RATING_COLUMNS:
        output[column] = output[column].astype(float)

    standings = _map_api_teams(_map_api_driver_codes(standings, base), base)
    results = _map_api_teams(_map_api_driver_codes(results, base), base)
    qualifying = _map_api_teams(_map_api_driver_codes(qualifying, base), base)
    constructor_standings = _map_api_teams(constructor_standings, base)

    if not standings.empty:
        standings_by_code = standings.drop_duplicates("code").set_index("code")
        for idx, row in output.iterrows():
            code = row["code"]
            if code not in standings_by_code.index:
                continue
            api_row = standings_by_code.loc[code]
            output.loc[idx, "current_points"] = api_row.get("current_points", row["current_points"])
            if api_row.get("team"):
                output.loc[idx, "team"] = api_row.get("team")
            if api_row.get("nationality"):
                output.loc[idx, "nationality"] = api_row.get("nationality")

    if results.empty:
        return clean_drivers(output)

    field_size = max(10, int(results.groupby("round")["code"].nunique().max()))
    recent_rounds = sorted(results["round"].dropna().astype(int).unique())[-lookback:]
    recent = results.loc[results["round"].isin(recent_rounds)].copy()
    qualifying_recent = qualifying.loc[qualifying["round"].isin(recent_rounds)].copy() if not qualifying.empty else pd.DataFrame()

    grouped = recent.groupby("code")
    race_metrics = grouped.agg(
        avg_position=("position", "mean"),
        position_std=("position", "std"),
        avg_grid=("grid", "mean"),
        avg_points=("points", "mean"),
        dnf_rate=("dnf", "mean"),
        starts=("round", "count"),
    )
    race_metrics["position_std"] = race_metrics["position_std"].fillna(0)
    race_metrics["avg_gain"] = race_metrics["avg_grid"] - race_metrics["avg_position"]

    if not qualifying_recent.empty:
        qualifying_metrics = qualifying_recent.groupby("code").agg(avg_qualifying=("qualifying_position", "mean"))
    else:
        qualifying_metrics = pd.DataFrame()

    team_points = recent.groupby("team")["points"].mean().rename("team_points_per_entry")
    if not constructor_standings.empty:
        constructor_count = max(1, len(constructor_standings))
        constructor_standings = constructor_standings.copy()
        constructor_standings["constructor_score"] = constructor_standings["constructor_position"].map(
            lambda pos: _score_from_position(float(pos), constructor_count)
        )
        constructor_scores = constructor_standings.set_index("team")["constructor_score"]
    else:
        constructor_scores = pd.Series(dtype=float)

    for idx, row in output.iterrows():
        code = row["code"]
        if code not in race_metrics.index:
            continue
        metrics = race_metrics.loc[code]
        race_position_score = _score_from_position(float(metrics["avg_position"]), field_size)
        points_score = _bounded(45.0 + 3.1 * float(metrics["avg_points"]))
        race_pace_api = 0.68 * race_position_score + 0.32 * points_score
        consistency_api = _bounded(96.0 - 5.0 * float(metrics["position_std"]) - 25.0 * float(metrics["dnf_rate"]))
        reliability_api = _bounded(96.0 - 38.0 * float(metrics["dnf_rate"]))
        racecraft_api = _bounded(72.0 + 3.5 * float(metrics["avg_gain"]))
        recent_form_api = _bounded(0.56 * race_pace_api + 0.44 * points_score)

        if code in qualifying_metrics.index:
            qualifying_api = _score_from_position(float(qualifying_metrics.loc[code, "avg_qualifying"]), field_size)
        elif metrics["avg_grid"] > 0:
            qualifying_api = _score_from_position(float(metrics["avg_grid"]), field_size)
        else:
            qualifying_api = row["qualifying"]

        team = row["team"]
        team_result_score = _bounded(45.0 + 3.0 * float(team_points.get(team, 0.0)))
        team_standing_score = float(constructor_scores.get(team, team_result_score))
        team_pace_api = _bounded(0.58 * team_result_score + 0.42 * team_standing_score)

        output.loc[idx, "race_pace"] = _blend(row["race_pace"], race_pace_api)
        output.loc[idx, "driver_rating"] = _blend(row["driver_rating"], 0.55 * race_pace_api + 0.45 * qualifying_api)
        output.loc[idx, "qualifying"] = _blend(row["qualifying"], qualifying_api)
        output.loc[idx, "consistency"] = _blend(row["consistency"], consistency_api)
        output.loc[idx, "reliability"] = _blend(row["reliability"], reliability_api)
        output.loc[idx, "racecraft"] = _blend(row["racecraft"], racecraft_api)
        output.loc[idx, "recent_form"] = _blend(row["recent_form"], recent_form_api)
        output.loc[idx, "team_pace"] = _blend(row["team_pace"], team_pace_api)
        output.loc[idx, "chassis"] = _blend(row["chassis"], team_pace_api, api_weight=0.22)
        output.loc[idx, "power_unit"] = _blend(row["power_unit"], team_pace_api, api_weight=0.18)

    return clean_drivers(output)


def _blend(seed_value: Any, api_value: Any, api_weight: float = 0.38) -> float:
    """Blend a seed value with an API-derived value using a weighted average.

    Parameters
    ----------
    seed_value : Any
        Existing seed rating; must be castable to float.
    api_value : Any
        API-derived rating; must be castable to float.
    api_weight : float, optional
        Weight assigned to the API value in [0, 1]. Default is 0.38.

    Returns
    -------
    float
        Blended value clipped to [1, 100].
    """
    seed = float(seed_value)
    api = float(api_value)
    return _bounded((1.0 - api_weight) * seed + api_weight * api)


def build_calendar_from_public_data(
    base_calendar: pd.DataFrame,
    schedule: pd.DataFrame,
    completed_rounds: set[int],
    completed_sprint_rounds: set[int] | None = None,
) -> pd.DataFrame:
    """Merge public schedule metadata into the editable calendar seed.

    For each round in ``schedule``, merges Jolpica metadata on top of
    inferred track features, then overlays any existing seed values so
    that manual circuit characteristics are preserved.

    Parameters
    ----------
    base_calendar : pd.DataFrame
        Current cleaned calendar seed table.
    schedule : pd.DataFrame
        Schedule DataFrame from ``fetch_jolpica_schedule``.
    completed_rounds : set[int]
        Round numbers confirmed to have published race results.
    completed_sprint_rounds : set[int] or None, optional
        Round numbers confirmed to have published sprint results.

    Returns
    -------
    pd.DataFrame
        Cleaned calendar DataFrame built from the public schedule with
        updated completion flags.
    """
    base = clean_calendar(base_calendar)
    if schedule.empty:
        return base
    completed_sprint_rounds = completed_sprint_rounds or set()

    by_round = base.set_index("round").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for _, race in schedule.iterrows():
        round_no = int(race["round"])
        seed = dict(by_round.get(round_no, {}))
        inferred = _infer_track_features(race)
        merged = {**inferred, **seed}
        merged.update(
            {
                "round": round_no,
                "grand_prix": race.get("grand_prix") or seed.get("grand_prix", ""),
                "country": race.get("country") or seed.get("country", ""),
                "circuit": race.get("circuit") or seed.get("circuit", ""),
                "race_date": race.get("race_date") or seed.get("race_date", ""),
                "sprint_remaining": int(
                    bool(race.get("has_sprint", 0))
                    and round_no not in completed_rounds
                    and round_no not in completed_sprint_rounds
                ),
                "completed": int(round_no in completed_rounds),
            }
        )
        rows.append({column: merged.get(column) for column in CALENDAR_REQUIRED_COLUMNS})
    return clean_calendar(pd.DataFrame(rows))


def _infer_track_features(race: pd.Series) -> dict[str, Any]:
    """Infer circuit feature defaults from the race name and circuit name.

    Uses keyword matching against a hardcoded feature map for known circuits
    (Monaco, Monza, Singapore, Spa, Silverstone, Suzuka, Zandvoort, Baku,
    Las Vegas) and a street-hybrid group for other urban venues.

    Parameters
    ----------
    race : pd.Series
        Schedule row with ``grand_prix`` and ``circuit`` fields.

    Returns
    -------
    dict[str, Any]
        Feature dict with keys ``track_type``, ``downforce``, ``power``,
        ``tyre_stress``, ``overtake_difficulty``, ``weather_risk``,
        ``safety_car_risk`` and ``qualifying_importance``.
    """
    text = _normalize_text(f"{race.get('grand_prix', '')} {race.get('circuit', '')}")
    defaults = {
        "track_type": "permanent",
        "downforce": 68,
        "power": 70,
        "tyre_stress": 62,
        "overtake_difficulty": 55,
        "weather_risk": 30,
        "safety_car_risk": 45,
        "qualifying_importance": 65,
    }
    feature_map = {
        "monaco": {"track_type": "street", "downforce": 95, "power": 25, "overtake_difficulty": 95, "safety_car_risk": 75, "qualifying_importance": 98},
        "monza": {"downforce": 35, "power": 98, "tyre_stress": 45, "overtake_difficulty": 35, "qualifying_importance": 55},
        "singapore": {"track_type": "street", "downforce": 92, "power": 40, "tyre_stress": 85, "overtake_difficulty": 82, "weather_risk": 70, "safety_car_risk": 85, "qualifying_importance": 88},
        "spa": {"downforce": 70, "power": 90, "weather_risk": 65, "overtake_difficulty": 35},
        "silverstone": {"downforce": 80, "power": 75, "tyre_stress": 70, "weather_risk": 55},
        "suzuka": {"downforce": 86, "power": 76, "tyre_stress": 72, "qualifying_importance": 70},
        "zandvoort": {"downforce": 85, "power": 55, "overtake_difficulty": 75, "weather_risk": 65, "qualifying_importance": 85},
        "baku": {"track_type": "street", "downforce": 45, "power": 88, "safety_car_risk": 75, "qualifying_importance": 70},
        "lasvegas": {"track_type": "street", "downforce": 35, "power": 95, "tyre_stress": 45, "overtake_difficulty": 40},
    }
    for key, values in feature_map.items():
        if key in text:
            return {**defaults, **values}
    if any(key in text for key in ["miami", "montreal", "jeddah", "madrid"]):
        return {**defaults, "track_type": "street-hybrid", "safety_car_risk": 55, "qualifying_importance": 68}
    return defaults


def fetch_openf1_sessions_for_race(season: int, race: pd.Series) -> pd.DataFrame:
    """Find OpenF1 race sessions matching a given calendar row.

    Fetches all race sessions for the season and scores them by how many
    country/circuit/grand-prix tokens appear in the OpenF1 location fields.

    Parameters
    ----------
    season : int
        F1 season year.
    race : pd.Series
        Calendar row with ``grand_prix``, ``country`` and ``circuit`` fields.

    Returns
    -------
    pd.DataFrame
        Candidate OpenF1 session rows sorted by descending match score,
        or an empty DataFrame if no matches are found.
    """
    rows = _openf1_json(
        "/sessions",
        params={"year": season, "session_name": "Race"},
        cache_name=f"openf1_sessions_race_{season}.json",
        ttl_seconds=1800,
    )
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    target = _normalize_text(f"{race.get('grand_prix', '')} {race.get('country', '')} {race.get('circuit', '')}")

    def score(row: pd.Series) -> int:
        haystack = _normalize_text(
            f"{row.get('country_name', '')} {row.get('location', '')} {row.get('circuit_short_name', '')}"
        )
        tokens = [token for token in [race.get("country"), race.get("circuit"), race.get("grand_prix")] if token]
        return sum(1 for token in tokens if _normalize_text(token) in haystack or haystack in target)

    frame["match_score"] = frame.apply(score, axis=1)
    return frame.loc[frame["match_score"] > 0].sort_values("match_score", ascending=False).reset_index(drop=True)


def _try_openf1_rows(
    path: str,
    params: dict[str, Any],
    cache_name: str,
    ttl_seconds: int = 900,
) -> tuple[list[dict[str, Any]], str | None]:
    """Call an OpenF1 endpoint and return rows with a silenced 404 error.

    Parameters
    ----------
    path : str
        API path relative to ``OPENF1_BASE_URL``.
    params : dict[str, Any]
        Query parameters for the request.
    cache_name : str
        Filename for the on-disk cache.
    ttl_seconds : int, optional
        Cache TTL in seconds. Default is 900.

    Returns
    -------
    tuple[list[dict[str, Any]], str or None]
        * **rows** – Parsed response rows, or an empty list on failure.
        * **error** – Error message string, or ``None`` for 404s and success.
    """
    try:
        return _openf1_json(path, params=params, cache_name=cache_name, ttl_seconds=ttl_seconds), None
    except PublicApiError as exc:
        message = str(exc)
        if "HTTP 404" in message:
            return [], None
        return [], message


def apply_openf1_weekend_inputs(calendar: pd.DataFrame, season: int, target_rounds: list[int]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Use OpenF1 weather and race-control data to adjust selected calendar rows.

    For each round in ``target_rounds``, locates the corresponding OpenF1
    session, fetches weather, race-control and session-result data, and
    updates ``weather_risk``, ``safety_car_risk`` and ``completed`` columns.

    Parameters
    ----------
    calendar : pd.DataFrame
        Cleaned calendar table to update.
    season : int
        F1 season year used to find matching OpenF1 sessions.
    target_rounds : list[int]
        Round numbers to process; non-existent rounds are skipped.

    Returns
    -------
    tuple[pd.DataFrame, list[str], list[str]]
        * **updated** – Cleaned calendar with adjusted risk columns.
        * **summary** – Human-readable update messages per processed round.
        * **errors** – Error messages for any failed API calls.
    """
    updated = clean_calendar(calendar).copy()
    for column in ["weather_risk", "safety_car_risk"]:
        updated[column] = updated[column].astype(float)
    summary: list[str] = []
    errors: list[str] = []

    for round_no in target_rounds:
        if round_no not in set(updated["round"]):
            continue
        idx = updated.index[updated["round"] == round_no][0]
        race = updated.loc[idx]
        try:
            sessions = fetch_openf1_sessions_for_race(season, race)
        except PublicApiError as exc:
            errors.append(str(exc))
            continue
        if sessions.empty:
            continue

        session = sessions.iloc[0]
        session_key = int(session["session_key"])
        meeting_key = int(session["meeting_key"])
        weather, weather_error = _try_openf1_rows(
            "/weather",
            params={"session_key": session_key},
            cache_name=f"openf1_weather_{session_key}.json",
            ttl_seconds=900,
        )
        race_control, race_control_error = _try_openf1_rows(
            "/race_control",
            params={"session_key": session_key},
            cache_name=f"openf1_race_control_{session_key}.json",
            ttl_seconds=900,
        )
        session_result, session_result_error = _try_openf1_rows(
            "/session_result",
            params={"session_key": session_key},
            cache_name=f"openf1_session_result_{session_key}.json",
            ttl_seconds=900,
        )
        for error in [weather_error, race_control_error, session_result_error]:
            if error:
                errors.append(error)

        if weather:
            weather_df = pd.DataFrame(weather)
            rainfall_share = pd.to_numeric(weather_df.get("rainfall", 0), errors="coerce").fillna(0).gt(0).mean()
            humidity = pd.to_numeric(weather_df.get("humidity", 0), errors="coerce").fillna(0).mean()
            wind = pd.to_numeric(weather_df.get("wind_speed", 0), errors="coerce").fillna(0).mean()
            weather_risk = _bounded(20.0 + 75.0 * rainfall_share + max(0.0, humidity - 70.0) * 0.6 + wind * 2.0, 0, 100)
            updated.loc[idx, "weather_risk"] = round(weather_risk, 1)

        if race_control:
            messages = " ".join(str(item.get("message", "")) for item in race_control).lower()
            incident_count = sum(messages.count(word) for word in ["safety car", "vsc", "red flag", "yellow"])
            updated.loc[idx, "safety_car_risk"] = round(_bounded(25.0 + 8.0 * incident_count, 0, 100), 1)

        if session_result:
            updated.loc[idx, "completed"] = 1

        summary.append(f"OpenF1 round {round_no}: session_key={session_key}, meeting_key={meeting_key}")
    return clean_calendar(updated), summary, errors


def refresh_model_inputs_from_public_apis(
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    season: int,
    lookback: int = 5,
    use_openf1: bool = True,
) -> PublicApiResult:
    """Fetch public API data and return updated model input tables.

    Orchestrates the full public-API refresh pipeline: Jolpica schedule,
    driver standings, constructor standings, per-round race and qualifying
    results, sprint completion flags, calendar reconstruction and optional
    OpenF1 weather/race-control patches.

    Parameters
    ----------
    drivers : pd.DataFrame
        Current driver seed table.
    calendar : pd.DataFrame
        Current calendar seed table.
    season : int
        F1 season year to fetch data for.
    lookback : int, optional
        Number of recent rounds used when computing rolling driver metrics.
        Default is 5.
    use_openf1 : bool, optional
        Whether to enrich the calendar with OpenF1 weather and race-control
        data. Default is ``True``.

    Returns
    -------
    PublicApiResult
        Dataclass containing updated ``drivers``, ``calendar``, ``summary``
        messages, ``sources`` URLs and ``errors`` encountered.
    """
    summary: list[str] = []
    sources = [
        f"{JOLPICA_BASE_URL}/ergast/f1/{season}/driverstandings.json",
        f"{JOLPICA_BASE_URL}/ergast/f1/{season}/races.json",
    ]
    errors: list[str] = []

    base_drivers = clean_drivers(drivers)
    base_calendar = clean_calendar(calendar)

    schedule = fetch_jolpica_schedule(season)
    standings = fetch_jolpica_driver_standings(season)
    constructors = fetch_jolpica_constructor_standings(season)
    results, qualifying, result_errors = fetch_completed_jolpica_results(season, schedule)
    completed_sprint_rounds, sprint_errors = fetch_completed_jolpica_sprints(season, schedule)
    errors.extend(result_errors[:5])
    errors.extend(sprint_errors[:3])

    completed_rounds = set(results["round"].dropna().astype(int).unique()) if not results.empty else set()
    updated_calendar = build_calendar_from_public_data(
        base_calendar,
        schedule,
        completed_rounds,
        completed_sprint_rounds,
    )
    updated_drivers = derive_driver_inputs_from_public_data(
        base_drivers,
        standings,
        constructors,
        results,
        qualifying,
        lookback=lookback,
    )

    summary.append(f"Jolpica: {len(schedule)} carreras, {len(standings)} pilotos en standings.")
    summary.append(f"Jolpica: {len(completed_rounds)} rondas con resultados y {len(qualifying)} filas de qualy.")
    if completed_sprint_rounds:
        summary.append(f"Jolpica: sprints ya publicados en rondas {sorted(completed_sprint_rounds)}.")

    if use_openf1:
        open_rounds = updated_calendar.loc[updated_calendar["completed"] == 0, "round"].astype(int).tolist()
        target_rounds = open_rounds[:1]
        if completed_rounds:
            target_rounds.append(max(completed_rounds))
        target_rounds = sorted(set(target_rounds))
        try:
            updated_calendar, open_summary, open_errors = apply_openf1_weekend_inputs(
                updated_calendar,
                season,
                target_rounds,
            )
            summary.extend(open_summary)
            errors.extend(open_errors[:3])
            sources.append(f"{OPENF1_BASE_URL}/sessions?year={season}&session_name=Race")
        except PublicApiError as exc:
            errors.append(str(exc))

    return PublicApiResult(
        drivers=updated_drivers,
        calendar=updated_calendar,
        summary=summary,
        sources=sources,
        errors=errors,
    )
