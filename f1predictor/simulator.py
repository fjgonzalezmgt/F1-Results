"""Monte Carlo engine for F1 race and championship prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from f1predictor.config import RACE_POINTS, SPRINT_POINTS
from f1predictor.data import clean_calendar, clean_drivers
from f1predictor.parameters import SimParams


def _center(value: float) -> float:
    """Convert a 0-100 rating to an approximately centered score.

    Maps the midpoint (50) to 0 and scales the range to roughly [-5, 5].

    Parameters
    ----------
    value : float
        A rating in the [0, 100] range.

    Returns
    -------
    float
        Centred score: ``(value - 50) / 10``.
    """
    return (float(value) - 50.0) / 10.0


def _event_fit(row: pd.Series, race: pd.Series) -> float:
    """Estimate how well a driver/team package fits a circuit.

    Combines downforce, power, tyre, racecraft and strategy affinities
    against the corresponding circuit feature values.

    Parameters
    ----------
    row : pd.Series
        Single driver row from a cleaned drivers DataFrame.
    race : pd.Series
        Single calendar row describing the event's circuit features.

    Returns
    -------
    float
        Composite circuit-fit score (positive means an advantage).
    """
    downforce_fit = _center(row["chassis"]) * (race["downforce"] - 50.0) / 28.0
    power_fit = _center(row["power_unit"]) * (race["power"] - 50.0) / 28.0
    tyre_fit = _center(row["tyre_management"]) * (race["tyre_stress"] - 50.0) / 30.0
    street_fit = _center(row["racecraft"]) * (race["overtake_difficulty"] - 50.0) / 42.0
    strategy_fit = _center(row["strategy"]) * (race["safety_car_risk"] - 50.0) / 48.0
    return float(downforce_fit + power_fit + tyre_fit + street_fit + strategy_fit)


def _combine_strength(
    driver_component: float | np.ndarray,
    constructor_component: float | np.ndarray,
    recent_form: float | np.ndarray,
    params: SimParams,
    form_scale: float = 1.0,
) -> float | np.ndarray:
    """Blend driver, constructor and decayed-form components."""
    effective_form_weight = max(0.0, params.form_weight) * max(0.0, float(form_scale))
    total = max(0.01, params.driver_weight + params.constructor_weight + effective_form_weight)
    return (
        params.driver_weight * driver_component
        + params.constructor_weight * constructor_component
        + effective_form_weight * recent_form
    ) / total


def _form_scale_for_race(race_index: int, params: SimParams) -> float:
    """Return the recent-form multiplier for a future race index."""
    decay = max(0.05, float(params.form_decay_races))
    return float(np.exp(-max(0, race_index) / decay))


def _package_uncertainty_scale(calendar: pd.DataFrame, params: SimParams) -> float:
    """Scale team-package uncertainty down as completed evidence accumulates."""
    completed_before_start = calendar.loc[
        (calendar["completed"] == 1) & (calendar["round"] < params.start_round),
        "round",
    ].nunique()
    return float(params.team_uncertainty / np.sqrt(completed_before_start + 1.0))


def _season_evidence_confidence(calendar: pd.DataFrame, params: SimParams, prior_rounds: float = 8.0) -> float:
    """Return how much to trust current-season derived ratings."""
    completed_before_start = calendar.loc[
        (calendar["completed"] == 1) & (calendar["round"] < params.start_round),
        "round",
    ].nunique()
    return float(completed_before_start / (completed_before_start + prior_rounds))


def _regress_to_mean(
    values: float | np.ndarray,
    center: float,
    confidence: float,
    floor_factor: float,
) -> float | np.ndarray:
    """Shrink values toward a center when current-season evidence is thin."""
    confidence = max(0.0, min(1.0, float(confidence)))
    factor = max(0.0, min(1.0, float(floor_factor) + (1.0 - float(floor_factor)) * confidence))
    return center + (values - center) * factor


def _base_strength(row: pd.Series, params: SimParams, form_scale: float = 1.0) -> float:
    """Blend driver, constructor and form ratings into a single latent strength.

    Parameters
    ----------
    row : pd.Series
        Single driver row from a cleaned drivers DataFrame.
    params : SimParams
        Simulation parameters controlling the relative weights.

    Returns
    -------
    float
        Weighted composite strength score for the driver-team package.
    """
    driver = 0.38 * row["driver_rating"] + 0.34 * row["race_pace"] + 0.16 * row["racecraft"] + 0.12 * row["consistency"]
    constructor = 0.38 * row["team_pace"] + 0.22 * row["chassis"] + 0.18 * row["power_unit"] + 0.12 * row["strategy"] + 0.10 * row["reliability"]
    return float(_combine_strength(driver, constructor, row["recent_form"], params, form_scale=form_scale))


def _dnf_probability(row: pd.Series, race: pd.Series, params: SimParams, sprint: bool) -> float:
    """Estimate the retirement probability for a given event.

    Parameters
    ----------
    row : pd.Series
        Single driver row; uses the ``reliability`` rating.
    race : pd.Series
        Calendar row; uses ``safety_car_risk``, ``tyre_stress`` and ``power``.
    params : SimParams
        Simulation parameters; uses ``reliability_multiplier``.
    sprint : bool
        If ``True``, uses the lower sprint base DNF rate (0.012).

    Returns
    -------
    float
        DNF probability clipped to [0.002, 0.18].
    """
    base = 0.012 if sprint else 0.038
    reliability_gap = (100.0 - row["reliability"]) / 100.0
    track_stress = 0.65 + (race["safety_car_risk"] + race["tyre_stress"] + race["power"]) / 300.0
    prob = base * (1.0 + 2.5 * reliability_gap) * track_stress * params.reliability_multiplier
    return float(np.clip(prob, 0.002, 0.18))


def _simulate_qualifying(
    drivers: pd.DataFrame,
    race: pd.Series,
    rng: np.random.Generator,
    params: SimParams,
    team_trend: dict[str, float] | None = None,
    form_scale: float = 1.0,
) -> pd.DataFrame:
    """Simulate a qualifying order for a race or sprint event.

    Parameters
    ----------
    drivers : pd.DataFrame
        Cleaned driver table used as the entry list.
    race : pd.Series
        Calendar row containing circuit features and weather risk.
    rng : np.random.Generator
        Seeded random number generator for reproducibility.
    params : SimParams
        Simulation parameters; uses ``chaos`` and ``weather_multiplier``.
    team_trend : dict[str, float] or None, optional
        Per-team development-drift values to add to qualifying scores.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``code``, ``qualifying_score`` and
        ``grid_position`` sorted by descending score.
    """
    rows: list[dict[str, Any]] = []
    weather_factor = 1.0 if rng.random() < race["weather_risk"] / 100.0 * params.weather_multiplier else 0.0
    for _, row in drivers.iterrows():
        trend = 0.0 if team_trend is None else team_trend.get(row["team"], 0.0)
        score = (
            0.42 * row["qualifying"]
            + 0.22 * row["team_pace"]
            + 0.16 * row["chassis"]
            + 0.10 * row["power_unit"]
            + 0.10 * form_scale * row["recent_form"]
            + 1.6 * _event_fit(row, race)
            + trend
        )
        if weather_factor:
            score += (row["wet_skill"] - row["driver_rating"]) * 0.25
        noise = rng.normal(0, params.chaos * (0.78 + (100.0 - row["consistency"]) / 260.0))
        rows.append({"code": row["code"], "qualifying_score": score + noise})

    order = pd.DataFrame(rows).sort_values("qualifying_score", ascending=False).reset_index(drop=True)
    order["grid_position"] = np.arange(1, len(order) + 1)
    return order


def simulate_event(
    drivers: pd.DataFrame,
    race: pd.Series,
    rng: np.random.Generator,
    params: SimParams,
    sprint: bool = False,
    team_trend: dict[str, float] | None = None,
    form_scale: float = 1.0,
) -> pd.DataFrame:
    """Simulate one race or sprint and return the finishing order.

    Runs qualifying internally, applies wet-weather and safety-car effects,
    rolls DNFs and assigns points according to the F1 or sprint points table.

    Parameters
    ----------
    drivers : pd.DataFrame
        Cleaned driver table used as the entry list.
    race : pd.Series
        Calendar row containing circuit features and risk parameters.
    rng : np.random.Generator
        Seeded random number generator for reproducibility.
    params : SimParams
        Simulation parameters for weights, noise and multipliers.
    sprint : bool, optional
        If ``True``, uses sprint-specific noise scaling and points table.
        Default is ``False``.
    team_trend : dict[str, float] or None, optional
        Per-team development-drift values applied to scores.

    Returns
    -------
    pd.DataFrame
        Finishing-order DataFrame with columns ``driver``, ``code``,
        ``team``, ``grid_position``, ``score``, ``dnf``, ``position``,
        ``points``, ``wet_race`` and ``safety_car``.
    """
    clean = clean_drivers(drivers)
    qualifying = _simulate_qualifying(clean, race, rng, params, team_trend=team_trend, form_scale=form_scale)
    field = clean.merge(qualifying[["code", "grid_position"]], on="code", how="left")
    n = len(field)
    wet_race = rng.random() < race["weather_risk"] / 100.0 * params.weather_multiplier
    safety_car = rng.random() < race["safety_car_risk"] / 100.0 * params.safety_car_multiplier

    rows: list[dict[str, Any]] = []
    for _, row in field.iterrows():
        trend = 0.0 if team_trend is None else team_trend.get(row["team"], 0.0)
        grid_edge = (n + 1 - row["grid_position"]) - ((n + 1) / 2.0)
        quali_lock = params.qualifying_weight * race["qualifying_importance"] / 100.0
        overtake_release = (100.0 - race["overtake_difficulty"]) / 100.0
        grid_bonus = grid_edge * (0.28 + 0.72 * quali_lock) * (1.15 - 0.55 * overtake_release)
        score = _base_strength(row, params, form_scale=form_scale) + 2.2 * _event_fit(row, race) + grid_bonus + trend
        score += 0.08 * row["tyre_management"] * race["tyre_stress"] / 100.0
        if wet_race:
            score += (row["wet_skill"] - 75.0) * 0.18
        if safety_car:
            score += rng.normal(0, params.chaos * 0.85) + (row["strategy"] - 75.0) * 0.10

        noise_scale = params.chaos * (1.0 + (100.0 - row["consistency"]) / 180.0)
        if sprint:
            noise_scale *= 0.72
        dnf = rng.random() < _dnf_probability(row, race, params, sprint=sprint)
        if dnf:
            score = -999.0 + rng.normal(0, 4.0)
        else:
            score += rng.normal(0, noise_scale)

        rows.append(
            {
                "driver": row["driver"],
                "code": row["code"],
                "team": row["team"],
                "grid_position": int(row["grid_position"]),
                "score": float(score),
                "dnf": bool(dnf),
            }
        )

    result = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    result["position"] = np.arange(1, n + 1)
    points = SPRINT_POINTS if sprint else RACE_POINTS
    result["points"] = [points[pos - 1] if pos <= len(points) else 0 for pos in result["position"]]
    result["wet_race"] = wet_race
    result["safety_car"] = safety_car
    return result


def _empty_counts(drivers: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return a zeroed probability accumulator keyed by driver code.

    Parameters
    ----------
    drivers : pd.DataFrame
        Driver table used to collect the set of driver codes.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict mapping each driver code to a zero-initialised dict
        with keys ``champion``, ``top3``, ``top6``, ``avg_points`` and
        ``avg_final_rank``.
    """
    return {
        row["code"]: {
            "champion": 0.0,
            "top3": 0.0,
            "top6": 0.0,
            "avg_points": 0.0,
            "avg_final_rank": 0.0,
        }
        for _, row in drivers.iterrows()
    }


def simulate_many(
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    params: SimParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run many vectorised season simulations and aggregate probabilities.

    Iterates ``params.simulations`` full seasons using NumPy array operations
    for speed, accumulates championship probabilities and expected points,
    then returns summary DataFrames for drivers, constructors and per-race
    winner probabilities.

    Parameters
    ----------
    drivers : pd.DataFrame
        Cleaned driver seed table with current standings and ratings.
    calendar : pd.DataFrame
        Cleaned calendar table; only rounds with ``completed == 0`` and
        ``round >= params.start_round`` are simulated forward.
    params : SimParams
        Simulation parameters controlling iterations, weights and noise.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A three-element tuple:

        * **driver_df** – Driver championship summary with columns
          ``driver``, ``code``, ``team``, ``current_points``,
          ``expected_points``, ``avg_final_rank``, ``champion_pct``,
          ``top3_pct`` and ``top6_pct``, sorted by ``champion_pct`` desc.
        * **constructor_df** – Constructor championship summary with
          columns ``team``, ``current_points``, ``expected_points``,
          ``avg_final_rank``, ``champion_pct`` and ``top3_pct``.
        * **race_df** – Per-race winner probabilities with columns
          ``round``, ``grand_prix``, ``driver``, ``code``, ``team`` and
          ``win_pct``.
    """
    clean_dr = clean_drivers(drivers)
    clean_cal = clean_calendar(calendar)
    remaining = clean_cal.loc[(clean_cal["completed"] == 0) & (clean_cal["round"] >= params.start_round)].copy()
    rng = np.random.default_rng(params.seed)

    codes = clean_dr["code"].to_numpy()
    driver_names = clean_dr["driver"].to_numpy()
    driver_teams = clean_dr["team"].to_numpy()
    teams = np.array(sorted(clean_dr["team"].unique()))
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    team_idx = np.array([team_to_idx[team] for team in driver_teams], dtype=int)

    current_points = clean_dr["current_points"].to_numpy(dtype=float)
    driver_rating = clean_dr["driver_rating"].to_numpy(dtype=float)
    qualifying = clean_dr["qualifying"].to_numpy(dtype=float)
    race_pace = clean_dr["race_pace"].to_numpy(dtype=float)
    consistency = clean_dr["consistency"].to_numpy(dtype=float)
    tyre_management = clean_dr["tyre_management"].to_numpy(dtype=float)
    wet_skill = clean_dr["wet_skill"].to_numpy(dtype=float)
    racecraft = clean_dr["racecraft"].to_numpy(dtype=float)
    team_pace = clean_dr["team_pace"].to_numpy(dtype=float)
    chassis = clean_dr["chassis"].to_numpy(dtype=float)
    power_unit = clean_dr["power_unit"].to_numpy(dtype=float)
    strategy = clean_dr["strategy"].to_numpy(dtype=float)
    reliability = clean_dr["reliability"].to_numpy(dtype=float)
    recent_form = clean_dr["recent_form"].to_numpy(dtype=float)

    driver_component = 0.38 * driver_rating + 0.34 * race_pace + 0.16 * racecraft + 0.12 * consistency
    constructor_component = 0.38 * team_pace + 0.22 * chassis + 0.18 * power_unit + 0.12 * strategy + 0.10 * reliability
    evidence_confidence = _season_evidence_confidence(clean_cal, params)
    driver_component = _regress_to_mean(driver_component, float(np.mean(driver_component)), evidence_confidence, 0.82)
    constructor_component = _regress_to_mean(
        constructor_component,
        float(np.mean(constructor_component)),
        evidence_confidence,
        0.68,
    )
    recent_form = _regress_to_mean(recent_form, 75.0, evidence_confidence, 0.45)
    qualifying_base_no_form = (
        0.42 * qualifying
        + 0.22 * team_pace
        + 0.16 * chassis
        + 0.10 * power_unit
    )
    qualifying_base_no_form = _regress_to_mean(
        qualifying_base_no_form,
        float(np.mean(qualifying_base_no_form)),
        evidence_confidence,
        0.78,
    )
    package_uncertainty_scale = _package_uncertainty_scale(clean_cal, params)
    q_noise_scale = params.chaos * (0.78 + (100.0 - consistency) / 260.0)
    race_noise_scale = params.chaos * (1.0 + (100.0 - consistency) / 180.0)

    n_drivers = len(clean_dr)
    n_teams = len(teams)
    base_team_points = np.zeros(n_teams, dtype=float)
    np.add.at(base_team_points, team_idx, current_points)

    champion_counts = np.zeros(n_drivers, dtype=float)
    top3_counts = np.zeros(n_drivers, dtype=float)
    top6_counts = np.zeros(n_drivers, dtype=float)
    points_sum = np.zeros(n_drivers, dtype=float)
    rank_sum = np.zeros(n_drivers, dtype=float)

    constructor_champion_counts = np.zeros(n_teams, dtype=float)
    constructor_top3_counts = np.zeros(n_teams, dtype=float)
    constructor_points_sum = np.zeros(n_teams, dtype=float)
    constructor_rank_sum = np.zeros(n_teams, dtype=float)

    race_rows_meta = remaining[["round", "grand_prix"]].to_dict(orient="records")
    race_winner_counts = np.zeros((len(race_rows_meta), n_drivers), dtype=float)

    race_point_values = np.array(RACE_POINTS, dtype=float)
    sprint_point_values = np.array(SPRINT_POINTS, dtype=float)

    def event_fit_vector(race: pd.Series) -> np.ndarray:
        downforce_fit = ((chassis - 50.0) / 10.0) * (race["downforce"] - 50.0) / 28.0
        power_fit = ((power_unit - 50.0) / 10.0) * (race["power"] - 50.0) / 28.0
        tyre_fit = ((tyre_management - 50.0) / 10.0) * (race["tyre_stress"] - 50.0) / 30.0
        street_fit = ((racecraft - 50.0) / 10.0) * (race["overtake_difficulty"] - 50.0) / 42.0
        strategy_fit = ((strategy - 50.0) / 10.0) * (race["safety_car_risk"] - 50.0) / 48.0
        return downforce_fit + power_fit + tyre_fit + street_fit + strategy_fit

    def run_session(
        race: pd.Series,
        sprint: bool,
        team_adjustment_by_driver: np.ndarray,
        form_scale: float,
    ) -> tuple[np.ndarray, int]:
        fit = event_fit_vector(race)
        weather_qualifying = rng.random() < race["weather_risk"] / 100.0 * params.weather_multiplier
        qualifying_score = qualifying_base_no_form + 0.10 * form_scale * recent_form + 1.6 * fit + team_adjustment_by_driver
        if weather_qualifying:
            qualifying_score = qualifying_score + (wet_skill - driver_rating) * 0.25
        qualifying_score = qualifying_score + rng.normal(0.0, q_noise_scale, size=n_drivers)
        qualifying_order = np.argsort(-qualifying_score)
        grid_position = np.empty(n_drivers, dtype=float)
        grid_position[qualifying_order] = np.arange(1, n_drivers + 1)

        wet_race = rng.random() < race["weather_risk"] / 100.0 * params.weather_multiplier
        safety_car = rng.random() < race["safety_car_risk"] / 100.0 * params.safety_car_multiplier
        grid_edge = (n_drivers + 1 - grid_position) - ((n_drivers + 1) / 2.0)
        quali_lock = params.qualifying_weight * race["qualifying_importance"] / 100.0
        overtake_release = (100.0 - race["overtake_difficulty"]) / 100.0
        grid_bonus = grid_edge * (0.28 + 0.72 * quali_lock) * (1.15 - 0.55 * overtake_release)
        base_strength = _combine_strength(
            driver_component,
            constructor_component,
            recent_form,
            params,
            form_scale=form_scale,
        )
        score = base_strength + 2.2 * fit + grid_bonus + team_adjustment_by_driver
        score = score + 0.08 * tyre_management * race["tyre_stress"] / 100.0
        if wet_race:
            score = score + (wet_skill - 75.0) * 0.18
        if safety_car:
            score = score + rng.normal(0.0, params.chaos * 0.85, size=n_drivers) + (strategy - 75.0) * 0.10

        noise_scale = race_noise_scale * (0.72 if sprint else 1.0)
        score = score + rng.normal(0.0, noise_scale, size=n_drivers)
        base_dnf = 0.012 if sprint else 0.038
        reliability_gap = (100.0 - reliability) / 100.0
        track_stress = 0.65 + (race["safety_car_risk"] + race["tyre_stress"] + race["power"]) / 300.0
        dnf_prob = base_dnf * (1.0 + 2.5 * reliability_gap) * track_stress * params.reliability_multiplier
        dnf_prob = np.clip(dnf_prob, 0.002, 0.18)
        dnf = rng.random(n_drivers) < dnf_prob
        score = np.where(dnf, -999.0 + rng.normal(0.0, 4.0, size=n_drivers), score)

        finishing_order = np.argsort(-score)
        point_values = sprint_point_values if sprint else race_point_values
        session_points = np.zeros(n_drivers, dtype=float)
        paying_positions = min(len(point_values), n_drivers)
        session_points[finishing_order[:paying_positions]] = point_values[:paying_positions]
        return session_points, int(finishing_order[0])

    for _ in range(params.simulations):
        points = current_points.copy()
        team_points = base_team_points.copy()
        season_trend = rng.normal(0.0, params.development_drift, size=n_teams)
        package_uncertainty = rng.normal(0.0, package_uncertainty_scale, size=n_teams)

        for race_index, (_, race) in enumerate(remaining.iterrows()):
            progress = 0.0 if len(remaining) <= 1 else race_index / (len(remaining) - 1)
            team_adjustment_by_driver = package_uncertainty[team_idx] + season_trend[team_idx] * progress
            form_scale = _form_scale_for_race(race_index, params)

            if params.include_sprints and int(race["sprint_remaining"]) == 1:
                sprint_points, _winner_idx = run_session(
                    race,
                    sprint=True,
                    team_adjustment_by_driver=team_adjustment_by_driver,
                    form_scale=form_scale,
                )
                points += sprint_points
                np.add.at(team_points, team_idx, sprint_points)

            race_points, winner_idx = run_session(
                race,
                sprint=False,
                team_adjustment_by_driver=team_adjustment_by_driver,
                form_scale=form_scale,
            )
            points += race_points
            np.add.at(team_points, team_idx, race_points)
            race_winner_counts[race_index, winner_idx] += 1

        driver_order = np.argsort(-points)
        ranks = np.empty(n_drivers, dtype=float)
        ranks[driver_order] = np.arange(1, n_drivers + 1)
        points_sum += points
        rank_sum += ranks
        champion_counts[driver_order[0]] += 1
        top3_counts[driver_order[:3]] += 1
        top6_counts[driver_order[:6]] += 1

        constructor_order = np.argsort(-team_points)
        constructor_ranks = np.empty(n_teams, dtype=float)
        constructor_ranks[constructor_order] = np.arange(1, n_teams + 1)
        constructor_points_sum += team_points
        constructor_rank_sum += constructor_ranks
        constructor_champion_counts[constructor_order[0]] += 1
        constructor_top3_counts[constructor_order[:3]] += 1

    driver_rows = [
        {
            "driver": driver_names[idx],
            "code": codes[idx],
            "team": driver_teams[idx],
            "current_points": current_points[idx],
            "expected_points": points_sum[idx] / params.simulations,
            "avg_final_rank": rank_sum[idx] / params.simulations,
            "champion_pct": 100 * champion_counts[idx] / params.simulations,
            "top3_pct": 100 * top3_counts[idx] / params.simulations,
            "top6_pct": 100 * top6_counts[idx] / params.simulations,
        }
        for idx in range(n_drivers)
    ]

    constructor_rows = [
        {
            "team": teams[idx],
            "current_points": base_team_points[idx],
            "expected_points": constructor_points_sum[idx] / params.simulations,
            "avg_final_rank": constructor_rank_sum[idx] / params.simulations,
            "champion_pct": 100 * constructor_champion_counts[idx] / params.simulations,
            "top3_pct": 100 * constructor_top3_counts[idx] / params.simulations,
        }
        for idx in range(n_teams)
    ]

    race_rows = []
    for race_index, meta in enumerate(race_rows_meta):
        for driver_idx, count in enumerate(race_winner_counts[race_index]):
            if count <= 0:
                continue
            race_rows.append(
                {
                    "round": int(meta["round"]),
                    "grand_prix": meta["grand_prix"],
                    "driver": driver_names[driver_idx],
                    "code": codes[driver_idx],
                    "team": driver_teams[driver_idx],
                    "win_pct": 100 * count / params.simulations,
                }
            )

    driver_df = pd.DataFrame(driver_rows).sort_values(
        ["champion_pct", "expected_points", "current_points"],
        ascending=False,
    )
    constructor_df = pd.DataFrame(constructor_rows).sort_values(
        ["champion_pct", "expected_points"],
        ascending=False,
    )
    race_df = pd.DataFrame(race_rows).sort_values(["round", "win_pct"], ascending=[True, False])
    return driver_df, constructor_df, race_df


def build_driver_diagnostics(
    driver_results: pd.DataFrame,
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    params: SimParams,
    limit: int = 12,
) -> pd.DataFrame:
    """Build an explainability table for the main championship result.

    The table is intentionally approximate: it decomposes the deterministic
    inputs that shape the Monte Carlo instead of trying to replay every random
    draw.  It is useful for spotting overconfident seeds, dominant team ratings
    and circuits that flatter one package.
    """
    clean_dr = clean_drivers(drivers)
    clean_cal = clean_calendar(calendar)
    remaining = clean_cal.loc[(clean_cal["completed"] == 0) & (clean_cal["round"] >= params.start_round)].copy()
    package_sd = _package_uncertainty_scale(clean_cal, params)
    evidence_confidence = _season_evidence_confidence(clean_cal, params)

    raw_driver_components = (
        0.38 * clean_dr["driver_rating"]
        + 0.34 * clean_dr["race_pace"]
        + 0.16 * clean_dr["racecraft"]
        + 0.12 * clean_dr["consistency"]
    )
    raw_constructor_components = (
        0.38 * clean_dr["team_pace"]
        + 0.22 * clean_dr["chassis"]
        + 0.18 * clean_dr["power_unit"]
        + 0.12 * clean_dr["strategy"]
        + 0.10 * clean_dr["reliability"]
    )
    driver_center = float(raw_driver_components.mean())
    constructor_center = float(raw_constructor_components.mean())

    if remaining.empty:
        avg_form_scale = 0.0
    else:
        avg_form_scale = float(np.mean([_form_scale_for_race(idx, params) for idx in range(len(remaining))]))

    results_by_code = (
        driver_results.drop_duplicates("code").set_index("code")
        if driver_results is not None and not driver_results.empty and "code" in driver_results.columns
        else pd.DataFrame()
    )
    result_order = {code: idx for idx, code in enumerate(driver_results["code"].tolist())} if not driver_results.empty else {}

    rows: list[dict[str, Any]] = []
    for _, row in clean_dr.iterrows():
        raw_driver_component = 0.38 * row["driver_rating"] + 0.34 * row["race_pace"] + 0.16 * row["racecraft"] + 0.12 * row["consistency"]
        raw_constructor_component = (
            0.38 * row["team_pace"]
            + 0.22 * row["chassis"]
            + 0.18 * row["power_unit"]
            + 0.12 * row["strategy"]
            + 0.10 * row["reliability"]
        )
        driver_component = float(_regress_to_mean(raw_driver_component, driver_center, evidence_confidence, 0.82))
        constructor_component = float(_regress_to_mean(raw_constructor_component, constructor_center, evidence_confidence, 0.68))
        recent_form_effective = float(_regress_to_mean(row["recent_form"], 75.0, evidence_confidence, 0.45))
        circuit_fits = [2.2 * _event_fit(row, race) for _, race in remaining.iterrows()]
        avg_circuit_fit = float(np.mean(circuit_fits)) if circuit_fits else 0.0
        base_strength_now = float(_combine_strength(driver_component, constructor_component, recent_form_effective, params, form_scale=1.0))
        base_strength_avg = float(
            _combine_strength(driver_component, constructor_component, recent_form_effective, params, form_scale=avg_form_scale)
        )

        code = row["code"]
        result = results_by_code.loc[code] if not results_by_code.empty and code in results_by_code.index else None
        expected_points = float(result["expected_points"]) if result is not None else float(row["current_points"])
        champion_pct = float(result["champion_pct"]) if result is not None else 0.0
        avg_final_rank = float(result["avg_final_rank"]) if result is not None else 0.0

        rows.append(
            {
                "driver": row["driver"],
                "code": code,
                "team": row["team"],
                "current_points": float(row["current_points"]),
                "expected_future_points": expected_points - float(row["current_points"]),
                "champion_pct": champion_pct,
                "avg_final_rank": avg_final_rank,
                "driver_component": float(driver_component),
                "constructor_component": float(constructor_component),
                "qualifying_rating": float(row["qualifying"]),
                "recent_form": float(row["recent_form"]),
                "recent_form_effective": recent_form_effective,
                "avg_form_scale": avg_form_scale,
                "avg_circuit_fit": avg_circuit_fit,
                "team_uncertainty_sd": package_sd,
                "race_noise_sd": float(params.chaos * (1.0 + (100.0 - row["consistency"]) / 180.0)),
                "base_strength_now": base_strength_now,
                "base_strength_avg": base_strength_avg,
                "_order": result_order.get(code, len(result_order) + len(rows)),
            }
        )

    diagnostic = pd.DataFrame(rows).sort_values("_order").drop(columns="_order")
    return diagnostic.head(limit).reset_index(drop=True)


def describe_driver(driver_code: str, drivers: pd.DataFrame) -> dict[str, Any]:
    """Return the editable model profile for one driver.

    Parameters
    ----------
    driver_code : str
        Three-letter driver code (e.g. ``"HAM"``).
    drivers : pd.DataFrame
        Cleaned driver table containing the driver.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``driver``, ``team``, ``driver_strength``,
        ``constructor_strength``, ``risk`` and ``form``.

    Raises
    ------
    IndexError
        If ``driver_code`` is not found in ``drivers``.
    """
    clean = clean_drivers(drivers)
    row = clean.loc[clean["code"] == driver_code].iloc[0]
    return {
        "driver": row["driver"],
        "team": row["team"],
        "driver_strength": round((row["driver_rating"] + row["race_pace"] + row["racecraft"]) / 3, 1),
        "constructor_strength": round((row["team_pace"] + row["chassis"] + row["power_unit"]) / 3, 1),
        "risk": round(100 - row["reliability"], 1),
        "form": round(row["recent_form"], 1),
    }
