"""Translate Monte Carlo event metrics into F1 Predict answer recommendations.

The official game publishes ten multiple-choice questions per round and may
change their wording and options.  This module keeps the simulator independent
from that presentation layer: a normalized question catalogue is scored
against the event-level probabilities produced by :mod:`f1predictor.simulator`.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from itertools import permutations
from typing import Any

import pandas as pd


QUESTION_COLUMNS = [
    "question_id",
    "question",
    "question_type",
    "selection_count",
    "option",
    "option_value",
    "subject_value",
    "opponent_value",
    "entity_type",
    "target_position",
    "points",
    "source_url",
    "is_official",
]

SUPPORTED_QUESTION_TYPES = {
    "podium",
    "winner",
    "pole",
    "qualifying_top5",
    "qualifying_top10",
    "exact_qualifying",
    "fastest_lap",
    "fastest_pit_stop",
    "most_positions_gained",
    "safety_car",
    "wet_race",
    "red_flag",
    "dnf",
    "first_retirement",
    "top5",
    "top10",
    "points_finish",
    "exact_finish",
    "driver_h2h",
    "qualifying_h2h",
    "team_h2h",
    "team_both_top10",
    "sprint_winner",
    "sprint_podium",
    "sprint_pole",
    "winner_from_pole",
    "classified_count",
}


def _key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def _json_mapping(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(key): _number(item) for key, item in value.items()}
    if isinstance(value, str) and value.strip():
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(data, dict):
            return {str(key): _number(item) for key, item in data.items()}
    return {}


def normalize_question_catalog(catalog: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    """Return a validated, row-per-option F1 Predict question catalogue."""
    frame = catalog.copy() if isinstance(catalog, pd.DataFrame) else pd.DataFrame(catalog)
    for column in QUESTION_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[QUESTION_COLUMNS].copy()
    frame["question_id"] = frame["question_id"].fillna("").astype(str).str.strip()
    frame["question"] = frame["question"].fillna("").astype(str).str.strip()
    frame["question_type"] = frame["question_type"].fillna("custom").astype(str).str.strip().str.lower()
    frame["option"] = frame["option"].fillna("").astype(str).str.strip()
    frame["option_value"] = frame["option_value"].fillna(frame["option"]).astype(str).str.strip()
    frame["subject_value"] = frame["subject_value"].fillna("").astype(str).str.strip()
    frame["opponent_value"] = frame["opponent_value"].fillna("").astype(str).str.strip()
    frame["entity_type"] = frame["entity_type"].fillna("").astype(str).str.strip().str.lower()
    frame["source_url"] = frame["source_url"].fillna("").astype(str).str.strip()
    frame["selection_count"] = pd.to_numeric(frame["selection_count"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    frame["target_position"] = pd.to_numeric(frame["target_position"], errors="coerce")
    frame["points"] = pd.to_numeric(frame["points"], errors="coerce")
    frame["is_official"] = frame["is_official"].map(
        lambda value: value if isinstance(value, bool) else _key(value) in {"true", "1", "yes", "si"}
    )
    frame = frame.loc[(frame["question_id"] != "") & (frame["question"] != "") & (frame["option"] != "")]
    return frame.drop_duplicates(["question_id", "option", "target_position"], keep="first").reset_index(drop=True)


def default_question_catalog(race_metrics: pd.DataFrame, round_no: int) -> pd.DataFrame:
    """Build a transparent ten-question fallback when live questions are unavailable.

    The catalogue mirrors the event concepts documented or referenced by the
    official rules/FAQ, but it is deliberately marked non-official because the
    exact ten questions and their point values can change every round.
    """
    race = race_metrics.loc[race_metrics["round"] == round_no].copy()
    if race.empty:
        return pd.DataFrame(columns=QUESTION_COLUMNS)
    race = race.sort_values("win_pct", ascending=False)
    source = "https://f1predict.formula1.com/en/game-rules"
    rows: list[dict[str, Any]] = []

    def add(
        question_id: str,
        question: str,
        question_type: str,
        options: list[tuple[str, str, str]],
        selection_count: int = 1,
    ) -> None:
        for option, value, entity_type in options:
            rows.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "question_type": question_type,
                    "selection_count": selection_count,
                    "option": option,
                    "option_value": value,
                    "subject_value": "",
                    "opponent_value": "",
                    "entity_type": entity_type,
                    "target_position": None,
                    "points": None,
                    "source_url": source,
                    "is_official": False,
                }
            )

    driver_options = [(row.driver, row.code, "driver") for row in race.itertuples()]
    team_rows = race.sort_values("team_expected_points", ascending=False).drop_duplicates("team")
    team_options = [(row.team, row.team, "team") for row in team_rows.itertuples()]
    add("Q1", "Pick the podium", "podium", driver_options, selection_count=3)
    add("Q2", "Who will take pole position?", "pole", driver_options)
    add("Q3", "Who will set the fastest lap?", "fastest_lap", driver_options)
    add("Q4", "Which team will record the fastest pit stop?", "fastest_pit_stop", team_options)
    add("Q5", "Who will gain the most positions?", "most_positions_gained", driver_options)
    add("Q6", "Will there be a Safety Car?", "safety_car", [("Yes", "yes", "event"), ("No", "no", "event")])
    add("Q7", "Will the race winner start from pole?", "winner_from_pole", [("Yes", "yes", "event"), ("No", "no", "event")])
    top_two = driver_options[:2]
    add("Q8", f"Who will finish higher: {top_two[0][0]} or {top_two[1][0]}?", "driver_h2h", top_two)
    top_teams = team_options[:2]
    add("Q9", f"Which team will score more: {top_teams[0][0]} or {top_teams[1][0]}?", "team_h2h", top_teams)
    add(
        "Q10",
        "How many drivers will be classified?",
        "classified_count",
        [("15 or fewer", "le_15", "event"), ("16 to 18", "16_18", "event"), ("19 or more", "ge_19", "event")],
    )
    return normalize_question_catalog(rows)


def _driver_row(race: pd.DataFrame, value: str) -> pd.Series | None:
    wanted = _key(value)
    matched = race.loc[
        race["code"].map(_key).eq(wanted) | race["driver"].map(_key).eq(wanted)
    ]
    return None if matched.empty else matched.iloc[0]


def _team_row(race: pd.DataFrame, value: str) -> pd.Series | None:
    wanted = _key(value)
    matched = race.loc[race["team"].map(_key).eq(wanted)]
    return None if matched.empty else matched.iloc[0]


def _event_probability(race: pd.DataFrame, column: str, option_value: str) -> float:
    probability = _number(race.iloc[0].get(column)) if not race.empty else 0.0
    return 100.0 - probability if _key(option_value) in {"no", "false", "0"} else probability


def _score_option(row: pd.Series, race: pd.DataFrame, group: pd.DataFrame) -> tuple[float, str]:
    question_type = str(row["question_type"])
    value = str(row["option_value"])
    subject_value = str(row.get("subject_value", ""))
    driver_from_option = _driver_row(race, value)
    team_from_option = _team_row(race, value)
    driver = driver_from_option if driver_from_option is not None else _driver_row(race, subject_value)
    team = team_from_option if team_from_option is not None else _team_row(race, subject_value)
    boolean_option = _key(value) in {"yes", "no", "true", "false", "1", "0", "si"}
    driver_columns = {
        "winner": "win_pct",
        "pole": "pole_pct",
        "fastest_lap": "fastest_lap_pct",
        "most_positions_gained": "most_positions_gained_pct",
        "dnf": "dnf_pct",
        "first_retirement": "first_retirement_pct",
        "top5": "top5_pct",
        "top10": "top10_pct",
        "points_finish": "points_pct",
        "podium": "podium_pct",
        "qualifying_top5": "qualifying_top5_pct",
        "qualifying_top10": "qualifying_top10_pct",
        "sprint_winner": "sprint_win_pct",
        "sprint_podium": "sprint_podium_pct",
        "sprint_pole": "sprint_pole_pct",
    }
    if question_type in driver_columns and driver is not None:
        column = driver_columns[question_type]
        probability = _number(driver.get(column))
        if boolean_option and _key(value) in {"no", "false", "0"}:
            probability = 100.0 - probability
        return probability, column
    if question_type == "exact_finish" and driver is not None:
        position = int(_number(row.get("target_position"), 1))
        return _number(driver.get(f"p{position}_pct")), f"p{position}_pct"
    if question_type == "exact_qualifying" and driver is not None:
        position = int(_number(row.get("target_position"), 1))
        return _number(driver.get(f"q{position}_pct")), f"q{position}_pct"
    if question_type == "fastest_pit_stop" and team is not None:
        probability = _number(team.get("team_fastest_pit_stop_pct"))
        if boolean_option and _key(value) in {"no", "false", "0"}:
            probability = 100.0 - probability
        return probability, "team_fastest_pit_stop_pct"
    if question_type == "team_both_top10":
        if team is not None:
            probability = _number(team.get("team_both_top10_pct"))
            if boolean_option and _key(value) in {"no", "false", "0"}:
                probability = 100.0 - probability
            return probability, "team_both_top10_pct"
        question_key = _key(row.get("question"))
        mentioned = race.loc[race["team"].map(lambda name: _key(name) in question_key)]
        if not mentioned.empty:
            probability = _number(mentioned.iloc[0].get("team_both_top10_pct"))
            if _key(value) in {"no", "false", "0"}:
                probability = 100.0 - probability
            return probability, "team_both_top10_pct"
    if question_type in {"safety_car", "wet_race", "red_flag", "winner_from_pole"}:
        column = {
            "safety_car": "safety_car_pct",
            "wet_race": "wet_race_pct",
            "red_flag": "red_flag_pct",
            "winner_from_pole": "winner_from_pole_pct",
        }[question_type]
        return _event_probability(race, column, value), column
    if question_type == "classified_count":
        normalized = _key(value)
        column = "classified_16_18_pct"
        if normalized in {"le15", "15orfewer", "15omenos"}:
            column = "classified_le_15_pct"
        elif normalized in {"ge19", "19ormore", "19omas"}:
            column = "classified_ge_19_pct"
        return _number(race.iloc[0].get(column)), column
    if question_type == "driver_h2h" and driver is not None:
        explicit_opponent = str(row.get("opponent_value", "")).strip()
        opponents = [explicit_opponent] if explicit_opponent else [
            str(item) for item in group["option_value"].tolist() if _key(item) != _key(value)
        ]
        if opponents:
            opponent = _driver_row(race, opponents[0])
            mapping = _json_mapping(driver.get("head_to_head_pct"))
            if opponent is not None:
                return _number(mapping.get(str(opponent["code"]))), f"H2H vs {opponent['code']}"
    if question_type == "qualifying_h2h" and driver is not None:
        explicit_opponent = str(row.get("opponent_value", "")).strip()
        opponents = [explicit_opponent] if explicit_opponent else [
            str(item) for item in group["option_value"].tolist() if _key(item) != _key(value)
        ]
        if opponents:
            opponent = _driver_row(race, opponents[0])
            mapping = _json_mapping(driver.get("qualifying_head_to_head_pct"))
            if opponent is not None:
                return _number(mapping.get(str(opponent["code"]))), f"Qualy H2H vs {opponent['code']}"
    if question_type == "team_h2h" and team is not None:
        explicit_opponent = str(row.get("opponent_value", "")).strip()
        opponents = [explicit_opponent] if explicit_opponent else [
            str(item) for item in group["option_value"].tolist() if _key(item) != _key(value)
        ]
        if opponents:
            opponent = _team_row(race, opponents[0])
            mapping = _json_mapping(team.get("team_head_to_head_pct"))
            if opponent is not None:
                return _number(mapping.get(str(opponent["team"]))), f"H2H vs {opponent['team']}"
    return 0.0, "unsupported"


def score_question_options(
    catalog: pd.DataFrame | list[dict[str, Any]],
    race_metrics: pd.DataFrame,
    round_no: int,
) -> pd.DataFrame:
    """Attach model probability and expected-score columns to every option."""
    questions = normalize_question_catalog(catalog)
    race = race_metrics.loc[race_metrics["round"] == round_no].copy()
    scored: list[dict[str, Any]] = []
    for _, group in questions.groupby("question_id", sort=False):
        for _, option in group.iterrows():
            probability, basis = _score_option(option, race, group)
            points = _number(option["points"], float("nan"))
            expected = probability * points / 100.0 if not math.isnan(points) else float("nan")
            scored.append(
                {
                    **option.to_dict(),
                    "model_probability_pct": max(0.0, min(100.0, probability)),
                    "expected_game_points": expected,
                    "model_basis": basis,
                    "supported": basis != "unsupported",
                }
            )
    return pd.DataFrame(scored)


def _podium_recommendation(group: pd.DataFrame, race: pd.DataFrame) -> dict[str, Any]:
    candidate_rows: list[tuple[pd.Series, pd.Series]] = []
    for _, option in group.iterrows():
        driver = _driver_row(race, str(option["option_value"]))
        if driver is not None:
            candidate_rows.append((option, driver))
    if len(candidate_rows) < 3:
        return {}
    best: tuple[float, tuple[int, int, int]] | None = None
    for indexes in permutations(range(len(candidate_rows)), 3):
        value = 0.0
        for position, candidate_index in enumerate(indexes, start=1):
            option, driver = candidate_rows[candidate_index]
            points = _number(option["points"], 0.0)
            exact = _number(driver.get(f"p{position}_pct"))
            podium = _number(driver.get("podium_pct"))
            value += points * (podium + exact) / 100.0 if points > 0 else exact + 0.15 * podium
        if best is None or value > best[0]:
            best = (value, indexes)
    assert best is not None
    answers: list[str] = []
    exact_probabilities: list[float] = []
    podium_probabilities: list[float] = []
    listed_points: list[str] = []
    for position, candidate_index in enumerate(best[1], start=1):
        option, driver = candidate_rows[candidate_index]
        answers.append(f"P{position} {option['option']}")
        exact_probabilities.append(_number(driver.get(f"p{position}_pct")))
        podium_probabilities.append(_number(driver.get("podium_pct")))
        if not pd.isna(option["points"]):
            listed_points.append(str(_number(option["points"])))
    return {
        "recommended_answer": " · ".join(answers),
        "model_probability_pct": sum(exact_probabilities) / 3.0,
        "support_probability_pct": sum(podium_probabilities) / 3.0,
        "game_points": ", ".join(listed_points),
        "expected_game_points": best[0] if listed_points else float("nan"),
        "model_basis": "asignacion P1-P3; el acierto de posicion duplica el valor base",
        "strategy": "max_expected_points" if listed_points else "max_probability",
    }


def answer_f1_predict_questions(
    catalog: pd.DataFrame | list[dict[str, Any]],
    race_metrics: pd.DataFrame,
    round_no: int,
) -> pd.DataFrame:
    """Return one scoring-aware recommended answer for each game question."""
    questions = normalize_question_catalog(catalog)
    race = race_metrics.loc[race_metrics["round"] == round_no].copy()
    option_scores = score_question_options(questions, race_metrics, round_no)
    answers: list[dict[str, Any]] = []
    for question_id, group in option_scores.groupby("question_id", sort=False):
        first = group.iloc[0]
        selection_count = int(first["selection_count"])
        if first["question_type"] == "podium" and selection_count >= 3:
            choice = _podium_recommendation(group, race)
        elif selection_count > 1:
            supported = group.loc[group["supported"]].copy()
            has_points = supported["expected_game_points"].notna().any()
            sort_column = "expected_game_points" if has_points else "model_probability_pct"
            selected_rows = supported.sort_values(sort_column, ascending=False).head(selection_count)
            choice = {
                "recommended_answer": " · ".join(selected_rows["option"].astype(str)),
                "model_probability_pct": selected_rows["model_probability_pct"].mean(),
                "support_probability_pct": selected_rows["model_probability_pct"].mean(),
                "game_points": ", ".join(selected_rows["points"].dropna().astype(str)),
                "expected_game_points": selected_rows["expected_game_points"].sum(min_count=1),
                "model_basis": ", ".join(selected_rows["model_basis"].drop_duplicates().astype(str)),
                "strategy": "max_expected_points" if has_points else "max_probability",
            }
        else:
            supported = group.loc[group["supported"]].copy()
            if supported.empty:
                choice = {
                    "recommended_answer": "Requiere analisis LLM/manual",
                    "model_probability_pct": float("nan"),
                    "support_probability_pct": float("nan"),
                    "game_points": "",
                    "expected_game_points": float("nan"),
                    "model_basis": "tipo no soportado por Monte Carlo",
                    "strategy": "manual",
                }
            else:
                has_points = supported["expected_game_points"].notna().any()
                sort_column = "expected_game_points" if has_points else "model_probability_pct"
                selected = supported.sort_values(sort_column, ascending=False).iloc[0]
                choice = {
                    "recommended_answer": selected["option"],
                    "model_probability_pct": selected["model_probability_pct"],
                    "support_probability_pct": selected["model_probability_pct"],
                    "game_points": "" if pd.isna(selected["points"]) else selected["points"],
                    "expected_game_points": selected["expected_game_points"],
                    "model_basis": selected["model_basis"],
                    "strategy": "max_expected_points" if has_points else "max_probability",
                }
        answers.append(
            {
                "question_id": question_id,
                "question": first["question"],
                "question_type": first["question_type"],
                **choice,
                "official_question": bool(first["is_official"]),
                "source_url": first["source_url"],
            }
        )
    return pd.DataFrame(answers)
