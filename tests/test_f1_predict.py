from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from streamlit.testing.v1 import AppTest

from f1predictor.data import load_calendar, load_drivers
from f1predictor.game import answer_f1_predict_questions, default_question_catalog
from f1predictor.parameters import SimParams
import f1predictor.report as report_module
from f1predictor.simulator import simulate_many


class F1PredictIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.drivers = load_drivers()
        cls.calendar = load_calendar()
        open_rounds = cls.calendar.loc[cls.calendar["completed"] == 0, "round"]
        cls.round_no = int(open_rounds.min())
        params = SimParams(simulations=30, seed=1234, start_round=cls.round_no)
        cls.driver_results, cls.constructor_results, cls.race_metrics = simulate_many(
            cls.drivers, cls.calendar, params
        )

    def test_simulator_exposes_question_metrics(self) -> None:
        required = {
            "win_pct",
            "pole_pct",
            "podium_pct",
            "top10_pct",
            "dnf_pct",
            "fastest_lap_pct",
            "most_positions_gained_pct",
            "team_fastest_pit_stop_pct",
            "safety_car_pct",
            "winner_from_pole_pct",
            "head_to_head_pct",
            "p1_pct",
        }
        self.assertTrue(required.issubset(self.race_metrics.columns))
        next_race = self.race_metrics.loc[self.race_metrics["round"] == self.round_no]
        self.assertAlmostEqual(float(next_race["win_pct"].sum()), 100.0, places=6)
        self.assertAlmostEqual(float(next_race["pole_pct"].sum()), 100.0, places=6)
        self.assertAlmostEqual(float(next_race["fastest_lap_pct"].sum()), 100.0, places=6)

    def test_fallback_produces_ten_answers(self) -> None:
        catalog = default_question_catalog(self.race_metrics, self.round_no)
        answers = answer_f1_predict_questions(catalog, self.race_metrics, self.round_no)
        self.assertEqual(catalog["question_id"].nunique(), 10)
        self.assertEqual(len(answers), 10)
        podium = answers.loc[answers["question_type"] == "podium"].iloc[0]
        self.assertIn("P1 ", podium["recommended_answer"])
        self.assertIn("P2 ", podium["recommended_answer"])
        self.assertIn("P3 ", podium["recommended_answer"])

    def test_variable_points_use_expected_value(self) -> None:
        race = pd.DataFrame(
            [
                {
                    "round": 1,
                    "driver": "Driver One",
                    "code": "ONE",
                    "team": "Team One",
                    "safety_car_pct": 60.0,
                }
            ]
        )
        catalog = pd.DataFrame(
            [
                {
                    "question_id": "Q1",
                    "question": "Safety Car?",
                    "question_type": "safety_car",
                    "selection_count": 1,
                    "option": "Yes",
                    "option_value": "yes",
                    "entity_type": "event",
                    "points": 1,
                },
                {
                    "question_id": "Q1",
                    "question": "Safety Car?",
                    "question_type": "safety_car",
                    "selection_count": 1,
                    "option": "No",
                    "option_value": "no",
                    "entity_type": "event",
                    "points": 10,
                },
            ]
        )
        answer = answer_f1_predict_questions(catalog, race, 1).iloc[0]
        self.assertEqual(answer["recommended_answer"], "No")
        self.assertEqual(answer["strategy"], "max_expected_points")
        self.assertAlmostEqual(float(answer["expected_game_points"]), 4.0)

    def test_yes_no_question_can_target_a_driver(self) -> None:
        race = pd.DataFrame(
            [{"round": 1, "driver": "Driver One", "code": "ONE", "team": "Team One", "top10_pct": 72.0}]
        )
        catalog = pd.DataFrame(
            [
                {
                    "question_id": "Q1", "question": "Will Driver One finish in the top 10?",
                    "question_type": "top10", "selection_count": 1, "option": "Yes",
                    "option_value": "yes", "subject_value": "ONE", "entity_type": "event",
                },
                {
                    "question_id": "Q1", "question": "Will Driver One finish in the top 10?",
                    "question_type": "top10", "selection_count": 1, "option": "No",
                    "option_value": "no", "subject_value": "ONE", "entity_type": "event",
                },
            ]
        )
        answer = answer_f1_predict_questions(catalog, race, 1).iloc[0]
        self.assertEqual(answer["recommended_answer"], "Yes")
        self.assertAlmostEqual(float(answer["model_probability_pct"]), 72.0)

    def test_excel_persists_question_catalog_and_answers(self) -> None:
        catalog = default_question_catalog(self.race_metrics, self.round_no)
        answers = answer_f1_predict_questions(catalog, self.race_metrics, self.round_no)
        old_path = report_module.MONTECARLO_RESULTS_PATH
        with TemporaryDirectory() as temporary_directory:
            report_module.MONTECARLO_RESULTS_PATH = Path(temporary_directory) / "results.xlsx"
            try:
                report_module.persist_montecarlo_results(
                    self.driver_results,
                    self.constructor_results,
                    self.race_metrics,
                    self.drivers,
                    self.calendar,
                    SimParams(simulations=30, seed=1234, start_round=self.round_no),
                    question_catalog=catalog,
                    game_answers=answers,
                )
                loaded = report_module.latest_f1_predict_data()
                self.assertIsNotNone(loaded)
                loaded_questions, loaded_answers = loaded
                self.assertEqual(loaded_questions["question_id"].nunique(), 10)
                self.assertEqual(len(loaded_answers), 10)
                saved_results = report_module.latest_montecarlo_results()
                self.assertIsNotNone(saved_results)
                self.assertEqual(len(saved_results), 5)
            finally:
                report_module.MONTECARLO_RESULTS_PATH = old_path

    def test_streamlit_renders_answer_sheet(self) -> None:
        catalog = default_question_catalog(self.race_metrics, self.round_no)
        answers = answer_f1_predict_questions(catalog, self.race_metrics, self.round_no)
        app = AppTest.from_file("app.py").run(timeout=30)
        app.session_state["simulation_results"] = (
            self.driver_results,
            self.constructor_results,
            self.race_metrics,
        )
        app.session_state["simulation_drivers"] = self.drivers
        app.session_state["simulation_calendar"] = self.calendar
        app.session_state["simulation_params"] = SimParams(
            simulations=30, seed=1234, start_round=self.round_no
        )
        app.session_state["f1_predict_catalog"] = catalog
        app.session_state["f1_predict_catalog_round"] = self.round_no
        app.session_state["f1_predict_catalog_status"] = "fallback"
        app.session_state["f1_predict_answers"] = answers
        app.run(timeout=30)
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
