"""Global configuration for the F1 predictor app.

Defines filesystem paths, external API base URLs, application metadata,
points-scoring tables, driver rating column names and team colour palette.
All values are module-level constants; nothing is imported from other
f1predictor modules here to avoid circular dependencies.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_ROOT / "resultados"
REPORT_DIR = PROJECT_ROOT / "reporte"
FIGS_DIR = PROJECT_ROOT / "figs"
DRIVERS_PATH = DATA_DIR / "drivers_seed.csv"
CALENDAR_PATH = DATA_DIR / "calendar_seed.csv"
LLM_ANALYSIS_PATH = RESULTS_DIR / "analisis_llm.md"
MONTECARLO_RESULTS_PATH = RESULTS_DIR / "resultados_montecarlo.xlsx"
REPORT_TEMPLATE_PATH = REPORT_DIR / "reporte_f1_template.tex"
REPORT_TEX_PATH = REPORT_DIR / "reporte_f1.tex"
REPORT_PDF_PATH = REPORT_DIR / "reporte_f1.pdf"

APP_TITLE = "F1 Predict Decision Lab"

DEFAULT_SEASON: int = date.today().year

APP_CAPTION = (
    f"Responde las 10 preguntas de F1 Predict para cada GP de {DEFAULT_SEASON} con Monte Carlo, "
    "puntos esperados, contexto LLM y una proyeccion secundaria del campeonato."
)

F1_DRIVERS_URL = "https://www.formula1.com/en/drivers"
F1_STANDINGS_URL = f"https://www.formula1.com/en/results/{DEFAULT_SEASON}/drivers"
F1_CALENDAR_URL = f"https://www.formula1.com/en/racing/{DEFAULT_SEASON}"
FASTF1_URL = "https://docs.fastf1.dev/"
OPENF1_URL = "https://openf1.org/docs/"
OPENAI_WEB_SEARCH_URL = "https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses"
JOLPICA_BASE_URL = "https://api.jolpi.ca"
OPENF1_BASE_URL = "https://api.openf1.org/v1"

RACE_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
SPRINT_POINTS = [8, 7, 6, 5, 4, 3, 2, 1]

DRIVER_RATING_COLUMNS = [
    "driver_rating",
    "qualifying",
    "race_pace",
    "consistency",
    "tyre_management",
    "wet_skill",
    "racecraft",
    "team_pace",
    "chassis",
    "power_unit",
    "strategy",
    "reliability",
    "recent_form",
]

DRIVER_REQUIRED_COLUMNS = [
    "driver",
    "code",
    "team",
    "nationality",
    "current_points",
    *DRIVER_RATING_COLUMNS,
]

CALENDAR_REQUIRED_COLUMNS = [
    "round",
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

TEAM_COLORS = {
    "Mercedes": "#00a19c",
    "Ferrari": "#dc0000",
    "McLaren": "#ff8700",
    "Red Bull Racing": "#1e41ff",
    "Haas F1 Team": "#b6babd",
    "Alpine": "#2293d1",
    "Alpine F1 Team": "#2293d1",
    "Racing Bulls": "#6692ff",
    "RB F1 Team": "#6692ff",
    "Audi": "#c0c0c0",
    "Williams": "#00a0de",
    "Cadillac": "#b08d57",
    "Cadillac F1 Team": "#b08d57",
    "Aston Martin": "#006f62",
}
