"""Streamlit UI for the F1 Championship Lab.

Builds the complete single-page Streamlit application: sidebar controls,
editable driver and calendar tables, data-update buttons (public APIs and
Formula1.com via LLM), simulation trigger, result tabs and the LLM
narrative-analysis panel.
"""

from __future__ import annotations

import html
import json

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from f1predictor.config import (
    APP_CAPTION,
    APP_TITLE,
    CALENDAR_PATH,
    DEFAULT_SEASON,
    DRIVER_RATING_COLUMNS,
    DRIVERS_PATH,
    F1_CALENDAR_URL,
    F1_DRIVERS_URL,
    F1_STANDINGS_URL,
    FASTF1_URL,
    LLM_ANALYSIS_PATH,
    MONTECARLO_RESULTS_PATH,
    OPENAI_WEB_SEARCH_URL,
    OPENF1_URL,
    REPORT_PDF_PATH,
    REPORT_TEX_PATH,
    TEAM_COLORS,
)
from f1predictor.data import (
    apply_official_formula1_update,
    clean_calendar,
    clean_drivers,
    dataframe_from_csv_text,
    dataframe_to_csv_text,
    load_calendar,
    load_drivers,
)
from f1predictor.llm import (
    api_key_available,
    build_analysis_payload,
    call_llm_analysis,
    call_llm_context_search,
    call_llm_formula1_official_update,
    default_model,
)
from f1predictor.parameters import SimParams
from f1predictor.public_apis import refresh_model_inputs_from_public_apis
from f1predictor.report import latest_llm_analysis, latest_montecarlo_results, persist_llm_analysis, persist_montecarlo_results, render_report
from f1predictor.simulator import build_driver_diagnostics, describe_driver, simulate_many


def configure_page() -> None:
    """Configure the Streamlit page and load environment variables.

    Sets the page title, icon and wide layout, loads ``.env`` variables
    and injects application-level CSS styles.
    """
    st.set_page_config(page_title=APP_TITLE, page_icon="F1", layout="wide")
    load_dotenv()
    inject_style()


def inject_style() -> None:
    """Inject compact CSS styles into the Streamlit app.

    Reduces container padding and adds borders and rounded corners to
    metric widgets.
    """
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.15rem; padding-bottom: 2rem;}
        [data-testid="stMetric"] {
            border: 1px solid #d0d7de;
            border-radius: 8px;
            padding: 12px 14px;
        }
        h1, h2, h3 {letter-spacing: 0;}
        .source-line {color: #59636e; font-size: 0.88rem;}
        .small-note {color: #59636e; font-size: 0.92rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_copy_button(text: str, key: str) -> None:
    """Render a browser clipboard copy button via an embedded HTML component.

    Parameters
    ----------
    text : str
        Text content to copy to the clipboard when the button is clicked.
    key : str
        Unique suffix used to generate element IDs for the button and
        status span, avoiding collisions when multiple buttons are rendered.
    """
    button_id = f"copy-llm-{key}"
    status_id = f"copy-llm-status-{key}"
    payload = json.dumps(text)
    components.html(
        f"""
        <div style="display:flex; justify-content:flex-end; margin:0 0 0.5rem 0;">
            <button
                id="{button_id}"
                type="button"
                style="border:1px solid #d0d7de; border-radius:0.5rem; background:#f6f8fa; color:#24292f; padding:0.4rem 0.8rem; font-size:0.9rem; cursor:pointer;"
            >
                {html.escape("Copiar analisis")}
            </button>
            <span id="{status_id}" style="margin-left:0.5rem; font-size:0.85rem; color:#57606a;"></span>
        </div>
        <script>
        const copyButton = document.getElementById({json.dumps(button_id)});
        const status = document.getElementById({json.dumps(status_id)});
        const text = {payload};
        copyButton?.addEventListener('click', async () => {{
            try {{
                await navigator.clipboard.writeText(text);
                if (status) {{
                    status.textContent = 'Copiado';
                    setTimeout(() => {{ status.textContent = ''; }}, 2000);
                }}
            }} catch (error) {{
                if (status) {{ status.textContent = 'No se pudo copiar'; }}
            }}
        }});
        </script>
        """,
        height=42,
    )


@st.cache_data(show_spinner=False)
def load_drivers_cached() -> pd.DataFrame:
    """Load and clean default driver data with Streamlit result caching.

    Returns
    -------
    pd.DataFrame
        Cleaned driver seed table, cached across reruns.
    """
    return clean_drivers(load_drivers())


@st.cache_data(show_spinner=False)
def load_calendar_cached() -> pd.DataFrame:
    """Load and clean default calendar data with Streamlit result caching.

    Returns
    -------
    pd.DataFrame
        Cleaned calendar seed table, cached across reruns.
    """
    return clean_calendar(load_calendar())


@st.cache_data(show_spinner="Simulando temporada F1...")
def run_simulation_cached(
    drivers_csv: str,
    calendar_csv: str,
    params: SimParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a full season simulation from CSV text snapshots with caching.

    Accepts CSV strings rather than DataFrames so that Streamlit can hash
    the inputs for cache invalidation.

    Parameters
    ----------
    drivers_csv : str
        CSV serialisation of the driver table (via ``dataframe_to_csv_text``).
    calendar_csv : str
        CSV serialisation of the calendar table.
    params : SimParams
        Simulation parameters; the frozen dataclass is hashable by Streamlit.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Three DataFrames returned by ``simulate_many``: driver results,
        constructor results and per-race winner probabilities.
    """
    drivers = clean_drivers(dataframe_from_csv_text(drivers_csv))
    calendar = clean_calendar(dataframe_from_csv_text(calendar_csv))
    return simulate_many(drivers, calendar, params)



def render_sidebar(calendar: pd.DataFrame) -> tuple[SimParams, str, int, int]:
    """Render the sidebar with model controls and return user selections.

    Parameters
    ----------
    calendar : pd.DataFrame
        Cleaned calendar table used to populate the starting-round selector.

    Returns
    -------
    tuple[SimParams, str, int, int]
        * **params** – ``SimParams`` built from the sidebar widgets.
        * **model** – OpenAI model name entered by the user.
        * **api_season** – Season year selected for API data fetching.
        * **history_seasons** – Number of historical seasons used as priors.
    """
    st.sidebar.header("Motor")
    simulations = st.sidebar.slider("Simulaciones", 500, 50000, 4000, step=500)
    seed = st.sidebar.number_input("Semilla", min_value=1, value=DEFAULT_SEASON, step=1)
    open_rounds = calendar.loc[calendar["completed"] == 0, "round"].tolist()
    default_round = int(open_rounds[0]) if open_rounds else int(calendar["round"].max())
    start_round = st.sidebar.selectbox(
        "Simular desde ronda",
        options=[int(v) for v in calendar["round"].tolist()],
        index=[int(v) for v in calendar["round"].tolist()].index(default_round),
    )
    include_sprints = st.sidebar.toggle("Incluir sprints pendientes", value=True)

    st.sidebar.header("Pesos")
    driver_weight = st.sidebar.slider("Piloto", 0.10, 0.80, 0.42, step=0.02)
    constructor_weight = st.sidebar.slider("Constructor", 0.10, 0.80, 0.48, step=0.02)
    form_weight = st.sidebar.slider("Forma reciente", 0.00, 0.30, 0.06, step=0.02)
    form_decay_races = st.sidebar.slider("Duracion forma (carreras)", 0.5, 8.0, 2.5, step=0.5)
    qualifying_weight = st.sidebar.slider("Peso de clasificacion", 0.20, 1.20, 0.72, step=0.02)

    st.sidebar.header("Incertidumbre")
    chaos = st.sidebar.slider("Caos carrera", 1.0, 14.0, 5.5, step=0.5)
    reliability_multiplier = st.sidebar.slider("Riesgo fiabilidad", 0.4, 2.4, 1.0, step=0.1)
    weather_multiplier = st.sidebar.slider("Clima", 0.3, 2.5, 1.0, step=0.1)
    safety_car_multiplier = st.sidebar.slider("Safety car", 0.3, 2.5, 1.0, step=0.1)
    development_drift = st.sidebar.slider("Desarrollo por equipo", 0.0, 8.0, 2.4, step=0.2)
    team_uncertainty = st.sidebar.slider("Incertidumbre auto", 0.0, 10.0, 6.0, step=0.5)

    st.sidebar.header("Datos")
    api_season = st.sidebar.number_input("Temporada inputs", min_value=2023, max_value=2100, value=DEFAULT_SEASON, step=1)
    history_seasons = st.sidebar.number_input("Temporadas historicas", min_value=0, max_value=5, value=3, step=1)

    st.sidebar.header("LLM")
    model = st.sidebar.text_input("Modelo OpenAI", value=default_model())
    st.sidebar.caption(f"OPENAI_API_KEY: {'detectada' if api_key_available() else 'no detectada'}")

    params = SimParams(
        simulations=int(simulations),
        seed=int(seed),
        start_round=int(start_round),
        include_sprints=bool(include_sprints),
        driver_weight=float(driver_weight),
        constructor_weight=float(constructor_weight),
        form_weight=float(form_weight),
        form_decay_races=float(form_decay_races),
        qualifying_weight=float(qualifying_weight),
        chaos=float(chaos),
        reliability_multiplier=float(reliability_multiplier),
        weather_multiplier=float(weather_multiplier),
        safety_car_multiplier=float(safety_car_multiplier),
        development_drift=float(development_drift),
        team_uncertainty=float(team_uncertainty),
    )
    return params, model, int(api_season), int(history_seasons)


def render_data_editors(
    default_drivers: pd.DataFrame,
    default_calendar: pd.DataFrame,
    model: str,
    api_season: int,
    history_seasons: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Render editable driver and calendar tables with data-update buttons.

    Manages the ``drivers_df`` and ``calendar_df`` keys in
    ``st.session_state``.  Provides buttons to refresh from public APIs
    and from Formula1.com via LLM web search.

    Parameters
    ----------
    default_drivers : pd.DataFrame
        Fallback driver table used when session state is empty.
    default_calendar : pd.DataFrame
        Fallback calendar table used when session state is empty.
    model : str
        OpenAI model identifier for the LLM update button.
    api_season : int
        Season year passed to the API refresh functions.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Cleaned ``(drivers, calendar)`` DataFrames reflecting any user edits.
    """
    if "drivers_df" not in st.session_state:
        st.session_state["drivers_df"] = default_drivers.copy()
    if "calendar_df" not in st.session_state:
        st.session_state["calendar_df"] = default_calendar.copy()

    col1, col2, col3 = st.columns([1.1, 1.1, 2.2])
    with col1:
        if st.button("Actualizar info (APIs publicas)", help="Usa Jolpica/Ergast y OpenF1 para standings, historico, calendario, clima y race control; guarda CSV en disco."):
            try:
                with st.spinner("Consultando Jolpica/OpenF1 y recalibrando inputs..."):
                    result = refresh_model_inputs_from_public_apis(
                        st.session_state["drivers_df"],
                        st.session_state["calendar_df"],
                        season=api_season,
                        history_seasons=history_seasons,
                    )
                st.session_state["drivers_df"] = result.drivers
                st.session_state["calendar_df"] = result.calendar
                st.session_state["drivers_df"].to_csv(DRIVERS_PATH, index=False)
                st.session_state["calendar_df"].to_csv(CALENDAR_PATH, index=False)
                load_drivers_cached.clear()
                load_calendar_cached.clear()
                run_simulation_cached.clear()
                st.session_state["data_update_log"] = {
                    "title": "APIs publicas",
                    "summary": result.summary,
                    "sources": result.sources,
                    "errors": result.errors,
                }
                st.success(f"Inputs actualizados y guardados en {DRIVERS_PATH.name} y {CALENDAR_PATH.name}.")
            except Exception as exc:
                st.error(f"No se pudo actualizar con APIs publicas: {exc}")

    with col2:
        if api_key_available():
            if st.button("Actualizar F1.com con LLM", help="Usa OpenAI web_search restringido a Formula1.com para standings y calendario oficial; guarda CSV en disco."):
                try:
                    with st.spinner("Buscando informacion oficial en Formula1.com..."):
                        payload = call_llm_formula1_official_update(
                            model,
                            st.session_state["drivers_df"],
                            st.session_state["calendar_df"],
                            season=api_season,
                        )
                        drivers_df, calendar_df = apply_official_formula1_update(
                            st.session_state["drivers_df"],
                            st.session_state["calendar_df"],
                            payload,
                        )
                    st.session_state["drivers_df"] = drivers_df
                    st.session_state["calendar_df"] = calendar_df
                    st.session_state["drivers_df"].to_csv(DRIVERS_PATH, index=False)
                    st.session_state["calendar_df"].to_csv(CALENDAR_PATH, index=False)
                    load_drivers_cached.clear()
                    load_calendar_cached.clear()
                    run_simulation_cached.clear()
                    sources = payload.get("sources", pd.DataFrame())
                    st.session_state["data_update_log"] = {
                        "title": "Formula1.com via LLM web_search",
                        "summary": [
                            f"{len(payload.get('drivers', pd.DataFrame()))} pilotos oficiales recibidos.",
                            f"{len(payload.get('calendar', pd.DataFrame()))} eventos de calendario recibidos.",
                        ],
                        "sources": sources["source"].dropna().tolist() if not sources.empty and "source" in sources.columns else [],
                        "errors": [],
                    }
                    st.success(f"Datos oficiales guardados en {DRIVERS_PATH.name} y {CALENDAR_PATH.name}.")
                except Exception as exc:
                    st.error(f"No se pudo actualizar Formula1.com con LLM: {exc}")
        else:
            st.caption("Agrega OPENAI_API_KEY para Formula1.com via LLM.")

    with col3:
        st.markdown(
            '<div class="small-note">Los ratings son semillas editables. La simulacion combina puntos actuales, ritmo piloto/equipo, pista, clasificacion, clima y fiabilidad.</div>',
            unsafe_allow_html=True,
        )

    if "data_update_log" in st.session_state:
        log = st.session_state["data_update_log"]
        with st.expander(f"Ultima actualizacion: {log.get('title', 'datos')}", expanded=False):
            for line in log.get("summary", []):
                st.write(f"- {line}")
            if log.get("sources"):
                st.caption("Fuentes")
                for source in log["sources"]:
                    st.write(source)
            if log.get("errors"):
                st.caption("Avisos")
                for error in log["errors"]:
                    st.write(f"- {error}")

    tab_drivers, tab_calendar = st.tabs(["Pilotos", "Calendario"])
    with tab_drivers:
        edited_drivers = st.data_editor(
            st.session_state["drivers_df"],
            width="stretch",
            num_rows="fixed",
            column_config={
                "current_points": st.column_config.NumberColumn("current_points", min_value=0),
                **{
                    column: st.column_config.NumberColumn(column, min_value=1, max_value=100)
                    for column in DRIVER_RATING_COLUMNS
                },
            },
        )
    with tab_calendar:
        edited_calendar = st.data_editor(
            st.session_state["calendar_df"],
            width="stretch",
            num_rows="fixed",
            column_config={
                "sprint_remaining": st.column_config.CheckboxColumn("sprint_remaining"),
                "completed": st.column_config.CheckboxColumn("completed"),
                "downforce": st.column_config.NumberColumn("downforce", min_value=0, max_value=100),
                "power": st.column_config.NumberColumn("power", min_value=0, max_value=100),
                "tyre_stress": st.column_config.NumberColumn("tyre_stress", min_value=0, max_value=100),
                "overtake_difficulty": st.column_config.NumberColumn("overtake_difficulty", min_value=0, max_value=100),
                "weather_risk": st.column_config.NumberColumn("weather_risk", min_value=0, max_value=100),
                "safety_car_risk": st.column_config.NumberColumn("safety_car_risk", min_value=0, max_value=100),
                "qualifying_importance": st.column_config.NumberColumn("qualifying_importance", min_value=0, max_value=100),
            },
        )
    return clean_drivers(edited_drivers), clean_calendar(edited_calendar)


def render_driver_view(driver_results: pd.DataFrame) -> None:
    """Render driver championship probability metrics and chart.

    Displays key metrics for the leading driver, a horizontal bar chart
    of top-12 championship probabilities and the full results table.

    Parameters
    ----------
    driver_results : pd.DataFrame
        Driver championship summary returned by ``simulate_many``.
    """
    favorite = driver_results.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Favorito pilotos", favorite["driver"], f"{favorite['champion_pct']:.1f}% campeon")
    col2.metric("Puntos esperados", favorite["driver"], f"{favorite['expected_points']:.1f}")
    col3.metric("Top 3", favorite["driver"], f"{favorite['top3_pct']:.1f}%")
    col4.metric("Ranking medio", favorite["driver"], f"{favorite['avg_final_rank']:.2f}")

    top = driver_results.head(12).sort_values("champion_pct")
    fig = px.bar(
        top,
        x="champion_pct",
        y="driver",
        color="team",
        orientation="h",
        text=top["champion_pct"].map(lambda value: f"{value:.1f}%"),
        labels={"champion_pct": "Probabilidad de campeonato", "driver": "Piloto"},
        color_discrete_map=TEAM_COLORS,
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(driver_results.round(2), width="stretch", hide_index=True)


def render_constructor_view(constructor_results: pd.DataFrame) -> None:
    """Render constructor championship probability metrics and chart.

    Displays key metrics for the leading constructor, a horizontal bar
    chart of championship probabilities and the full results table.

    Parameters
    ----------
    constructor_results : pd.DataFrame
        Constructor championship summary returned by ``simulate_many``.
    """
    leader = constructor_results.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Favorito constructores", leader["team"], f"{leader['champion_pct']:.1f}%")
    c2.metric("Puntos esperados", leader["team"], f"{leader['expected_points']:.1f}")
    c3.metric("Top 3", leader["team"], f"{leader['top3_pct']:.1f}%")

    chart = constructor_results.sort_values("champion_pct")
    fig = px.bar(
        chart,
        x="champion_pct",
        y="team",
        color="team",
        orientation="h",
        text=chart["champion_pct"].map(lambda value: f"{value:.1f}%"),
        labels={"champion_pct": "Probabilidad de campeonato", "team": "Equipo"},
        color_discrete_map=TEAM_COLORS,
    )
    fig.update_layout(height=460, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(constructor_results.round(2), width="stretch", hide_index=True)


def render_race_view(race_winners: pd.DataFrame, calendar: pd.DataFrame, drivers: pd.DataFrame) -> None:
    """Render race-specific winner probabilities and a driver profile panel.

    Allows the user to select an upcoming Grand Prix and a driver to inspect.

    Parameters
    ----------
    race_winners : pd.DataFrame
        Per-race winner probabilities from ``simulate_many``.
    calendar : pd.DataFrame
        Cleaned calendar table used to list open races.
    drivers : pd.DataFrame
        Cleaned driver table used to populate the driver-profile selector.
    """
    open_races = calendar.loc[calendar["completed"] == 0]
    if open_races.empty:
        st.info("No hay carreras pendientes en el calendario.")
        return
    race_labels = {
        int(row["round"]): f"R{int(row['round'])} - {row['grand_prix']}"
        for _, row in open_races.iterrows()
    }
    selected = st.selectbox("Gran Premio", options=list(race_labels), format_func=lambda value: race_labels[value])
    race_probs = race_winners.loc[race_winners["round"] == selected].copy()
    if race_probs.empty:
        st.warning("No hay probabilidades de ganador para esta ronda. Revisa que se haya simulado desde una ronda anterior o igual.")
    else:
        top = race_probs.head(12).sort_values("win_pct")
        fig = px.bar(
            top,
            x="win_pct",
            y="driver",
            color="team",
            orientation="h",
            text=top["win_pct"].map(lambda value: f"{value:.1f}%"),
            labels={"win_pct": "Probabilidad de victoria", "driver": "Piloto"},
            color_discrete_map=TEAM_COLORS,
        )
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=20, b=10))
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(race_probs.round(2), width="stretch", hide_index=True)

    driver_code = st.selectbox("Perfil de piloto", options=drivers["code"].tolist(), format_func=lambda code: f"{code} - {drivers.loc[drivers['code'] == code, 'driver'].iloc[0]}")
    profile = describe_driver(driver_code, drivers)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Piloto", profile["driver_strength"])
    c2.metric("Constructor", profile["constructor_strength"])
    c3.metric("Riesgo fiabilidad", profile["risk"])
    c4.metric("Forma", profile["form"])


def render_diagnostic_view(
    driver_results: pd.DataFrame | None,
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    params: SimParams,
) -> None:
    """Render deterministic model diagnostics for the simulated championship.

    Parameters
    ----------
    driver_results : pd.DataFrame or None
        Driver championship summary from the main simulation, or ``None``
        if the simulation has not been run yet.
    drivers : pd.DataFrame
        Cleaned driver seed table with current ratings.
    calendar : pd.DataFrame
        Cleaned calendar table used to derive circuit-fit metrics.
    params : SimParams
        Current simulation parameters used for the diagnostics calculation.
    """
    st.subheader("Diagnostico")
    if driver_results is None:
        st.info("Primero corre la simulacion.")
        return

    diagnostics = build_driver_diagnostics(driver_results, drivers, calendar, params, limit=12)
    favorite = diagnostics.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Puntos actuales", favorite["driver"], f"{favorite['current_points']:.1f}")
    c2.metric("Auto", favorite["driver"], f"{favorite['constructor_component']:.1f}")
    c3.metric("Piloto", favorite["driver"], f"{favorite['driver_component']:.1f}")
    c4.metric("Ruido carrera", favorite["driver"], f"{favorite['race_noise_sd']:.1f} sd")

    st.dataframe(
        diagnostics.round(2),
        width="stretch",
        hide_index=True,
        column_config={
            "expected_future_points": st.column_config.NumberColumn("future_points"),
            "champion_pct": st.column_config.NumberColumn("champion_pct"),
            "driver_component": st.column_config.NumberColumn("piloto"),
            "constructor_component": st.column_config.NumberColumn("auto"),
            "qualifying_rating": st.column_config.NumberColumn("clasificacion"),
            "recent_form_effective": st.column_config.NumberColumn("forma_regresada"),
            "avg_form_scale": st.column_config.NumberColumn("escala_forma_media"),
            "avg_circuit_fit": st.column_config.NumberColumn("fit_pistas_medio"),
            "team_uncertainty_sd": st.column_config.NumberColumn("incertidumbre_auto_sd"),
            "race_noise_sd": st.column_config.NumberColumn("ruido_carrera_sd"),
        },
    )

    fig = px.scatter(
        diagnostics,
        x="constructor_component",
        y="driver_component",
        size="champion_pct",
        color="team",
        hover_name="driver",
        labels={"constructor_component": "Auto", "driver_component": "Piloto"},
        color_discrete_map=TEAM_COLORS,
    )
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")


def render_model_view() -> None:
    """Render a static explanation of the simulation model methodology.

    Describes driver ratings, constructor contributions, circuit modifiers,
    qualifying, retirements and points accumulation.  Includes hyperlinks
    to external data sources.
    """
    st.subheader("Como piensa el modelo")
    st.markdown(
        """
        - Cada piloto tiene ratings de habilidad, clasificacion, ritmo de carrera, consistencia, neumaticos, lluvia y racecraft.
        - Cada equipo aporta ritmo de auto, chasis, unidad de potencia, estrategia y fiabilidad.
        - Las temporadas previas construyen un prior historico; la temporada actual solo ajusta ese prior segun la evidencia disponible.
        - Los ratings derivados de pocas carreras se regresan a la media para evitar sobreconfianza temprana.
        - La forma reciente pesa mas en los siguientes eventos y se diluye conforme avanza el calendario.
        - El paquete de cada equipo se simula con incertidumbre persistente, mayor cuando hay poca evidencia de temporada.
        - Cada circuito modifica la mezcla: carga aerodinamica, potencia, degradacion, dificultad de adelantar, clima y safety car.
        - Cada GP se simula clasificando primero y corriendo despues; los abandonos se modelan por fiabilidad y estres de pista.
        - La temporada suma puntos actuales mas carreras pendientes; los sprints pendientes pueden incluirse o apagarse.
        """
    )
    st.markdown(
        f"""
        <div class="source-line">
        Fuentes y metodologia:
        <a href="{F1_DRIVERS_URL}" target="_blank">drivers F1</a>,
        <a href="{F1_STANDINGS_URL}" target="_blank">standings F1</a>,
        <a href="{F1_CALENDAR_URL}" target="_blank">calendario F1</a>,
        <a href="{FASTF1_URL}" target="_blank">FastF1</a>,
        <a href="{OPENF1_URL}" target="_blank">OpenF1</a>,
        <a href="{OPENAI_WEB_SEARCH_URL}" target="_blank">OpenAI web_search</a>.
        La sintesis completa esta en docs/research_f1_models.md.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_llm_view(
    driver_results: pd.DataFrame | None,
    constructor_results: pd.DataFrame | None,
    race_winners: pd.DataFrame | None,
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    model: str,
    params: SimParams,
) -> None:
    """Render the LLM context-search and narrative-analysis panel.

    Provides a button to fetch web-based race context, a text area for
    qualitative notes, a button to generate a narrative LLM analysis and
    a copy button for the generated text.

    Parameters
    ----------
    driver_results : pd.DataFrame or None
        Driver championship summary; ``None`` if simulation has not run yet.
    constructor_results : pd.DataFrame or None
        Constructor championship summary; ``None`` if simulation not run.
    race_winners : pd.DataFrame or None
        Per-race winner probabilities; ``None`` if simulation not run.
    drivers : pd.DataFrame
        Cleaned driver seed table for context.
    calendar : pd.DataFrame
        Cleaned calendar table for context.
    model : str
        OpenAI model identifier to use for the API calls.
    """
    st.subheader("LLM en la ecuacion")
    if not api_key_available():
        st.warning("No detecte OPENAI_API_KEY en el entorno. Revisa el archivo .env en esta carpeta.")
        return

    open_rounds = calendar.loc[calendar["completed"] == 0, "round"].tolist()
    selected_round = int(open_rounds[0]) if open_rounds else int(calendar["round"].max())
    if st.button("Buscar contexto del proximo GP"):
        try:
            with st.spinner("Buscando clima, parrilla, upgrades y sanciones..."):
                st.session_state["llm_notes"] = call_llm_context_search(model, drivers, calendar, selected_round)
        except Exception as exc:
            st.error(f"No se pudo buscar contexto: {exc}")

    notes = st.text_area(
        "Contexto cualitativo",
        key="llm_notes",
        height=180,
        placeholder="Ejemplo: Miami tiene lluvia probable; Antonelli sale en pole; McLaren trajo upgrades; Hadjar tiene penalizacion.",
    )

    if st.button("Generar analisis LLM", type="primary"):
        if driver_results is None or constructor_results is None or race_winners is None:
            st.warning("Primero corre la simulacion.")
        else:
            next_race_winners = race_winners.loc[race_winners["round"] == selected_round]
            diagnostics = build_driver_diagnostics(driver_results, drivers, calendar, params, limit=12)
            payload = build_analysis_payload(
                driver_results,
                constructor_results,
                next_race_winners,
                drivers,
                calendar,
                notes,
                diagnostics=diagnostics,
            )
            try:
                with st.spinner("Consultando al LLM..."):
                    answer = call_llm_analysis(model, payload).strip()
                    persist_llm_analysis(answer)
                    st.session_state["llm_answer"] = answer
                    st.session_state["llm_analysis_path"] = str(LLM_ANALYSIS_PATH)
            except Exception as exc:
                st.error(f"No se pudo generar analisis: {exc}")

    if st.session_state.get("llm_answer"):
        answer = st.session_state["llm_answer"]
        render_copy_button(answer, "analysis")
        st.markdown(answer)

    report_ready = (
        driver_results is not None
        and constructor_results is not None
        and race_winners is not None
        and bool(st.session_state.get("llm_answer"))
    )
    st.divider()
    if st.button("Guardar reporte", disabled=not report_ready):
        try:
            with st.spinner("Generando graficos, renderizando LaTeX y compilando PDF..."):
                pdf_path = render_report(
                    driver_results,
                    constructor_results,
                    race_winners,
                    drivers,
                    calendar,
                    params,
                    st.session_state.get("llm_answer", ""),
                )
            st.success(f"Reporte guardado: {pdf_path.name}")
            st.caption(f"Markdown: {LLM_ANALYSIS_PATH} | TeX: {REPORT_TEX_PATH} | PDF: {REPORT_PDF_PATH}")
        except Exception as exc:
            st.error(f"No se pudo guardar el reporte: {exc}")
    elif not report_ready:
        st.caption("El reporte se habilita despues de correr la simulacion y generar el analisis LLM.")


def render_app() -> None:
    """Main Streamlit application flow.

    Loads and caches seed data, renders the sidebar, data editors and
    the simulation button.  On successful simulation, populates the driver,
    constructor, race, model and LLM tabs with results.
    """
    default_drivers = load_drivers_cached()
    default_calendar = load_calendar_cached()

    # Load persisted data on first run of the session
    if "simulation_results" not in st.session_state:
        saved = latest_montecarlo_results()
        if saved is not None:
            dr, cr, rw, saved_drivers, saved_calendar = saved
            st.session_state["simulation_results"] = (dr, cr, rw)
            st.session_state["simulation_drivers"] = clean_drivers(saved_drivers)
            st.session_state["simulation_calendar"] = clean_calendar(saved_calendar)
    if "llm_answer" not in st.session_state:
        analysis = latest_llm_analysis()
        if analysis:
            st.session_state["llm_answer"] = analysis

    params, llm_model, api_season, history_seasons = render_sidebar(default_calendar)

    st.title(APP_TITLE)
    st.caption(APP_CAPTION)

    drivers, calendar = render_data_editors(default_drivers, default_calendar, llm_model, api_season, history_seasons)

    if st.button("Simular campeonato", type="primary"):
        try:
            simulation_results = run_simulation_cached(
                dataframe_to_csv_text(drivers),
                dataframe_to_csv_text(calendar),
                params,
            )
            persist_montecarlo_results(*simulation_results, drivers, calendar, params)
            st.session_state["simulation_results"] = simulation_results
            st.session_state["simulation_drivers"] = drivers.copy()
            st.session_state["simulation_calendar"] = calendar.copy()
            st.session_state["simulation_params"] = params
            st.session_state.pop("llm_answer", None)
            st.session_state.pop("llm_analysis_path", None)
            st.success(f"Resultados Monte Carlo guardados en {MONTECARLO_RESULTS_PATH}.")
        except Exception as exc:
            st.error(f"No se pudo correr la simulacion: {exc}")

    results = st.session_state.get("simulation_results")
    sim_drivers = st.session_state.get("simulation_drivers", drivers)
    sim_calendar = st.session_state.get("simulation_calendar", calendar)
    sim_params = st.session_state.get("simulation_params", params)

    tab_drivers, tab_teams, tab_races, tab_diagnostics, tab_model, tab_llm = st.tabs(
        ["Pilotos", "Constructores", "GP", "Diagnostico", "Modelo", "LLM"]
    )
    if results is None:
        with tab_drivers:
            st.info("Pulsa **Simular campeonato** para ver probabilidades.")
        with tab_teams:
            st.info("Pulsa **Simular campeonato** para ver constructores.")
        with tab_races:
            st.info("Pulsa **Simular campeonato** para ver carreras.")
        with tab_diagnostics:
            render_diagnostic_view(None, sim_drivers, sim_calendar, sim_params)
        driver_results = constructor_results = race_winners = None
    else:
        driver_results, constructor_results, race_winners = results
        with tab_drivers:
            render_driver_view(driver_results)
        with tab_teams:
            render_constructor_view(constructor_results)
        with tab_races:
            render_race_view(race_winners, sim_calendar, sim_drivers)
        with tab_diagnostics:
            render_diagnostic_view(driver_results, sim_drivers, sim_calendar, sim_params)

    with tab_model:
        render_model_view()
    with tab_llm:
        render_llm_view(driver_results, constructor_results, race_winners, sim_drivers, sim_calendar, llm_model, sim_params)


def main() -> None:
    """Streamlit entry point.

    Configures the page and launches the main application loop.
    Should be called directly from ``app.py``.
    """
    configure_page()
    render_app()
