"""Streamlit UI for the F1 Championship Lab.

Builds the complete single-page Streamlit application: sidebar controls,
editable driver and calendar tables, data-update buttons (public APIs and
trusted F1 search via LLM), simulation trigger, result tabs and the LLM
narrative-analysis panel.
"""

from __future__ import annotations

import html
import json
import urllib.parse

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
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
from f1predictor.logging_utils import configure_logging, instrument_module_functions, logger
from f1predictor.parameters import SimParams
from f1predictor.public_apis import refresh_model_inputs_from_public_apis
from f1predictor.report import latest_llm_analysis, latest_montecarlo_results, persist_llm_analysis, persist_montecarlo_results, render_report
from f1predictor.simulator import build_driver_diagnostics, describe_driver, simulate_many


PLOTLY_CONFIG = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "f1_championship_lab",
        "height": 900,
        "width": 1400,
        "scale": 2,
    },
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "sendDataToCloud"],
}


def is_dark_theme() -> bool:
    """Return whether Streamlit is currently using a dark base theme."""
    base_theme = st.get_option("theme.base")
    return isinstance(base_theme, str) and base_theme.lower() == "dark"


def theme_tokens() -> dict[str, str]:
    """Get semantic color tokens used by charts and custom UI elements."""
    if is_dark_theme():
        return {
            "text": "#f3f4f6",
            "title": "#ffffff",
            "grid": "#263244",
            "legend_bg": "rgba(17,24,39,0.72)",
            "diag_line": "#94a3b8",
            "bar_neutral": "#9ca3af",
            "teal": "#2dd4bf",
            "heat_low": "#0f172a",
            "heat_mid": "#dc2626",
            "heat_high": "#f8fafc",
        }
    return {
        "text": "#111827",
        "title": "#111827",
        "grid": "#edf2f7",
        "legend_bg": "rgba(255,255,255,0.75)",
        "diag_line": "#9ca3af",
        "bar_neutral": "#111827",
        "teal": "#00a19c",
        "heat_low": "#f9fafb",
        "heat_mid": "#dc0000",
        "heat_high": "#111827",
    }


def configure_page() -> None:
    """Configure the Streamlit page and load environment variables.

    Sets the page title, icon and wide layout, loads ``.env`` variables
    and injects application-level CSS styles.
    """
    configure_logging()
    logger.info("Configurando pagina Streamlit")
    st.set_page_config(page_title=APP_TITLE, page_icon="F1", layout="wide")
    load_dotenv()
    inject_style()


def inject_style() -> None:
    """Inject compact CSS styles into the Streamlit app."""
    st.markdown(
        """
        <style>
        :root {
            --app-bg-1: #f8fafc;
            --app-bg-2: #ffffff;
            --app-bg-3: #f7f9fb;
            --app-shell-text: #111827;
            --sidebar-bg: #111827;
            --sidebar-text: #f8fafc;
            --sidebar-muted: #d1d5db;
            --sidebar-input-text: #111827;
            --sidebar-input-bg: #ffffff;
            --card-bg: rgba(255, 255, 255, 0.92);
            --card-border: #e5e7eb;
            --muted-text: #59636e;
            --kicker-text: #6b7280;
            --tab-bg: #ffffff;
            --tab-active-bg: #111827;
            --tab-active-text: #ffffff;
            --shadow-soft: 0 6px 18px rgba(17, 24, 39, 0.05);
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --app-bg-1: #0b1220;
                --app-bg-2: #0f172a;
                --app-bg-3: #111827;
                --app-shell-text: #e5e7eb;
                --sidebar-bg: #0b1220;
                --sidebar-text: #f8fafc;
                --sidebar-muted: #9ca3af;
                --sidebar-input-text: #f3f4f6;
                --sidebar-input-bg: #111827;
                --card-bg: rgba(17, 24, 39, 0.76);
                --card-border: #334155;
                --muted-text: #a8b1bd;
                --kicker-text: #cbd5e1;
                --tab-bg: #0f172a;
                --tab-active-bg: #dc0000;
                --tab-active-text: #ffffff;
                --shadow-soft: 0 8px 22px rgba(2, 6, 23, 0.45);
            }
        }
        .stApp {
            color: var(--app-shell-text);
            background:
                radial-gradient(circle at top left, rgba(220, 0, 0, 0.08), transparent 28rem),
                linear-gradient(180deg, var(--app-bg-1) 0%, var(--app-bg-2) 36%, var(--app-bg-3) 100%);
        }
        .block-container {padding-top: 1rem; padding-bottom: 2.4rem; max-width: 1380px;}
        section[data-testid="stSidebar"] {background: var(--sidebar-bg);}
        section[data-testid="stSidebar"] * {color: var(--sidebar-text);}
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {color: var(--sidebar-muted);}
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] [data-baseweb="select"] * {
            color: var(--sidebar-input-text);
            background: var(--sidebar-input-bg);
        }
        section[data-testid="stSidebar"] [data-baseweb="slider"] * {
            color: inherit;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--card-border);
            border-radius: 8px;
        }
        [data-testid="stMetric"] {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 13px 15px;
            box-shadow: var(--shadow-soft);
        }
        [data-testid="stMetricLabel"] {color: var(--muted-text);}
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 8px;
            min-height: 2.55rem;
            font-weight: 650;
        }
        div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {
            background: #dc0000;
            border-color: #dc0000;
        }
        .stTabs [data-baseweb="tab-list"] {gap: 0.35rem;}
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.65rem 0.95rem;
            background: var(--tab-bg);
            border: 1px solid var(--card-border);
        }
        .stTabs [aria-selected="true"] {
            background: var(--tab-active-bg);
            color: var(--tab-active-text);
        }
        h1, h2, h3 {letter-spacing: 0;}
        h1 {font-size: 2.25rem;}
        .source-line {color: var(--muted-text); font-size: 0.88rem;}
        .small-note {color: var(--muted-text); font-size: 0.92rem;}
        .app-kicker {
            color: var(--kicker-text);
            font-size: 0.95rem;
            margin-top: -0.5rem;
        }
        .status-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.8rem 0 1.1rem 0;
        }
        .status-pill {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 0.78rem 0.9rem;
            box-shadow: var(--shadow-soft);
        }
        .status-pill span {
            color: var(--kicker-text);
            display: block;
            font-size: 0.78rem;
            text-transform: uppercase;
        }
        .status-pill strong {color: var(--app-shell-text); font-size: 1.02rem;}
        .report-actions {
            margin: 0.5rem 0 1.0rem 0;
            padding: 0.9rem;
            border-radius: 10px;
            border: 1px solid var(--card-border);
            background: var(--card-bg);
            box-shadow: var(--shadow-soft);
        }
        .report-actions-title {
            font-size: 0.86rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--kicker-text);
            margin-bottom: 0.4rem;
        }
        @media (max-width: 900px) {
            .status-strip {grid-template-columns: repeat(2, minmax(0, 1fr));}
        }
        @media (max-width: 560px) {
            .block-container {padding-top: 0.65rem; padding-left: 1rem; padding-right: 1rem;}
            .status-strip {grid-template-columns: 1fr; gap: 0.5rem;}
            h1 {font-size: 1.85rem;}
        }
        .workflow-note {
            border-left: 4px solid #dc0000;
            padding: 0.65rem 0.85rem;
            margin: 0.8rem 0 1rem 0;
            color: var(--muted-text);
            background: var(--card-bg);
            border-radius: 0 8px 8px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status_strip(drivers: pd.DataFrame, calendar: pd.DataFrame, params: SimParams) -> None:
    """Render a compact app status strip above the main workflow."""
    completed = int(calendar["completed"].sum()) if "completed" in calendar.columns else 0
    pending = max(0, len(calendar) - completed)
    first_open = calendar.loc[calendar["completed"] == 0, "grand_prix"]
    next_gp = str(first_open.iloc[0]) if not first_open.empty else "Temporada completa"
    st.markdown(
        f"""
        <div class="status-strip">
            <div class="status-pill"><span>Pilotos</span><strong>{len(drivers)}</strong></div>
            <div class="status-pill"><span>Eventos pendientes</span><strong>{pending}</strong></div>
            <div class="status-pill"><span>Proximo GP</span><strong>{html.escape(next_gp)}</strong></div>
            <div class="status-pill"><span>Simulaciones</span><strong>{params.simulations:,}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_theme(fig: go.Figure, title: str | None = None, height: int = 500) -> go.Figure:
    """Apply the app's Plotly theme to a figure."""
    tokens = theme_tokens()
    fig.update_layout(
        title=dict(
            text=title,
            x=0.01,
            xanchor="left",
            y=0.98,
            yanchor="top",
            pad=dict(b=12),
            font=dict(size=18, color=tokens["title"]),
        ) if title else None,
        template="plotly_dark" if is_dark_theme() else "plotly_white",
        height=height,
        margin=dict(l=18, r=28, t=72 if title else 28, b=86),
        font=dict(family="Arial, sans-serif", size=13, color=tokens["text"]),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        transition=dict(duration=350, easing="cubic-in-out"),
        uirevision="f1-lab",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=tokens["grid"],
        zeroline=False,
        title_standoff=12,
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        title_standoff=12,
    )
    return fig


def render_plotly(fig: go.Figure, filename: str) -> None:
    """Render a Plotly chart with a consistent toolbar and image export name."""
    config = {**PLOTLY_CONFIG, "toImageButtonOptions": {**PLOTLY_CONFIG["toImageButtonOptions"], "filename": filename}}
    st.plotly_chart(fig, width="stretch", config=config)


def select_relevant_top(
    df: pd.DataFrame,
    probability_column: str,
    min_items: int = 5,
    max_items: int = 12,
    min_probability: float = 1.0,
    cumulative_target: float = 85.0,
) -> pd.DataFrame:
    """Return a relevance-based top-N slice for probability tables.

    Keeps enough rows to cover a strong share of cumulative probability,
    while enforcing practical minimum/maximum bounds for readability.
    """
    if df.empty:
        return df.copy()

    ranked = df.sort_values(probability_column, ascending=False).reset_index(drop=True)
    n_by_prob = int((ranked[probability_column] >= min_probability).sum())
    cumulative = ranked[probability_column].cumsum()
    n_by_cum = int((cumulative < cumulative_target).sum()) + 1

    top_n = max(1, min_items, n_by_prob, n_by_cum)
    top_n = min(max_items, top_n, len(ranked))
    return ranked.head(top_n)


def probability_bar(
    df: pd.DataFrame,
    probability_column: str,
    label_column: str,
    title: str,
    axis_label: str,
    filename: str,
    limit: int | None = None,
) -> None:
    """Render a polished horizontal probability bar chart."""
    chart_df = select_relevant_top(df, probability_column) if limit is None else df.sort_values(probability_column, ascending=False).head(limit)
    if chart_df.empty:
        st.info("Sin datos para mostrar en la grafica de probabilidades.")
        return

    chart_df = chart_df.sort_values(probability_column, ascending=False)
    fig = px.bar(
        chart_df,
        x=probability_column,
        y=label_column,
        color="team" if "team" in chart_df.columns else label_column,
        orientation="h",
        text=chart_df[probability_column].map(lambda value: f"{value:.1f}%"),
        labels={probability_column: axis_label, label_column: ""},
        color_discrete_map=TEAM_COLORS,
        hover_data=[column for column in ["team", "expected_points", "top3_pct", "top6_pct", "avg_final_rank"] if column in chart_df.columns],
    )
    apply_plotly_theme(fig, title=title, height=max(420, 38 * len(chart_df) + 160))
    fig.update_layout(showlegend=False)
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
        marker_line_width=0,
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=chart_df[label_column].tolist(),
        autorange="reversed",
    )
    fig.update_xaxes(range=[0, max(5, float(chart_df[probability_column].max()) * 1.18)])
    render_plotly(fig, filename)
    st.caption(f"Top {len(chart_df)} mostrado por relevancia (de {len(df)} total).")


def render_report_download(pdf_path=REPORT_PDF_PATH, key: str = "download_report_pdf") -> None:
    """Render a PDF download button when a report exists."""
    if not pdf_path.exists():
        return
    st.download_button(
        "Descargar reporte PDF",
        data=pdf_path.read_bytes(),
        file_name=pdf_path.name,
        mime="application/pdf",
        type="primary",
        width="stretch",
        key=key,
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
    iframe_html = f"""
        <style>
            :root {{
                --btn-border: #d0d7de;
                --btn-bg: #f6f8fa;
                --btn-text: #24292f;
                --status-text: #57606a;
            }}
            @media (prefers-color-scheme: dark) {{
                :root {{
                    --btn-border: #334155;
                    --btn-bg: #0f172a;
                    --btn-text: #e5e7eb;
                    --status-text: #94a3b8;
                }}
            }}
        </style>
        <div style="display:flex; justify-content:flex-end; margin:0 0 0.5rem 0;">
            <button
                id="{button_id}"
                type="button"
                style="border:1px solid var(--btn-border); border-radius:0.5rem; background:var(--btn-bg); color:var(--btn-text); padding:0.4rem 0.8rem; font-size:0.9rem; cursor:pointer;"
            >
                {html.escape("Copiar analisis")}
            </button>
            <span id="{status_id}" style="margin-left:0.5rem; font-size:0.85rem; color:var(--status-text);"></span>
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
        """
    iframe_src = "data:text/html;charset=utf-8," + urllib.parse.quote(iframe_html)
    st.iframe(
        iframe_src,
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
    logger.info("Cargando pilotos base")
    return clean_drivers(load_drivers())


@st.cache_data(show_spinner=False)
def load_calendar_cached() -> pd.DataFrame:
    """Load and clean default calendar data with Streamlit result caching.

    Returns
    -------
    pd.DataFrame
        Cleaned calendar seed table, cached across reruns.
    """
    logger.info("Cargando calendario base")
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
    logger.info("Preparando simulacion cacheada")
    drivers = clean_drivers(dataframe_from_csv_text(drivers_csv))
    calendar = clean_calendar(dataframe_from_csv_text(calendar_csv))
    return simulate_many(drivers, calendar, params)



def render_sidebar(calendar: pd.DataFrame) -> tuple[SimParams, int, int]:
    """Render the sidebar with model controls and return user selections.

    Parameters
    ----------
    calendar : pd.DataFrame
        Cleaned calendar table used to populate the starting-round selector.

    Returns
    -------
    tuple[SimParams, int, int]
        * **params** – ``SimParams`` built from the sidebar widgets.
        * **api_season** – Season year selected for API data fetching.
        * **history_seasons** – Number of historical seasons used as priors.
    """
    st.sidebar.header("Configuracion")
    st.sidebar.caption("Los valores predeterminados ofrecen un escenario equilibrado. Ajusta solo lo que necesites.")
    simulations = st.sidebar.slider("Simulaciones", 500, 50000, 4000, step=500, help="Numero de iteraciones Monte Carlo. Mas iteraciones mejoran estabilidad pero tardan mas.")
    open_rounds = calendar.loc[calendar["completed"] == 0, "round"].tolist()
    default_round = int(open_rounds[0]) if open_rounds else int(calendar["round"].max())
    start_round = st.sidebar.selectbox(
        "Simular desde ronda",
        options=[int(v) for v in calendar["round"].tolist()],
        index=[int(v) for v in calendar["round"].tolist()].index(default_round),
        help="Ronda inicial para proyectar el resto de la temporada.",
    )
    include_sprints = st.sidebar.toggle("Incluir sprints pendientes", value=True, help="Si esta activo, suma puntos esperados de sprint en las rondas restantes.")

    with st.sidebar.expander("Ajustes avanzados del modelo", expanded=False):
        seed = st.number_input("Semilla", min_value=1, value=DEFAULT_SEASON, step=1, help="Valor para reproducir exactamente los mismos resultados aleatorios.")
        st.markdown("**Pesos**")
        driver_weight = st.slider("Piloto", 0.10, 0.80, 0.42, step=0.02, help="Peso del talento y ejecucion del piloto en el ritmo base.")
        constructor_weight = st.slider("Constructor", 0.10, 0.80, 0.48, step=0.02, help="Peso del rendimiento del auto/equipo en el ritmo base.")
        form_weight = st.slider("Forma reciente", 0.00, 0.30, 0.06, step=0.02, help="Impacto de resultados recientes sobre el rendimiento esperado.")
        form_decay_races = st.slider("Duracion forma (carreras)", 0.5, 8.0, 2.5, step=0.5, help="Cantidad de carreras en las que se diluye el efecto de forma reciente.")
        qualifying_weight = st.slider("Peso de clasificacion", 0.20, 1.20, 0.72, step=0.02, help="Importancia de la clasificacion en el resultado final de carrera.")
        st.markdown("**Incertidumbre**")
        chaos = st.slider("Caos carrera", 1.0, 14.0, 5.5, step=0.5, help="Ruido global de carrera: incidentes, estrategia, variabilidad y azar.")
        reliability_multiplier = st.slider("Riesgo fiabilidad", 0.4, 2.4, 1.0, step=0.1, help="Escala de abandonos y problemas mecanicos.")
        weather_multiplier = st.slider("Clima", 0.3, 2.5, 1.0, step=0.1, help="Escala del impacto de clima sobre el orden esperado.")
        safety_car_multiplier = st.slider("Safety car", 0.3, 2.5, 1.0, step=0.1, help="Escala del impacto de neutralizaciones y reinicios.")
        development_drift = st.slider("Desarrollo por equipo", 0.0, 8.0, 2.4, step=0.2, help="Magnitud de mejoras/regresiones de rendimiento entre equipos.")
        team_uncertainty = st.slider("Incertidumbre auto", 0.0, 10.0, 6.0, step=0.5, help="Varianza persistente del paquete tecnico por equipo.")

    with st.sidebar.expander("Fuentes de datos", expanded=False):
        api_season = st.number_input("Temporada", min_value=2023, max_value=2100, value=DEFAULT_SEASON, step=1, help="Temporada para consultar standings y resultados externos.")
        history_seasons = st.number_input("Temporadas historicas", min_value=0, max_value=5, value=3, step=1, help="Numero de temporadas previas usadas para construir priors.")
        st.caption(f"LLM: {default_model()} · API key {'disponible' if api_key_available() else 'no configurada'}")

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
    return params, int(api_season), int(history_seasons)


def _render_data_editors_content(
    default_drivers: pd.DataFrame,
    default_calendar: pd.DataFrame,
    api_season: int,
    history_seasons: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Render editable driver and calendar tables with data-update buttons.

    Manages the ``drivers_df`` and ``calendar_df`` keys in
    ``st.session_state``.  Provides buttons to refresh from public APIs
    and from trusted F1 sources via LLM deep web search.

    Parameters
    ----------
    default_drivers : pd.DataFrame
        Fallback driver table used when session state is empty.
    default_calendar : pd.DataFrame
        Fallback calendar table used when session state is empty.
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
            if st.button("Busqueda profunda F1 con LLM", help="Usa OpenAI web_search en fuentes confiables de F1 para standings, pilotos y calendario; guarda CSV en disco."):
                try:
                    with st.spinner("Buscando informacion F1 en fuentes confiables..."):
                        payload = call_llm_formula1_official_update(
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
                        "title": "Busqueda profunda F1 via LLM web_search",
                        "summary": [
                            f"{len(payload.get('drivers', pd.DataFrame()))} pilotos oficiales recibidos.",
                            f"{len(payload.get('calendar', pd.DataFrame()))} eventos de calendario recibidos.",
                        ],
                        "sources": sources["source"].dropna().tolist() if not sources.empty and "source" in sources.columns else [],
                        "errors": [],
                    }
                    st.success(f"Datos oficiales guardados en {DRIVERS_PATH.name} y {CALENDAR_PATH.name}.")
                except Exception as exc:
                    st.error(f"No se pudo actualizar con busqueda profunda F1: {exc}")
        else:
            st.caption("Agrega OPENAI_API_KEY para busqueda profunda F1 via LLM.")

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


def render_data_editors(
    default_drivers: pd.DataFrame,
    default_calendar: pd.DataFrame,
    api_season: int,
    history_seasons: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep detailed data preparation available without dominating the main flow."""
    with st.expander("1 · Revisar o actualizar datos (opcional)", expanded=False):
        st.caption("Actualiza fuentes o edita ratings y calendario. Si no haces cambios, se usan los datos guardados.")
        return _render_data_editors_content(default_drivers, default_calendar, api_season, history_seasons)


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

    probability_bar(
        driver_results,
        probability_column="champion_pct",
        label_column="driver",
        title="Probabilidad de campeonato de pilotos",
        axis_label="Probabilidad de campeonato (%)",
        filename="pilotos_campeonato",
        limit=12,
    )

    chart_tab, points_tab = st.tabs(["Top 6", "Puntos"])
    tokens = theme_tokens()
    with chart_tab:
        top_probability = select_relevant_top(driver_results, "champion_pct", min_items=6, max_items=10).copy()
        probability_long = top_probability.melt(
            id_vars=["driver", "team"],
            value_vars=["champion_pct", "top3_pct", "top6_pct"],
            var_name="bucket",
            value_name="probability",
        )
        probability_long["bucket"] = probability_long["bucket"].map(
            {
                "champion_pct": "Campeon",
                "top3_pct": "Top 3",
                "top6_pct": "Top 6",
            }
        )
        fig = px.bar(
            probability_long,
            x="driver",
            y="probability",
            color="bucket",
            barmode="group",
            text=probability_long["probability"].map(lambda value: f"{value:.0f}%"),
            labels={"driver": "", "probability": "Probabilidad (%)", "bucket": ""},
            color_discrete_sequence=["#dc0000", tokens["bar_neutral"], tokens["teal"]],
            hover_data={"team": True, "probability": ":.2f", "bucket": True, "driver": False},
        )
        apply_plotly_theme(fig, title="Amenaza real: titulo, podio y zona fuerte", height=470)
        fig.update_layout(hovermode="x unified")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(range=[0, max(10, float(probability_long["probability"].max()) * 1.18)])
        render_plotly(fig, "pilotos_top6")
    with points_tab:
        fig = px.scatter(
            driver_results,
            x="current_points",
            y="expected_points",
            size="champion_pct",
            color="team",
            hover_name="driver",
            text="code",
            labels={
                "current_points": "Puntos actuales",
                "expected_points": "Puntos esperados al final",
                "champion_pct": "Campeon %",
            },
            color_discrete_map=TEAM_COLORS,
            size_max=34,
        )
        apply_plotly_theme(fig, title="Puntos actuales vs puntos proyectados", height=520)
        max_points = float(max(driver_results["current_points"].max(), driver_results["expected_points"].max()))
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_points * 1.05,
            y1=max_points * 1.05,
            line=dict(color=tokens["diag_line"], dash="dash"),
        )
        fig.update_traces(textposition="top center")
        render_plotly(fig, "pilotos_puntos")
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

    probability_bar(
        constructor_results,
        probability_column="champion_pct",
        label_column="team",
        title="Probabilidad de campeonato de constructores",
        axis_label="Probabilidad de campeonato (%)",
        filename="constructores_campeonato",
    )

    top_constructor = select_relevant_top(constructor_results, "champion_pct", min_items=4, max_items=8)
    constructor_long = top_constructor.melt(
        id_vars=["team"],
        value_vars=["champion_pct", "top3_pct"],
        var_name="bucket",
        value_name="probability",
    )
    constructor_long["bucket"] = constructor_long["bucket"].map({"champion_pct": "Campeon", "top3_pct": "Top 3"})
    fig = px.bar(
        constructor_long,
        x="team",
        y="probability",
        color="bucket",
        barmode="group",
        text=constructor_long["probability"].map(lambda value: f"{value:.0f}%"),
        labels={"team": "", "probability": "Probabilidad (%)", "bucket": ""},
        color_discrete_sequence=["#dc0000", theme_tokens()["bar_neutral"]],
        hover_data={"probability": ":.2f", "bucket": True, "team": False},
    )
    apply_plotly_theme(fig, title="Constructor: titulo vs top 3", height=460)
    fig.update_layout(hovermode="x unified")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(range=[0, max(10, float(constructor_long["probability"].max()) * 1.18)])
    render_plotly(fig, "constructores_top3")
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
    selected = st.selectbox(
        "Gran Premio",
        options=list(race_labels),
        format_func=lambda value: race_labels[value],
        help="Selecciona la ronda para ver probabilidades de victoria y el detalle de esa carrera.",
    )
    race_probs = race_winners.loc[race_winners["round"] == selected].copy()
    if race_probs.empty:
        st.warning("No hay probabilidades de ganador para esta ronda. Revisa que se haya simulado desde una ronda anterior o igual.")
    else:
        probability_bar(
            race_probs,
            probability_column="win_pct",
            label_column="driver",
            title=f"Probabilidad de victoria: {race_labels[selected]}",
            axis_label="Probabilidad de victoria (%)",
            filename="gp_probabilidad_victoria",
        )
        st.dataframe(race_probs.round(2), width="stretch", hide_index=True)

    heatmap_df = race_winners.copy()
    tokens = theme_tokens()
    if not heatmap_df.empty:
        top_codes = (
            heatmap_df.groupby(["code", "driver"], as_index=False)["win_pct"].sum()
            .sort_values("win_pct", ascending=False)
            .head(10)["code"]
            .tolist()
        )
        heatmap_df = heatmap_df.loc[heatmap_df["code"].isin(top_codes)]
        heatmap_df["race_label"] = heatmap_df["round"].astype(int).astype(str) + " - " + heatmap_df["grand_prix"].astype(str)
        pivot = heatmap_df.pivot_table(index="driver", columns="race_label", values="win_pct", aggfunc="sum", fill_value=0)
        if not pivot.empty:
            fig = px.imshow(
                pivot,
                aspect="auto",
                color_continuous_scale=[tokens["heat_low"], "#fca5a5", tokens["heat_mid"], tokens["heat_high"]],
                labels=dict(x="", y="", color="Victoria %"),
                text_auto=".0f",
            )
            apply_plotly_theme(fig, title="Mapa de victorias probables por GP", height=max(420, 32 * len(pivot) + 160))
            fig.update_layout(coloraxis_colorbar=dict(title="Victoria %"))
            fig.update_xaxes(tickangle=-35)
            render_plotly(fig, "gp_mapa_victorias")

    driver_code = st.selectbox(
        "Perfil de piloto",
        options=drivers["code"].tolist(),
        format_func=lambda code: f"{code} - {drivers.loc[drivers['code'] == code, 'driver'].iloc[0]}",
        help="Muestra un resumen rapido de atributos del piloto seleccionado.",
    )
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
    apply_plotly_theme(fig, title="Mapa de fuerza: auto vs piloto", height=500)
    render_plotly(fig, "diagnostico_auto_piloto")


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
    """
    st.subheader("LLM en la ecuacion")
    if not api_key_available():
        st.warning("No detecte OPENAI_API_KEY en el entorno. Revisa el archivo .env en esta carpeta.")
        return

    open_rounds = calendar.loc[calendar["completed"] == 0, "round"].tolist()
    selected_round = int(open_rounds[0]) if open_rounds else int(calendar["round"].max())
    if st.button("Buscar contexto del proximo GP", help="Consulta fuentes web para clima, sanciones, upgrades y contexto competitivo del siguiente GP."):
        try:
            with st.spinner("Buscando clima, parrilla, upgrades y sanciones..."):
                st.session_state["llm_notes"] = call_llm_context_search(drivers, calendar, selected_round)
        except Exception as exc:
            st.error(f"No se pudo buscar contexto: {exc}")

    notes = st.text_area(
        "Contexto cualitativo",
        key="llm_notes",
        height=180,
        placeholder="Ejemplo: Miami tiene lluvia probable; Antonelli sale en pole; McLaren trajo upgrades; Hadjar tiene penalizacion.",
        help="Notas adicionales que quieres forzar en el analisis LLM (insights, rumores, condiciones de carrera).",
    )

    if st.button("Generar analisis LLM", type="primary", help="Genera una interpretacion narrativa usando los resultados simulados y el contexto cualitativo."):
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
                    answer = call_llm_analysis(payload).strip()
                    persist_llm_analysis(answer)
                    st.session_state["llm_answer"] = answer
                    st.session_state["llm_analysis_path"] = str(LLM_ANALYSIS_PATH)
            except Exception as exc:
                st.error(f"No se pudo generar analisis: {exc}")

    if st.session_state.get("llm_answer"):
        answer = st.session_state["llm_answer"]
        render_copy_button(answer, "analysis")
        st.markdown(answer)

    st.caption("El analisis generado se incluye automaticamente al guardar el reporte desde la barra superior.")


def render_report_actions_bar(
    driver_results: pd.DataFrame | None,
    constructor_results: pd.DataFrame | None,
    race_winners: pd.DataFrame | None,
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    params: SimParams,
) -> None:
    """Render prominent report save/download actions near the top of the app."""
    report_ready = (
        driver_results is not None
        and constructor_results is not None
        and race_winners is not None
    )

    st.markdown('<div class="report-actions"><div class="report-actions-title">Reporte</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.25, 1.15, 2.6])
    with c1:
        if st.button(
            "Guardar reporte",
            type="primary",
            disabled=not report_ready,
            width="stretch",
            key="save_report_top",
            help="Compila un reporte final en Markdown, TeX y PDF con tablas, graficas y analisis.",
        ):
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
            except Exception as exc:
                st.error(f"No se pudo guardar el reporte: {exc}")

    with c2:
        render_report_download(key="download_report_pdf_top")

    with c3:
        if report_ready:
            st.caption("Listo para generar reporte. Puedes guardarlo o descargar el ultimo PDF disponible.")
        else:
            st.caption("Ejecuta la simulacion para habilitar el reporte. El analisis LLM es opcional.")
        if REPORT_PDF_PATH.exists():
            st.caption(f"Ultimo PDF: {REPORT_PDF_PATH.name}")

    st.markdown("</div>", unsafe_allow_html=True)


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

    params, api_season, history_seasons = render_sidebar(default_calendar)

    st.title(APP_TITLE)
    st.markdown(f'<div class="app-kicker">{html.escape(APP_CAPTION)}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="workflow-note"><strong>Flujo recomendado:</strong> revisa los datos solo si hace falta, ejecuta la simulacion y explora los resultados. Los ajustes avanzados estan en la barra lateral.</div>',
        unsafe_allow_html=True,
    )

    drivers, calendar = render_data_editors(default_drivers, default_calendar, api_season, history_seasons)
    render_status_strip(drivers, calendar, params)

    st.markdown("### 2 · Ejecutar escenario")
    if st.button("Simular campeonato", type="primary", width="stretch", help="Ejecuta Monte Carlo con los parametros actuales y actualiza todas las probabilidades."):
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

    if results is None:
        driver_results = constructor_results = race_winners = None
    else:
        driver_results, constructor_results, race_winners = results

    render_report_actions_bar(driver_results, constructor_results, race_winners, sim_drivers, sim_calendar, sim_params)

    st.markdown("### 3 · Explorar resultados")
    tab_drivers, tab_teams, tab_races, tab_diagnostics, tab_model, tab_llm = st.tabs(
        ["Campeonato · Pilotos", "Campeonato · Equipos", "Proximos GP", "Por que da este resultado", "Metodologia", "Analisis con IA"]
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
    else:
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
        render_llm_view(driver_results, constructor_results, race_winners, sim_drivers, sim_calendar, sim_params)


def main() -> None:
    """Streamlit entry point.

    Configures the page and launches the main application loop.
    Should be called directly from ``app.py``.
    """
    configure_page()
    render_app()


instrument_module_functions(
    __name__,
    skip=("load_drivers_cached", "load_calendar_cached", "run_simulation_cached"),
)
