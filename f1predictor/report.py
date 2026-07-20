"""LaTeX report generation for saved F1 analyses."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

from f1predictor.config import (
    FIGS_DIR,
    LLM_ANALYSIS_PATH,
    MONTECARLO_RESULTS_PATH,
    REPORT_DIR,
    REPORT_PDF_PATH,
    REPORT_TEMPLATE_PATH,
    REPORT_TEX_PATH,
    TEAM_COLORS,
)
from f1predictor.logging_utils import instrument_module_functions
from f1predictor.parameters import SimParams


def persist_llm_analysis(markdown: str) -> Path:
    """Save the latest LLM analysis as the persistent Markdown source.

    Parameters
    ----------
    markdown : str
        Markdown-formatted analysis text produced by the LLM.

    Returns
    -------
    Path
        Absolute path to the written ``analisis_llm.md`` file.
    """
    LLM_ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LLM_ANALYSIS_PATH.write_text(markdown.strip() + "\n", encoding="utf-8")
    return LLM_ANALYSIS_PATH


def latest_llm_analysis() -> str:
    """Return the latest persisted LLM analysis, if one exists.

    Returns
    -------
    str
        Markdown text read from ``analisis_llm.md``, or an empty string
        if the file does not exist.
    """
    if not LLM_ANALYSIS_PATH.exists():
        return ""
    return LLM_ANALYSIS_PATH.read_text(encoding="utf-8").strip()


def latest_montecarlo_results() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None
):
    """Return the latest persisted Monte Carlo results, if they exist.

    Returns
    -------
    tuple or None
        ``(driver_results, constructor_results, race_winners, drivers, calendar)``
        read from the saved Excel workbook, or ``None`` if no file exists or
        reading fails.
    """
    if not MONTECARLO_RESULTS_PATH.exists():
        return None
    try:
        driver_results = pd.read_excel(MONTECARLO_RESULTS_PATH, sheet_name="pilotos")
        constructor_results = pd.read_excel(MONTECARLO_RESULTS_PATH, sheet_name="constructores")
        race_winners = pd.read_excel(MONTECARLO_RESULTS_PATH, sheet_name="gp_probabilidades")
        drivers = pd.read_excel(MONTECARLO_RESULTS_PATH, sheet_name="inputs_pilotos")
        calendar = pd.read_excel(MONTECARLO_RESULTS_PATH, sheet_name="inputs_calendario")
        return driver_results, constructor_results, race_winners, drivers, calendar
    except Exception:
        return None


def latest_f1_predict_data() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Return the last saved official/fallback catalogue and answer sheet."""
    if not MONTECARLO_RESULTS_PATH.exists():
        return None
    try:
        questions = pd.read_excel(MONTECARLO_RESULTS_PATH, sheet_name="f1_predict_preguntas")
        answers = pd.read_excel(MONTECARLO_RESULTS_PATH, sheet_name="f1_predict_respuestas")
        return questions, answers
    except Exception:
        return None


def persist_montecarlo_results(
    driver_results: pd.DataFrame,
    constructor_results: pd.DataFrame,
    race_winners: pd.DataFrame,
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    params: SimParams,
    question_catalog: pd.DataFrame | None = None,
    game_answers: pd.DataFrame | None = None,
) -> Path:
    """Save the latest Monte Carlo outputs as a persistent Excel workbook.

    Writes six sheets: ``pilotos``, ``constructores``, ``gp_probabilidades``,
    ``inputs_pilotos``, ``inputs_calendario`` and ``parametros``.

    Parameters
    ----------
    driver_results : pd.DataFrame
        Driver championship summary from ``simulate_many``.
    constructor_results : pd.DataFrame
        Constructor championship summary from ``simulate_many``.
    race_winners : pd.DataFrame
        Per-race winner probabilities from ``simulate_many``.
    drivers : pd.DataFrame
        Cleaned driver seed table used as simulation input.
    calendar : pd.DataFrame
        Cleaned calendar table used as simulation input.
    params : SimParams
        Simulation parameters saved in the ``parametros`` sheet.

    Returns
    -------
    Path
        Absolute path to the written ``resultados_montecarlo.xlsx`` file.
    """
    MONTECARLO_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    params_df = pd.DataFrame(
        [{"parametro": key, "valor": value} for key, value in asdict(params).items()]
    )
    excel_races = race_winners.copy()
    for column in ("head_to_head_pct", "qualifying_head_to_head_pct", "team_head_to_head_pct"):
        if column in excel_races.columns:
            excel_races[column] = excel_races[column].map(
                lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
            )
    with pd.ExcelWriter(MONTECARLO_RESULTS_PATH, engine="openpyxl") as writer:
        driver_results.to_excel(writer, sheet_name="pilotos", index=False)
        constructor_results.to_excel(writer, sheet_name="constructores", index=False)
        excel_races.to_excel(writer, sheet_name="gp_probabilidades", index=False)
        drivers.to_excel(writer, sheet_name="inputs_pilotos", index=False)
        calendar.to_excel(writer, sheet_name="inputs_calendario", index=False)
        params_df.to_excel(writer, sheet_name="parametros", index=False)
        if question_catalog is not None and not question_catalog.empty:
            question_catalog.to_excel(writer, sheet_name="f1_predict_preguntas", index=False)
        if game_answers is not None and not game_answers.empty:
            game_answers.to_excel(writer, sheet_name="f1_predict_respuestas", index=False)
    return MONTECARLO_RESULTS_PATH


def render_report(
    driver_results: pd.DataFrame,
    constructor_results: pd.DataFrame,
    race_winners: pd.DataFrame,
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    params: SimParams,
    markdown_analysis: str | None = None,
    question_catalog: pd.DataFrame | None = None,
    game_answers: pd.DataFrame | None = None,
) -> Path:
    """Render the LaTeX report and compile it with pdflatex.

    Fills every ``__PLACEHOLDER__`` in the LaTeX template with formatted
    tables, figures and the LLM analysis, writes the ``.tex`` source and
    runs ``pdflatex`` to produce the final PDF.

    Parameters
    ----------
    driver_results : pd.DataFrame
        Driver championship summary from ``simulate_many``.
    constructor_results : pd.DataFrame
        Constructor championship summary from ``simulate_many``.
    race_winners : pd.DataFrame
        Per-race winner probabilities from ``simulate_many``.
    drivers : pd.DataFrame
        Cleaned driver seed table used as simulation input.
    calendar : pd.DataFrame
        Cleaned calendar table used to determine the next Grand Prix.
    params : SimParams
        Simulation parameters shown in the parameters table.
    markdown_analysis : str or None, optional
        LLM narrative analysis to embed; falls back to the persisted
        ``analisis_llm.md`` when ``None``.

    Returns
    -------
    Path
        Absolute path to the compiled ``reporte_f1.pdf``.

    Raises
    ------
    ValueError
        If no LLM analysis is available (neither argument nor persisted file).
    FileNotFoundError
        If the LaTeX template does not exist at ``REPORT_TEMPLATE_PATH``.
    RuntimeError
        If ``pdflatex`` exits with a non-zero return code.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    analysis = latest_llm_analysis().strip() if markdown_analysis is None else markdown_analysis.strip()
    if not analysis:
        analysis = (
            "## Lectura automatica\n\n"
            "La hoja de respuestas anterior proviene directamente del Monte Carlo. "
            "No se agrego contexto cualitativo LLM; revisa clima, sanciones y parrilla antes del cierre."
        )
    if not REPORT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"No existe la plantilla LaTeX: {REPORT_TEMPLATE_PATH}")

    next_round = int(calendar.loc[calendar["completed"] == 0, "round"].min()) if not calendar.loc[calendar["completed"] == 0].empty else int(calendar["round"].max())
    next_race = calendar.loc[calendar["round"] == next_round].iloc[0]
    next_race_winners = race_winners.loc[race_winners["round"] == next_round] if "round" in race_winners.columns else race_winners
    figures = save_report_figures(driver_results, constructor_results, next_race_winners)

    replacements = {
        "__REPORT_TITLE__": _latex_escape("F1 Predict Decision Lab"),
        "__REPORT_SUBTITLE__": _latex_escape("10 respuestas, Monte Carlo, contexto LLM y valor esperado"),
        "__BRAND_NAME__": _latex_escape("Gamer Insight Analytics"),
        "__BRAND_URL__": _latex_escape("gamerinsightanalytics.com"),
        "__GENERATED_AT__": _latex_escape(datetime.now().strftime("%Y-%m-%d %H:%M")),
        "__NEXT_GP__": _latex_escape(f"R{next_round} - {next_race['grand_prix']}"),
        "__SIMULATION_SUMMARY__": _simulation_summary(driver_results, constructor_results, next_race_winners, params),
        "__DRIVER_TABLE__": _table_to_latex(
            driver_results.head(10),
            ["driver", "team", "expected_points", "champion_pct", "top3_pct", "avg_final_rank"],
            ["Piloto", "Equipo", "Pts esp.", "Campeon %", "Top 3 %", "Rank medio"],
        ),
        "__CONSTRUCTOR_TABLE__": _table_to_latex(
            constructor_results,
            ["team", "expected_points", "champion_pct", "top3_pct"],
            ["Equipo", "Pts esp.", "Campeon %", "Top 3 %"],
        ),
        "__NEXT_RACE_TABLE__": _table_to_latex(
            next_race_winners.head(10),
            ["driver", "team", "win_pct", "pole_pct", "podium_pct", "fastest_lap_pct"],
            ["Piloto", "Equipo", "Victoria %", "Pole %", "Podio %", "V. rapida %"],
        ),
        "__F1_PREDICT_TABLE__": _game_answers_table(game_answers),
        "__F1_PREDICT_RULES__": _f1_predict_rules_note(question_catalog),
        "__PARAMETER_TABLE__": _params_table(params),
        "__LLM_ANALYSIS__": markdown_to_latex(analysis),
        "__DATA_NOTES__": _data_notes(drivers, calendar),
        "__DRIVER_CHART__": _figure_latex(figures["drivers"], "Probabilidad de campeonato de pilotos"),
        "__CONSTRUCTOR_CHART__": _figure_latex(figures["constructors"], "Probabilidad de campeonato de constructores"),
        "__NEXT_RACE_CHART__": _figure_latex(figures["next_race"], "Probabilidad de victoria del proximo GP"),
    }

    tex = REPORT_TEMPLATE_PATH.read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        tex = tex.replace(placeholder, value)
    REPORT_TEX_PATH.write_text(tex, encoding="utf-8")

    completed = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", REPORT_TEX_PATH.name],
        cwd=REPORT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        log_tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-20:])
        raise RuntimeError(f"pdflatex no pudo compilar el reporte:\n{log_tail}")
    return REPORT_PDF_PATH


def _game_answers_table(game_answers: pd.DataFrame | None) -> str:
    if game_answers is None or game_answers.empty:
        return _key_value_table([("Estado", "No hay preguntas cargadas para esta ronda")])
    table = game_answers.copy()
    table["probability_display"] = table["model_probability_pct"].map(
        lambda value: "--" if pd.isna(value) else f"{float(value):.1f}%"
    )
    table["expected_display"] = table["expected_game_points"].map(
        lambda value: "--" if pd.isna(value) else f"{float(value):.2f}"
    )
    lines = [
        r"\footnotesize",
        r"\rowcolors{2}{GIASoft}{white}",
        r"\begin{longtable}{@{}p{0.05\linewidth}p{0.34\linewidth}p{0.35\linewidth}p{0.09\linewidth}p{0.09\linewidth}@{}}",
        r"\rowcolor{GIADark}",
        r"\textcolor{white}{\textbf{\#}} & \textcolor{white}{\textbf{Pregunta}} & \textcolor{white}{\textbf{Respuesta}} & \textcolor{white}{\textbf{Prob.}} & \textcolor{white}{\textbf{V. esp.}} \\",
        r"\endhead",
    ]
    for _, row in table.iterrows():
        values = [
            _format_cell(row.get("question_id", "")),
            _format_cell(row.get("question", "")),
            _format_cell(row.get("recommended_answer", "")),
            _format_cell(row.get("probability_display", "")),
            _format_cell(row.get("expected_display", "")),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\end{longtable}", r"\rowcolors{2}{}{}", r"\normalsize"])
    return "\n".join(lines)


def _f1_predict_rules_note(question_catalog: pd.DataFrame | None) -> str:
    official = bool(
        question_catalog is not None
        and not question_catalog.empty
        and "is_official" in question_catalog.columns
        and question_catalog["is_official"].all()
    )
    rows = [
        ("Formato", "10 preguntas por ronda con opciones cerradas"),
        ("Puntuacion", "Variable por opcion; el modelo prioriza valor esperado cuando hay puntos"),
        ("Podio", "Da puntos por piloto en top 3 y duplica al acertar la posicion exacta"),
        ("Pole", "Piloto mas rapido de Q3, aunque luego tenga sancion de parrilla"),
        ("Catalogo", "Oficial extraido de F1 Predict" if official else "Fallback analitico; confirmar contra F1 Predict"),
    ]
    return _key_value_table(rows)


def save_report_figures(
    driver_results: pd.DataFrame,
    constructor_results: pd.DataFrame,
    next_race_winners: pd.DataFrame,
) -> dict[str, Path]:
    """Generate report figures as PNG files under the project figs folder.

    Produces three horizontal bar charts: driver championship probabilities,
    constructor championship probabilities and next-race win probabilities.

    Parameters
    ----------
    driver_results : pd.DataFrame
        Driver championship summary from ``simulate_many``.
    constructor_results : pd.DataFrame
        Constructor championship summary from ``simulate_many``.
    next_race_winners : pd.DataFrame
        Winner probabilities for the next upcoming Grand Prix.

    Returns
    -------
    dict[str, Path]
        Mapping with keys ``"drivers"``, ``"constructors"`` and
        ``"next_race"`` pointing to the written PNG paths.
    """
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    driver_path = FIGS_DIR / "pilotos_campeonato.png"
    constructor_path = FIGS_DIR / "constructores_campeonato.png"
    race_path = FIGS_DIR / "proximo_gp.png"

    _save_bar_chart(
        _probability_chart_data(driver_results, "champion_pct", limit=12, exclude_zero=True),
        x="champion_pct",
        y="driver",
        color="team",
        title="Campeonato de pilotos",
        labels={"champion_pct": "Probabilidad de campeonato (%)", "driver": "Piloto"},
        path=driver_path,
    )
    _save_bar_chart(
        _probability_chart_data(constructor_results, "champion_pct", exclude_zero=True),
        x="champion_pct",
        y="team",
        color="team",
        title="Campeonato de constructores",
        labels={"champion_pct": "Probabilidad de campeonato (%)", "team": "Equipo"},
        path=constructor_path,
    )
    _save_bar_chart(
        _probability_chart_data(next_race_winners, "win_pct", limit=12, exclude_zero=True),
        x="win_pct",
        y="driver",
        color="team",
        title="Proximo Gran Premio",
        labels={"win_pct": "Probabilidad de victoria (%)", "driver": "Piloto"},
        path=race_path,
    )
    return {"drivers": driver_path, "constructors": constructor_path, "next_race": race_path}


def _probability_chart_data(
    df: pd.DataFrame,
    probability_column: str,
    limit: int | None = None,
    exclude_zero: bool = False,
) -> pd.DataFrame:
    """Prepare probability chart rows in descending probability order.

    Parameters
    ----------
    df : pd.DataFrame
        Source probability table.
    probability_column : str
        Numeric column used for sorting and optional zero filtering.
    limit : int or None, optional
        Maximum number of rows to keep after sorting.
    exclude_zero : bool, optional
        Whether to remove rows whose probability is zero.

    Returns
    -------
    pd.DataFrame
        Filtered and sorted chart table. If the input is empty or the
        probability column is absent, the original DataFrame is returned.
    """
    if df.empty or probability_column not in df.columns:
        return df
    chart_df = df.copy()
    if exclude_zero:
        chart_df = chart_df.loc[chart_df[probability_column] > 0]
    chart_df = chart_df.sort_values(probability_column, ascending=False)
    if limit is not None:
        chart_df = chart_df.head(limit)
    return chart_df


def markdown_to_latex(markdown: str) -> str:
    """Convert the app's simple Markdown analysis into basic LaTeX.

    Handles headings (``#`` through ``####``), unordered bullet lists
    (``-`` or ``*``), inline bold (``**text**``) and inline code
    (`` `text` ``).  Horizontal rules and blank lines are preserved as
    paragraph breaks.

    Parameters
    ----------
    markdown : str
        Markdown-formatted text, typically the LLM narrative analysis.

    Returns
    -------
    str
        LaTeX source string suitable for direct inclusion in a ``.tex`` file.
    """
    lines = markdown.splitlines()
    output: list[str] = []
    in_items = False

    def close_items() -> None:
        """Close the active LaTeX itemize block when one is open."""
        nonlocal in_items
        if in_items:
            output.append(r"\end{itemize}")
            in_items = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            close_items()
            output.append("")
            continue
        if set(line) <= {"-"} and len(line) >= 3:
            close_items()
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            close_items()
            level = len(heading.group(1))
            text = _inline_markdown(heading.group(2))
            command = {1: "section", 2: "subsection", 3: "subsubsection"}.get(level, "paragraph")
            output.append(fr"\{command}{{{text}}}")
            continue
        bullet = re.match(r"^[-*]\s+(.+)$", line)
        if bullet:
            if not in_items:
                output.append(r"\begin{itemize}")
                in_items = True
            output.append(rf"\item {_inline_markdown(bullet.group(1))}")
            continue
        close_items()
        output.append(_inline_markdown(line) + r"\par")

    close_items()
    return "\n".join(output)


def _simulation_summary(
    driver_results: pd.DataFrame,
    constructor_results: pd.DataFrame,
    next_race_winners: pd.DataFrame,
    params: SimParams,
) -> str:
    """Build a key-value LaTeX table summarising the simulation highlights.

    Parameters
    ----------
    driver_results : pd.DataFrame
        Driver championship summary; uses the first row as the favourite.
    constructor_results : pd.DataFrame
        Constructor championship summary; uses the first row as the leader.
    next_race_winners : pd.DataFrame
        Winner probabilities for the next Grand Prix.
    params : SimParams
        Simulation parameters; exposes the iteration count.

    Returns
    -------
    str
        LaTeX key-value table string produced by ``_key_value_table``.
    """
    driver = driver_results.iloc[0]
    constructor = constructor_results.iloc[0]
    race = next_race_winners.iloc[0] if not next_race_winners.empty else None
    rows = [
        ("Simulaciones", f"{params.simulations:,}"),
        ("Favorito pilotos", f"{driver['driver']} ({driver['champion_pct']:.2f}%)"),
        ("Favorito constructores", f"{constructor['team']} ({constructor['champion_pct']:.2f}%)"),
    ]
    if race is not None:
        rows.append(("Favorito proximo GP", f"{race['driver']} ({race['win_pct']:.2f}%)"))
    return _key_value_table(rows)


def _params_table(params: SimParams) -> str:
    """Build a key-value LaTeX table of all simulation parameters.

    Parameters
    ----------
    params : SimParams
        Frozen simulation parameters dataclass.

    Returns
    -------
    str
        LaTeX key-value table string produced by ``_key_value_table``.
    """
    labels = {
        "simulations": "Simulaciones",
        "seed": "Semilla",
        "start_round": "Ronda inicial",
        "include_sprints": "Sprints pendientes",
        "driver_weight": "Peso piloto",
        "constructor_weight": "Peso constructor",
        "form_weight": "Peso forma",
        "form_decay_races": "Decay forma",
        "qualifying_weight": "Peso qualy",
        "chaos": "Caos",
        "reliability_multiplier": "Fiabilidad",
        "weather_multiplier": "Clima",
        "safety_car_multiplier": "Safety car",
        "development_drift": "Desarrollo",
        "team_uncertainty": "Incertidumbre auto",
    }
    rows = [(labels.get(key, key), str(value)) for key, value in asdict(params).items()]
    return _key_value_table(rows)


def _data_notes(drivers: pd.DataFrame, calendar: pd.DataFrame) -> str:
    """Build a key-value LaTeX table with counts from the input data.

    Parameters
    ----------
    drivers : pd.DataFrame
        Cleaned driver table used to report the number of active drivers.
    calendar : pd.DataFrame
        Cleaned calendar table used to report completed and pending events.

    Returns
    -------
    str
        LaTeX key-value table string produced by ``_key_value_table``.
    """
    completed = int(calendar["completed"].sum()) if "completed" in calendar.columns else 0
    total = len(calendar)
    rows = [
        ("Pilotos en input", str(len(drivers))),
        ("Eventos calendario", str(total)),
        ("Eventos completados", str(completed)),
        ("Eventos pendientes", str(total - completed)),
    ]
    return _key_value_table(rows)


def _save_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
    labels: dict[str, str],
    path: Path,
) -> None:
    """Render a horizontal Plotly bar chart and export it as a PNG file.

    Parameters
    ----------
    df : pd.DataFrame
        Source data for the chart.
    x : str
        Column name mapped to the x-axis (numeric values).
    y : str
        Column name mapped to the y-axis (category labels).
    color : str
        Column name used to colour-code bars; matched against ``TEAM_COLORS``.
    title : str
        Chart title displayed above the figure.
    labels : dict[str, str]
        Axis label overrides passed to ``plotly.express.bar``.
    path : Path
        Destination path for the exported PNG file.

    Raises
    ------
    RuntimeError
        If ``plotly`` is not installed or ``pdflatex`` / Kaleido cannot
        export the PNG image.
    """
    if df.empty:
        return
    try:
        import plotly.express as px
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "No se pudieron crear los graficos del reporte porque falta plotly. "
            "Instala el entorno con `conda env update -f environment.yml --prune`."
        ) from exc

    _configure_kaleido_browser()
    chart_df = df.copy()
    chart_df[y] = chart_df[y].astype(str)
    y_order = chart_df[y].tolist()
    left_margin = _chart_left_margin(chart_df[y])
    chart_height = max(520, 48 * len(chart_df) + 170)
    fig = px.bar(
        chart_df,
        x=x,
        y=y,
        color=color if color in df.columns else None,
        orientation="h",
        text=chart_df[x].map(lambda value: f"{value:.1f}%"),
        title=title,
        labels=labels,
        color_discrete_map=TEAM_COLORS,
    )
    fig.update_layout(
        width=1400,
        height=chart_height,
        margin=dict(l=left_margin, r=130, t=95, b=95),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", size=22, color="#24292f"),
        title=dict(x=0.5, xanchor="center", font=dict(size=28)),
        showlegend=False,
        bargap=0.26,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#eaeef2",
        zeroline=False,
        ticks="outside",
        title_font=dict(size=20),
        tickfont=dict(size=18),
    )
    fig.update_yaxes(
        automargin=True,
        autorange="reversed",
        categoryorder="array",
        categoryarray=y_order,
        title_standoff=18,
        title_font=dict(size=20),
        tickfont=dict(size=18),
    )
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=18, color="#24292f"),
        marker_line_width=0,
    )
    try:
        fig.write_image(path, scale=2)
    except Exception as exc:
        raise RuntimeError(
            "No se pudieron exportar los graficos PNG. Revisa que `python-kaleido` "
            "este instalado por Conda y que Chrome este disponible para Kaleido."
        ) from exc


def _chart_left_margin(labels: pd.Series) -> int:
    """Compute a left margin in pixels proportional to the longest label.

    Parameters
    ----------
    labels : pd.Series
        Category labels displayed on the y-axis of the chart.

    Returns
    -------
    int
        Left margin in pixels, clamped to [230, 380].
    """
    longest = max((len(str(value)) for value in labels), default=12)
    return min(380, max(230, longest * 12 + 70))


def _configure_kaleido_browser() -> None:
    """Set the ``BROWSER_PATH`` environment variable for Kaleido PNG export.

    Searches common Windows installation paths for ``chrome.exe`` and sets
    ``BROWSER_PATH`` when found.  Does nothing if the variable is already
    set or if Chrome cannot be located.
    """
    if os.getenv("BROWSER_PATH"):
        return
    chrome_candidates = [
        Path(os.environ.get("PROGRAMFILES", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
    ]
    for candidate in chrome_candidates:
        if candidate.exists():
            os.environ["BROWSER_PATH"] = str(candidate)
            return


def _figure_latex(path: Path, caption: str) -> str:
    """Build a LaTeX ``figure`` environment for a PNG chart.

    Parameters
    ----------
    path : Path
        Absolute path to the PNG file; converted to a relative path from
        the report directory for the LaTeX ``\\includegraphics`` command.
    caption : str
        Figure caption; special LaTeX characters are escaped.

    Returns
    -------
    str
        LaTeX ``figure`` block as a multi-line string.
    """
    relative = Path("..") / path.relative_to(path.parents[1])
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            fr"\includegraphics[width=\linewidth]{{{relative.as_posix()}}}",
            fr"\caption{{{_latex_escape(caption)}}}",
            r"\end{figure}",
        ]
    )


def _table_to_latex(df: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    """Render a DataFrame subset as a styled LaTeX ``longtable``.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame; columns not present in ``df`` are silently skipped.
    columns : list[str]
        Ordered list of column names to include.
    headers : list[str]
        Display headers corresponding to each entry in ``columns``.

    Returns
    -------
    str
        LaTeX ``longtable`` source with alternating row colours and a dark
        header row, or an ``\\emph{Sin datos disponibles.}`` fallback when
        ``df`` is empty.
    """
    available = [column for column in columns if column in df.columns]
    header_map = dict(zip(columns, headers))
    if df.empty or not available:
        return r"\emph{Sin datos disponibles.}"

    alignment = "l" * len(available)
    lines = [
        r"\small",
        r"\rowcolors{2}{GIASoft}{white}",
        fr"\begin{{longtable}}{{@{{}}{alignment}@{{}}}}",
        r"\rowcolor{GIADark}",
    ]
    lines.append(
        " & ".join(r"\textcolor{white}{\textbf{" + _latex_escape(header_map[column]) + "}}" for column in available)
        + r" \\"
    )
    lines.extend([r"\endhead"])
    for _, row in df[available].iterrows():
        values = [_format_cell(row[column]) for column in available]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\end{longtable}", r"\rowcolors{2}{}{}", r"\normalsize"])
    return "\n".join(lines)


def _key_value_table(rows: list[tuple[str, str]]) -> str:
    """Render a list of key-value pairs as a two-column LaTeX table.

    Parameters
    ----------
    rows : list[tuple[str, str]]
        Sequence of ``(key, value)`` string pairs to display.

    Returns
    -------
    str
        LaTeX ``tabular`` source with a dark header row and alternating
        row colours.
    """
    lines = [
        r"\begin{center}",
        r"\rowcolors{2}{GIASoft}{white}",
        r"\begin{tabular}{@{}p{0.34\linewidth}p{0.46\linewidth}@{}}",
        r"\rowcolor{GIADark}\textcolor{white}{\textbf{Campo}} & \textcolor{white}{\textbf{Valor}} \\",
    ]
    for key, value in rows:
        lines.append(f"{_latex_escape(key)} & {_latex_escape(value)}" + r" \\")
    lines.extend([r"\end{tabular}", r"\rowcolors{2}{}{}", r"\end{center}"])
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    """Format a single table cell value as a LaTeX-escaped string.

    Parameters
    ----------
    value : object
        Cell value from a DataFrame; floats are formatted with two decimal
        places and NaN/None renders as an empty string.

    Returns
    -------
    str
        LaTeX-escaped string representation of ``value``.
    """
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return _latex_escape(f"{value:.2f}")
    return _latex_escape(str(value))


def _inline_markdown(text: str) -> str:
    """Convert inline Markdown bold and code spans to LaTeX equivalents.

    Escapes special LaTeX characters first, then replaces ``**text**``
    with ``\\textbf{text}`` and `` `text` `` with ``\\texttt{text}``.

    Parameters
    ----------
    text : str
        Raw Markdown text that may contain inline formatting.

    Returns
    -------
    str
        LaTeX-safe string with inline formatting commands applied.
    """
    escaped = _latex_escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", escaped)
    escaped = re.sub(r"`(.+?)`", r"\\texttt{\1}", escaped)
    return escaped


def _latex_escape(text: str) -> str:
    """Escape all special LaTeX characters in a plain-text string.

    Replaces ``\\``, ``&``, ``%``, ``$``, ``#``, ``_``, ``{``, ``}``,
    ``~`` and ``^`` with their safe LaTeX command equivalents.

    Parameters
    ----------
    text : str
        Plain text that may contain LaTeX special characters.

    Returns
    -------
    str
        String safe to embed directly in LaTeX source.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


instrument_module_functions(__name__)
