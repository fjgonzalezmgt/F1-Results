"""LaTeX report generation for saved F1 analyses."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
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
from f1predictor.parameters import SimParams


def persist_llm_analysis(markdown: str) -> Path:
    """Save the latest LLM analysis as the persistent Markdown source."""
    LLM_ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LLM_ANALYSIS_PATH.write_text(markdown.strip() + "\n", encoding="utf-8")
    return LLM_ANALYSIS_PATH


def latest_llm_analysis() -> str:
    """Return the latest persisted LLM analysis, if one exists."""
    if not LLM_ANALYSIS_PATH.exists():
        return ""
    return LLM_ANALYSIS_PATH.read_text(encoding="utf-8").strip()


def persist_montecarlo_results(
    driver_results: pd.DataFrame,
    constructor_results: pd.DataFrame,
    race_winners: pd.DataFrame,
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    params: SimParams,
) -> Path:
    """Save the latest Monte Carlo outputs as a persistent Excel workbook."""
    MONTECARLO_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    params_df = pd.DataFrame(
        [{"parametro": key, "valor": value} for key, value in asdict(params).items()]
    )
    with pd.ExcelWriter(MONTECARLO_RESULTS_PATH, engine="openpyxl") as writer:
        driver_results.to_excel(writer, sheet_name="pilotos", index=False)
        constructor_results.to_excel(writer, sheet_name="constructores", index=False)
        race_winners.to_excel(writer, sheet_name="gp_probabilidades", index=False)
        drivers.to_excel(writer, sheet_name="inputs_pilotos", index=False)
        calendar.to_excel(writer, sheet_name="inputs_calendario", index=False)
        params_df.to_excel(writer, sheet_name="parametros", index=False)
    return MONTECARLO_RESULTS_PATH


def render_report(
    driver_results: pd.DataFrame,
    constructor_results: pd.DataFrame,
    race_winners: pd.DataFrame,
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    params: SimParams,
    markdown_analysis: str | None = None,
) -> Path:
    """Render the LaTeX report and compile it with pdflatex."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    analysis = (markdown_analysis or latest_llm_analysis()).strip()
    if not analysis:
        raise ValueError("No hay analisis LLM persistente para generar el reporte.")
    if not REPORT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"No existe la plantilla LaTeX: {REPORT_TEMPLATE_PATH}")

    next_round = int(calendar.loc[calendar["completed"] == 0, "round"].min()) if not calendar.loc[calendar["completed"] == 0].empty else int(calendar["round"].max())
    next_race = calendar.loc[calendar["round"] == next_round].iloc[0]
    next_race_winners = race_winners.loc[race_winners["round"] == next_round] if "round" in race_winners.columns else race_winners
    figures = save_report_figures(driver_results, constructor_results, next_race_winners)

    replacements = {
        "__REPORT_TITLE__": _latex_escape("Reporte F1 Championship Lab"),
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
            ["driver", "team", "win_pct"],
            ["Piloto", "Equipo", "Victoria %"],
        ),
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


def save_report_figures(
    driver_results: pd.DataFrame,
    constructor_results: pd.DataFrame,
    next_race_winners: pd.DataFrame,
) -> dict[str, Path]:
    """Generate report figures as PNG files under the project figs folder."""
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    driver_path = FIGS_DIR / "pilotos_campeonato.png"
    constructor_path = FIGS_DIR / "constructores_campeonato.png"
    race_path = FIGS_DIR / "proximo_gp.png"

    _save_bar_chart(
        driver_results.head(12).sort_values("champion_pct"),
        x="champion_pct",
        y="driver",
        color="team",
        title="Campeonato de pilotos",
        labels={"champion_pct": "Probabilidad de campeonato (%)", "driver": "Piloto"},
        path=driver_path,
    )
    _save_bar_chart(
        constructor_results.sort_values("champion_pct"),
        x="champion_pct",
        y="team",
        color="team",
        title="Campeonato de constructores",
        labels={"champion_pct": "Probabilidad de campeonato (%)", "team": "Equipo"},
        path=constructor_path,
    )
    _save_bar_chart(
        next_race_winners.head(12).sort_values("win_pct"),
        x="win_pct",
        y="driver",
        color="team",
        title="Proximo Gran Premio",
        labels={"win_pct": "Probabilidad de victoria (%)", "driver": "Piloto"},
        path=race_path,
    )
    return {"drivers": driver_path, "constructors": constructor_path, "next_race": race_path}


def markdown_to_latex(markdown: str) -> str:
    """Convert the app's simple Markdown analysis into basic LaTeX."""
    lines = markdown.splitlines()
    output: list[str] = []
    in_items = False

    def close_items() -> None:
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
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color if color in df.columns else None,
        orientation="h",
        text=df[x].map(lambda value: f"{value:.1f}%"),
        title=title,
        labels=labels,
        color_discrete_map=TEAM_COLORS,
    )
    fig.update_layout(
        width=1100,
        height=650,
        margin=dict(l=20, r=40, t=70, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=18),
        showlegend=False,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    try:
        fig.write_image(path, scale=2)
    except Exception as exc:
        raise RuntimeError(
            "No se pudieron exportar los graficos PNG. Revisa que `python-kaleido` "
            "este instalado por Conda y que Chrome este disponible para Kaleido."
        ) from exc


def _configure_kaleido_browser() -> None:
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
    relative = Path("..") / path.relative_to(path.parents[1])
    return "\n".join(
        [
            r"\begin{figure}[htbp]",
            r"\centering",
            fr"\includegraphics[width=0.95\textwidth]{{{relative.as_posix()}}}",
            fr"\caption{{{_latex_escape(caption)}}}",
            r"\end{figure}",
        ]
    )


def _table_to_latex(df: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    available = [column for column in columns if column in df.columns]
    header_map = dict(zip(columns, headers))
    if df.empty or not available:
        return r"\emph{Sin datos disponibles.}"

    alignment = "l" * len(available)
    lines = [fr"\begin{{longtable}}{{{alignment}}}", r"\toprule"]
    lines.append(" & ".join(_latex_escape(header_map[column]) for column in available) + r" \\")
    lines.extend([r"\midrule", r"\endhead"])
    for _, row in df[available].iterrows():
        values = [_format_cell(row[column]) for column in available]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def _key_value_table(rows: list[tuple[str, str]]) -> str:
    lines = [r"\begin{tabular}{ll}", r"\toprule", r"Campo & Valor \\", r"\midrule"]
    for key, value in rows:
        lines.append(f"{_latex_escape(key)} & {_latex_escape(value)}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return _latex_escape(f"{value:.2f}")
    return _latex_escape(str(value))


def _inline_markdown(text: str) -> str:
    escaped = _latex_escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", escaped)
    escaped = re.sub(r"`(.+?)`", r"\\texttt{\1}", escaped)
    return escaped


def _latex_escape(text: str) -> str:
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
