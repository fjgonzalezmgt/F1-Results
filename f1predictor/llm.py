"""OpenAI integration for official F1 inputs, context and narrative analysis.

Provides helpers that call the OpenAI Responses API with web-search tools
to fetch official Formula1.com data, search race-weekend context and generate
qualitative championship analyses.  All network calls require the
``OPENAI_API_KEY`` environment variable to be set.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Any

import pandas as pd

from f1predictor.config import DRIVER_RATING_COLUMNS
from f1predictor.logging_utils import instrument_module_functions, logger


LLM_INSTRUCTIONS = (
    "Eres analista cuantitativo de Formula 1. Usa solo el contexto entregado. "
    "Separa lo que viene del Monte Carlo de lo que viene de noticias o hipotesis. "
    "No inventes datos, lesiones, sanciones, parrillas ni clima. Responde en espanol "
    "claro, con bullets cortos, una tesis firme y advertencias accionables."
)


def _extract_response_text(response: Any) -> str:
    """Extract visible text from an OpenAI Responses API object.

    Parameters
    ----------
    response : Any
        Response object returned by the OpenAI Responses API.

    Returns
    -------
    str
        Concatenated text content, or an empty string if no text is found.
    """
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    fragments: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text_value = getattr(content, "text", None)
            if isinstance(text_value, str) and text_value.strip():
                fragments.append(text_value.strip())
    return "\n\n".join(fragments).strip()


def _extract_web_search_sources(response: Any) -> list[str]:
    """Extract source URLs returned by a Responses API web_search call.

    Parameters
    ----------
    response : Any
        Response object returned by the OpenAI Responses API.

    Returns
    -------
    list[str]
        Deduplicated list of source URLs found in the response output.
    """
    urls: list[str] = []
    for item in getattr(response, "output", []) or []:
        action = item.get("action") if isinstance(item, dict) else getattr(item, "action", None)
        if action is None:
            continue
        sources = action.get("sources") if isinstance(action, dict) else getattr(action, "sources", None)
        for source in sources or []:
            url = source.get("url") if isinstance(source, dict) else getattr(source, "url", None)
            if isinstance(url, str) and url and url not in urls:
                urls.append(url)
    return urls


def api_key_available() -> bool:
    """Return True when the OPENAI_API_KEY environment variable is set.

    Returns
    -------
    bool
        ``True`` if the key is non-empty, ``False`` otherwise.
    """
    return bool(os.getenv("OPENAI_API_KEY"))


def default_model() -> str:
    """Return the configured OpenAI model name.

    Reads the ``OPENAI_MODEL`` environment variable; falls back to
    ``"gpt-5.5"`` when the variable is unset.

    Returns
    -------
    str
        Model identifier string to pass to the OpenAI API.
    """
    return os.getenv("OPENAI_MODEL", "gpt-5.5")


def _extract_json(text: str) -> Any:
    """Extract the first valid JSON object or array from model output text.

    Tries direct parsing, then extracts from a markdown code fence, then
    performs a broad regex search for the first ``{...}`` or ``[...]`` block.

    Parameters
    ----------
    text : str
        Raw text output from an LLM response.

    Returns
    -------
    Any
        Parsed JSON value (dict or list).

    Raises
    ------
    ValueError
        If no valid JSON can be found in ``text``.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    match = re.search(r"(\{[\s\S]+\}|\[[\s\S]+\])", text)
    if match:
        return json.loads(match.group(1).strip())
    raise ValueError("La respuesta del LLM no contiene JSON valido.")


def call_llm_formula1_official_update(
    model: str,
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    season: int,
) -> dict[str, pd.DataFrame]:
    """Use LLM web search restricted to Formula1.com for official inputs.

    Constructs a structured prompt with the current driver/calendar seeds,
    calls the OpenAI Responses API with a web-search tool filtered to
    ``formula1.com``, and returns the parsed JSON as DataFrames.

    Parameters
    ----------
    model : str
        OpenAI model identifier (e.g. ``"gpt-5.5"``).
    drivers : pd.DataFrame
        Current driver seed table used to guide the LLM.
    calendar : pd.DataFrame
        Current calendar seed table used to guide the LLM.
    season : int
        Target F1 season year (e.g. ``2026``).

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys ``"drivers"``, ``"calendar"`` and
        ``"sources"`` containing the parsed update DataFrames.

    Raises
    ------
    ValueError
        If the LLM response cannot be parsed as a JSON object.
    """
    from openai import OpenAI

    logger.info("Consultando Formula1.com via LLM: modelo={}, temporada={}", model, season)
    client = OpenAI()
    driver_seed = drivers[["driver", "code", "team"]].to_dict(orient="records")
    calendar_seed = calendar[["round", "grand_prix", "country", "circuit", "race_date"]].to_dict(orient="records")
    schema = {
        "drivers": [
            {
                "driver": "Kimi Antonelli",
                "code": "ANT",
                "team": "Mercedes",
                "nationality": "ITA",
                "current_points": 75,
            }
        ],
        "calendar": [
            {
                "round": 4,
                "grand_prix": "Miami GP",
                "country": "USA",
                "circuit": "Miami International Autodrome",
                "race_date": f"{season}-05-03",
                "sprint_remaining": 0,
                "completed": 0,
            }
        ],
        "sources": [
            f"https://www.formula1.com/en/results/{season}/drivers",
            f"https://www.formula1.com/en/racing/{season}",
        ],
    }
    prompt = (
        f"Fecha de consulta: {date.today().isoformat()}.\n"
        f"Temporada objetivo: {season}.\n"
        f"Pilotos esperados del CSV: {json.dumps(driver_seed, ensure_ascii=False)}\n"
        f"Calendario actual del CSV: {json.dumps(calendar_seed, ensure_ascii=False)}\n\n"
        "Busca solamente en Formula1.com la informacion oficial actual para la temporada: "
        "line-up de pilotos/equipos, standings de pilotos y calendario. Devuelve SOLO JSON "
        "valido con este esquema exacto, sin markdown ni texto adicional:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        "Reglas: conserva el code de tres letras del CSV cuando lo puedas mapear; "
        "current_points debe venir del standing oficial; completed debe ser 1 solo si "
        "Formula1.com muestra resultado para esa ronda; sprint_remaining debe ser 1 si "
        "el evento tiene sprint y todavia no esta completado. No inventes ratings."
    )

    response = client.responses.create(
        model=model,
        tools=[
            {
                "type": "web_search",
                "search_context_size": "high",
                "filters": {
                    "allowed_domains": [
                        "formula1.com",
                        "www.formula1.com",
                    ]
                },
                "user_location": {
                    "type": "approximate",
                    "country": "US",
                    "timezone": "America/Guatemala",
                },
            }
        ],
        tool_choice="required",
        include=["web_search_call.action.sources"],
        instructions=(
            "Eres un extractor de datos oficiales de Formula 1. Usa web_search restringido "
            "a Formula1.com. Responde solamente JSON valido con las claves drivers, calendar "
            "y sources. No agregues explicaciones."
        ),
        input=prompt,
    )
    data = _extract_json(_extract_response_text(response))
    if not isinstance(data, dict):
        raise ValueError("La respuesta oficial de Formula1.com no fue un objeto JSON.")

    sources = list(data.get("sources", [])) if isinstance(data.get("sources", []), list) else []
    for source in _extract_web_search_sources(response):
        if source not in sources:
            sources.append(source)

    return {
        "drivers": pd.DataFrame(data.get("drivers", [])),
        "calendar": pd.DataFrame(data.get("calendar", [])),
        "sources": pd.DataFrame({"source": sources}),
    }


def call_llm_context_search(model: str, drivers: pd.DataFrame, calendar: pd.DataFrame, round_no: int) -> str:
    """Search recent context for a selected Grand Prix.

    Queries the OpenAI Responses API with a web-search tool to find
    current F1 news relevant to the specified race: weather forecasts,
    grid penalties, upgrades, tyre performance and safety-car history.

    Parameters
    ----------
    model : str
        OpenAI model identifier.
    drivers : pd.DataFrame
        Current driver standings used to contextualise the prompt.
    calendar : pd.DataFrame
        Calendar table used to identify the selected race.
    round_no : int
        Round number of the target Grand Prix.

    Returns
    -------
    str
        Formatted bullet-point analysis with cited sources.
    """
    from openai import OpenAI

    logger.info("Buscando contexto LLM: modelo={}, ronda={}", model, round_no)
    client = OpenAI()
    race = calendar.loc[calendar["round"] == round_no].iloc[0].to_dict()
    standings = drivers[["driver", "code", "team", "current_points", "recent_form"]].to_dict(orient="records")
    prompt = (
        f"Fecha de consulta: {date.today().isoformat()}.\n"
        f"Gran Premio seleccionado: {json.dumps(race, ensure_ascii=False)}\n"
        f"Standings/model seed: {json.dumps(standings, ensure_ascii=False)}\n\n"
        "Busca noticias recientes y verificables que puedan cambiar una prediccion F1: "
        "clima, parrilla, sanciones, cambios de motor, actualizaciones aerodinamicas, "
        "ritmo de tanda larga, declaraciones tecnicas, accidentes, safety car esperado "
        "y rendimiento de neumaticos. Organiza por tema. Cada bullet debe incluir fuente "
        "o URL visible. Si no hay dato confiable, dilo."
    )
    response = client.responses.create(
        model=model,
        tools=[
            {
                "type": "web_search",
                "search_context_size": "medium",
                "user_location": {
                    "type": "approximate",
                    "country": "US",
                    "timezone": "America/Guatemala",
                },
            }
        ],
        tool_choice="required",
        include=["web_search_call.action.sources"],
        instructions=(
            "Eres analista F1 actualizado. Usa busqueda web antes de responder. "
            "No inventes. Responde en espanol con bullets cortos y fuentes visibles."
        ),
        input=prompt,
    )
    return _extract_response_text(response)


def build_analysis_payload(
    driver_results: pd.DataFrame,
    constructor_results: pd.DataFrame,
    race_winners: pd.DataFrame,
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    notes: str,
    diagnostics: pd.DataFrame | None = None,
    scenarios: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Create a compact payload for LLM championship interpretation.

    Assembles the top simulation results, upcoming calendar rows, driver
    seed ratings and qualitative context notes into a single dictionary
    ready to be serialised and sent to the LLM.

    Parameters
    ----------
    driver_results : pd.DataFrame
        Driver championship summary from ``simulate_many``.
    constructor_results : pd.DataFrame
        Constructor championship summary from ``simulate_many``.
    race_winners : pd.DataFrame
        Per-race winner probabilities from ``simulate_many``.
    drivers : pd.DataFrame
        Cleaned driver seed table with current ratings.
    calendar : pd.DataFrame
        Cleaned calendar table; only the next six incomplete rounds are
        included.
    notes : str
        Free-text qualitative context (weather, penalties, upgrades, etc.).

    Returns
    -------
    dict[str, Any]
        Serialisable payload with keys for championship top-12,
        constructors, next-race winner probabilities, driver seed, upcoming
        calendar and a fixed ``request`` instruction string.
    """
    next_races = calendar.loc[calendar["completed"] == 0].head(6)
    payload = {
        "driver_championship_top_12": driver_results.head(12).round(2).to_dict(orient="records"),
        "constructor_championship": constructor_results.round(2).to_dict(orient="records"),
        "next_race_winner_probabilities": race_winners.head(15).round(2).to_dict(orient="records"),
        "driver_seed": drivers[["driver", "code", "team", "current_points", *DRIVER_RATING_COLUMNS]].to_dict(orient="records"),
        "upcoming_calendar": next_races.to_dict(orient="records"),
        "qualitative_context": notes,
        "request": (
            "Entrega un veredicto final sobre campeonato de pilotos, constructores y proximo GP. "
            "Incluye favoritos, amenazas, escenarios base/conservador/agresivo y donde el modelo "
            "podria estar ciego por datos cualitativos."
        ),
    }
    if diagnostics is not None and not diagnostics.empty:
        payload["model_diagnostics"] = diagnostics.head(12).round(2).to_dict(orient="records")
    if scenarios is not None and not scenarios.empty:
        payload["scenario_summary"] = scenarios.round(2).to_dict(orient="records")
    return payload


def call_llm_analysis(model: str, payload: dict[str, Any]) -> str:
    """Call OpenAI to generate a narrative championship analysis.

    Parameters
    ----------
    model : str
        OpenAI model identifier.
    payload : dict[str, Any]
        Serialisable analysis payload, typically built with
        ``build_analysis_payload``.

    Returns
    -------
    str
        Narrative analysis in Spanish with championship verdict, favourites,
        scenario breakdowns and model blind-spot warnings.
    """
    from openai import OpenAI

    logger.info("Generando analisis LLM: modelo={}, payload_keys={}", model, list(payload))
    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=LLM_INSTRUCTIONS,
        input=json.dumps(payload, ensure_ascii=True),
    )
    return _extract_response_text(response)


instrument_module_functions(__name__)
