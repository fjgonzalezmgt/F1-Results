"""OpenAI integration for official F1 inputs, context and narrative analysis.

Provides helpers that call the OpenAI Responses API with web-search tools
to fetch official/trusted F1 data, search race-weekend context and generate
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
from f1predictor.game import QUESTION_COLUMNS, normalize_question_catalog
from f1predictor.logging_utils import instrument_module_functions, logger


TRUSTED_F1_DEEP_SEARCH_DOMAINS = [
    "f1predict.formula1.com",
    "formula1.com",
    "www.formula1.com",
    "fia.com",
    "www.fia.com",
    "motorsport.com",
    "www.motorsport.com",
    "autosport.com",
    "www.autosport.com",
    "racefans.net",
    "www.racefans.net",
    "the-race.com",
    "www.the-race.com",
    "bbc.com",
    "www.bbc.com",
    "skysports.com",
    "www.skysports.com",
    "espn.com",
    "www.espn.com",
]

TRUSTED_F1_CONTEXT_SEARCH_DOMAINS = [
    *TRUSTED_F1_DEEP_SEARCH_DOMAINS,
    "weather.com",
    "www.weather.com",
    "accuweather.com",
    "www.accuweather.com",
    "metoffice.gov.uk",
    "www.metoffice.gov.uk",
    "meteoblue.com",
    "www.meteoblue.com",
    "pirelli.com",
    "www.pirelli.com",
    "ferrari.com",
    "www.ferrari.com",
    "mclaren.com",
    "www.mclaren.com",
    "mercedesamgf1.com",
    "www.mercedesamgf1.com",
    "redbullracing.com",
    "www.redbullracing.com",
    "astonmartinf1.com",
    "www.astonmartinf1.com",
    "williamsf1.com",
    "www.williamsf1.com",
    "alpinecars.com",
    "www.alpinecars.com",
    "haasf1team.com",
    "www.haasf1team.com",
    "racingbulls.com",
    "www.racingbulls.com",
    "sauber-group.com",
    "www.sauber-group.com",
    "audi.com",
    "www.audi.com",
    "cadillac.com",
    "www.cadillac.com",
]

LLM_INSTRUCTIONS = (
    "Eres analista cuantitativo de Formula 1 especializado en F1 Predict. Usa solo el contexto entregado. "
    "Separa lo que viene del Monte Carlo de lo que viene de noticias o hipotesis. "
    "No inventes datos, lesiones, sanciones, parrillas ni clima. Responde en espanol "
    "claro. Abre con una hoja de respuestas numerada para las preguntas de F1 Predict, "
    "indicando confianza, valor esperado cuando exista puntuacion y cualquier pregunta "
    "que requiera criterio manual. Despues resume riesgos y, de forma secundaria, el campeonato."
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
    """Return the OpenAI model used by every LLM action.

    The model is intentionally fixed so searches, updates and narrative
    analyses cannot diverge through environment or UI configuration.

    Returns
    -------
    str
        Model identifier string to pass to the OpenAI API.
    """
    return "gpt-5.6-luna"


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
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    season: int,
) -> dict[str, pd.DataFrame]:
    """Use deep LLM web search across trusted F1 sources for official inputs.

    Constructs a structured prompt with the current driver/calendar seeds,
    calls the OpenAI Responses API with a high-context web-search tool
    filtered to trusted F1 domains, and returns the parsed JSON as DataFrames.

    Parameters
    ----------
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

    model = default_model()
    logger.info("Consultando fuentes F1 confiables via LLM: modelo={}, temporada={}", model, season)
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
            "https://www.fia.com/",
        ],
    }
    prompt = (
        f"Fecha de consulta: {date.today().isoformat()}.\n"
        f"Temporada objetivo: {season}.\n"
        f"Pilotos esperados del CSV: {json.dumps(driver_seed, ensure_ascii=False)}\n"
        f"Calendario actual del CSV: {json.dumps(calendar_seed, ensure_ascii=False)}\n\n"
        "Haz una busqueda profunda en fuentes confiables de Formula 1 para la informacion "
        "actual de la temporada: line-up de pilotos/equipos, standings de pilotos y calendario. "
        "Prioriza Formula1.com y FIA para datos oficiales; usa medios especializados confiables "
        "para confirmar o resolver datos faltantes/ambiguos. Devuelve SOLO JSON "
        "valido con este esquema exacto, sin markdown ni texto adicional:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        "Reglas: conserva el code de tres letras del CSV cuando lo puedas mapear; "
        "current_points debe venir del standing oficial; completed debe ser 1 solo si "
        "hay resultado publicado para esa ronda; sprint_remaining debe ser 1 si "
        "el evento tiene sprint y todavia no esta completado. No inventes ratings. "
        "Incluye en sources las URLs concretas usadas y evita fuentes sin fecha, blogs anonimos, "
        "foros o redes sociales. Si las fuentes discrepan, usa Formula1.com/FIA y deja la fuente "
        "principal en sources."
    )

    response = client.responses.create(
        model=model,
        tools=[
            {
                "type": "web_search",
                "search_context_size": "high",
                "filters": {
                    "allowed_domains": TRUSTED_F1_DEEP_SEARCH_DOMAINS,
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
            "Eres un extractor de datos F1 actualizado. Usa busqueda web profunda en fuentes "
            "confiables, priorizando Formula1.com y FIA para datos oficiales y usando medios "
            "especializados solo como verificacion. Responde solamente JSON valido con las "
            "claves drivers, calendar y sources. No agregues explicaciones."
        ),
        input=prompt,
    )
    data = _extract_json(_extract_response_text(response))
    if not isinstance(data, dict):
        raise ValueError("La respuesta de fuentes F1 confiables no fue un objeto JSON.")

    sources = list(data.get("sources", [])) if isinstance(data.get("sources", []), list) else []
    for source in _extract_web_search_sources(response):
        if source not in sources:
            sources.append(source)

    return {
        "drivers": pd.DataFrame(data.get("drivers", [])),
        "calendar": pd.DataFrame(data.get("calendar", [])),
        "sources": pd.DataFrame({"source": sources}),
    }


def call_llm_context_search(
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    round_no: int,
    questions: pd.DataFrame | None = None,
) -> str:
    """Search broad recent context for a selected Grand Prix.

    Queries the OpenAI Responses API with a high-context web-search tool
    across trusted F1, team, tyre and weather sources to find current race
    context: weather forecasts, grid penalties, upgrades, tyre performance
    and safety-car history.

    Parameters
    ----------
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

    model = default_model()
    logger.info("Buscando contexto LLM: modelo={}, ronda={}", model, round_no)
    client = OpenAI()
    race = calendar.loc[calendar["round"] == round_no].iloc[0].to_dict()
    standings = drivers[["driver", "code", "team", "current_points", "recent_form"]].to_dict(orient="records")
    question_rows = []
    if questions is not None and not questions.empty:
        question_rows = (
            questions[["question_id", "question", "question_type"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
    prompt = (
        f"Fecha de consulta: {date.today().isoformat()}.\n"
        f"Gran Premio seleccionado: {json.dumps(race, ensure_ascii=False)}\n"
        f"Standings/model seed: {json.dumps(standings, ensure_ascii=False)}\n\n"
        f"Preguntas F1 Predict que deben guiar la busqueda: {json.dumps(question_rows, ensure_ascii=False)}\n\n"
        "Haz una busqueda amplia y verificable que pueda cambiar una prediccion F1: "
        "clima local del circuito, probabilidad de lluvia/viento/temperatura, parrilla, "
        "sanciones, cambios de motor o caja, actualizaciones aerodinamicas, ritmo de tanda "
        "larga, declaraciones tecnicas, accidentes recientes, safety car esperado, compuestos "
        "Pirelli y degradacion de neumaticos. Cruza fuentes cuando sea posible: F1/FIA para "
        "datos oficiales, Pirelli para neumaticos, meteorologia reconocida para clima, equipos "
        "para upgrades y medios especializados para analisis. Organiza por tema. Cada bullet "
        "debe incluir fuente o URL visible. Prioriza evidencia que ayude a escoger entre las opciones "
        "de las preguntas F1 Predict entregadas; no inventes preguntas ni respuestas. Si no hay dato "
        "confiable, dilo."
    )
    response = client.responses.create(
        model=model,
        tools=[
            {
                "type": "web_search",
                "search_context_size": "high",
                "filters": {
                    "allowed_domains": TRUSTED_F1_CONTEXT_SEARCH_DOMAINS,
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
            "Eres analista F1 actualizado. Usa busqueda web amplia en fuentes confiables "
            "antes de responder. Prioriza datos oficiales y fuentes primarias; usa medios "
            "especializados como verificacion. No inventes. Responde en espanol con bullets "
            "cortos, accionables y fuentes visibles."
        ),
        input=prompt,
    )
    return _extract_response_text(response)


def call_llm_f1_predict_questions(
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    round_no: int,
) -> dict[str, pd.DataFrame]:
    """Discover the current round's ten official F1 Predict questions and options.

    The game is a JavaScript application and its question set changes by round,
    so this search is intentionally separate from the static rules.  Missing or
    inaccessible questions are returned as an empty catalogue rather than being
    invented.
    """
    from openai import OpenAI

    race_rows = calendar.loc[calendar["round"] == round_no]
    if race_rows.empty:
        raise ValueError(f"No existe la ronda {round_no} en el calendario.")
    race = race_rows.iloc[0].to_dict()
    entrants = drivers[["driver", "code", "team"]].to_dict(orient="records")
    schema = {
        "questions": [
            {
                "question_id": "Q1",
                "question": "Exact wording shown by F1 Predict",
                "question_type": "podium",
                "selection_count": 3,
                "options": [
                    {
                        "option": "Driver or answer label exactly as displayed",
                        "option_value": "three-letter driver code, team name, yes/no or canonical bin",
                        "subject_value": "driver code/team targeted by a yes-no or position question, else blank",
                        "opponent_value": "other driver/team for a head-to-head option, else blank",
                        "entity_type": "driver|team|event",
                        "target_position": None,
                        "points": 10,
                    }
                ],
                "source_url": "https://f1predict.formula1.com/en",
            }
        ],
        "sources": ["https://f1predict.formula1.com/en"],
        "status": "open|locked|not_published",
    }
    prompt = (
        f"Fecha de consulta: {date.today().isoformat()}.\n"
        f"Ronda y GP objetivo: {json.dumps(race, ensure_ascii=False)}\n"
        f"Pilotos/equipos validos: {json.dumps(entrants, ensure_ascii=False)}\n\n"
        "Busca en el juego oficial F1 Predict y extrae las 10 preguntas publicadas para esta ronda, "
        "todas sus opciones y los puntos visibles junto a cada opcion. Revisa tambien Game Rules y FAQs. "
        "No confundas F1 Predict con F1 Fantasy ni con juegos de terceros. La pregunta de pole significa "
        "el piloto mas rapido en Q3 de la clasificacion final, aunque tenga sancion de parrilla. En podio, "
        "selection_count es 3 y acertar posicion exacta duplica los puntos base. Usa uno de estos tipos: "
        "podium, winner, pole, qualifying_top5, qualifying_top10, exact_qualifying, fastest_lap, "
        "fastest_pit_stop, most_positions_gained, safety_car, wet_race, red_flag, dnf, "
        "first_retirement, top5, top10, points_finish, exact_finish, driver_h2h, qualifying_h2h, "
        "team_h2h, team_both_top10, sprint_winner, sprint_podium, sprint_pole, winner_from_pole, "
        "classified_count. Si una pregunta no encaja, usa custom. "
        "Para opciones que son pilotos usa option_value=code de tres letras; para equipos, el nombre; "
        "para booleanos usa yes/no y coloca el piloto/equipo preguntado en subject_value. En cara a cara "
        "usa opponent_value cuando la opcion necesite explicitar rival. Para clasificados usa le_15, "
        "16_18 o ge_19. Devuelve SOLO JSON valido con este esquema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        "No reconstruyas ni inventes preguntas. Si la ronda esta cerrada, aun no publicada o el contenido "
        "no es accesible, devuelve questions=[] y el status correcto."
    )
    model = default_model()
    client = OpenAI()
    response = client.responses.create(
        model=model,
        tools=[
            {
                "type": "web_search",
                "search_context_size": "high",
                "filters": {"allowed_domains": TRUSTED_F1_DEEP_SEARCH_DOMAINS},
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
            "Eres extractor del juego oficial F1 Predict. Verifica el GP y copia preguntas, opciones "
            "y puntos sin inventar. Responde exclusivamente JSON valido."
        ),
        input=prompt,
    )
    data = _extract_json(_extract_response_text(response))
    if not isinstance(data, dict):
        raise ValueError("La respuesta sobre F1 Predict no fue un objeto JSON.")
    rows: list[dict[str, Any]] = []
    for question in data.get("questions", []) if isinstance(data.get("questions"), list) else []:
        if not isinstance(question, dict):
            continue
        for option in question.get("options", []) if isinstance(question.get("options"), list) else []:
            if not isinstance(option, dict):
                continue
            rows.append(
                {
                    "question_id": question.get("question_id"),
                    "question": question.get("question"),
                    "question_type": question.get("question_type"),
                    "selection_count": question.get("selection_count", 1),
                    "option": option.get("option"),
                    "option_value": option.get("option_value", option.get("option")),
                    "subject_value": option.get("subject_value"),
                    "opponent_value": option.get("opponent_value"),
                    "entity_type": option.get("entity_type"),
                    "target_position": option.get("target_position"),
                    "points": option.get("points"),
                    "source_url": question.get("source_url", "https://f1predict.formula1.com/en"),
                    "is_official": True,
                }
            )
    questions = normalize_question_catalog(pd.DataFrame(rows, columns=QUESTION_COLUMNS))
    sources = list(data.get("sources", [])) if isinstance(data.get("sources"), list) else []
    for source in _extract_web_search_sources(response):
        if source not in sources:
            sources.append(source)
    return {
        "questions": questions,
        "sources": pd.DataFrame({"source": sources}),
        "status": pd.DataFrame([{"status": str(data.get("status", "unknown"))}]),
    }


def build_analysis_payload(
    driver_results: pd.DataFrame,
    constructor_results: pd.DataFrame,
    race_winners: pd.DataFrame,
    drivers: pd.DataFrame,
    calendar: pd.DataFrame,
    notes: str,
    diagnostics: pd.DataFrame | None = None,
    scenarios: pd.DataFrame | None = None,
    question_catalog: pd.DataFrame | None = None,
    game_answers: pd.DataFrame | None = None,
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
            "Entrega primero la hoja final de respuestas para las 10 preguntas de F1 Predict del "
            "proximo GP. Conserva exactamente el orden y el texto de las preguntas, explica en una "
            "linea la evidencia cuantitativa/cualitativa y marca confianza. Si hay puntos, distingue "
            "la opcion mas probable de la de mayor valor esperado. Despues resume riesgos del GP y "
            "solo al final agrega una nota breve de campeonato de pilotos y constructores."
        ),
    }
    if diagnostics is not None and not diagnostics.empty:
        payload["model_diagnostics"] = diagnostics.head(12).round(2).to_dict(orient="records")
    if scenarios is not None and not scenarios.empty:
        payload["scenario_summary"] = scenarios.round(2).to_dict(orient="records")
    if question_catalog is not None and not question_catalog.empty:
        payload["f1_predict_question_catalog"] = question_catalog.where(pd.notna(question_catalog), None).to_dict(orient="records")
    if game_answers is not None and not game_answers.empty:
        payload["f1_predict_recommended_answers"] = game_answers.where(pd.notna(game_answers), None).to_dict(orient="records")
    return payload


def call_llm_analysis(payload: dict[str, Any]) -> str:
    """Call OpenAI to generate a narrative championship analysis.

    Parameters
    ----------
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

    model = default_model()
    logger.info("Generando analisis LLM: modelo={}, payload_keys={}", model, list(payload))
    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=LLM_INSTRUCTIONS,
        input=json.dumps(payload, ensure_ascii=True),
    )
    return _extract_response_text(response)


instrument_module_functions(__name__)
