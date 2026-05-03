"""F1 Predictor package.

Provides Monte Carlo simulation, data loading, public API ingestion,
LLM analysis and Streamlit UI for predicting the Formula 1 Championship.

Modules
-------
config
    Global constants: paths, URLs, points tables and team colours.
data
    Data loading, validation, cleaning and update helpers.
parameters
    Frozen dataclass with simulation control parameters.
public_apis
    Network ingestion from Jolpica-F1 and OpenF1 REST APIs.
simulator
    Vectorised Monte Carlo race and season simulation engine.
llm
    OpenAI integration for official F1 inputs and narrative analysis.
ui
    Streamlit application layout, widgets and cached computations.
"""

