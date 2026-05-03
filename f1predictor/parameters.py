"""Simulation parameters for the F1 Monte Carlo engine.

Exposes a single frozen dataclass, ``SimParams``, whose fields control
every tuneable knob of the simulation: iteration count, random seed,
driver/constructor/form weights, noise level and external-factor multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass

from f1predictor.config import DEFAULT_SEASON


@dataclass(frozen=True)
class SimParams:
    """Controls for race and season simulation.

    Attributes
    ----------
    simulations : int
        Number of Monte Carlo season iterations to run.
    seed : int
        Random seed for the NumPy generator, ensuring reproducible results.
    start_round : int
        First round included in the forward simulation; rounds before this
        value are treated as completed with ``current_points`` locked.
    include_sprints : bool
        Whether to award sprint points for events with a pending sprint.
    driver_weight : float
        Relative weight of the driver-skill component in base strength.
    constructor_weight : float
        Relative weight of the constructor-pace component in base strength.
    form_weight : float
        Initial relative weight of the recent-form component in base
        strength. The simulator decays this effect across future races so
        early-season form does not dominate the full calendar.
    form_decay_races : float
        Number of future races over which recent-form influence fades.
        Lower values make form mostly affect the next few events.
    qualifying_weight : float
        Scaling factor controlling how strongly grid position carries over
        to race performance.
    chaos : float
        Standard deviation multiplier for lap-time noise; higher values
        produce more unpredictable outcomes.
    reliability_multiplier : float
        Scales all DNF probabilities uniformly (1.0 = baseline).
    weather_multiplier : float
        Scales the probability of a wet qualifying or race session
        (1.0 = baseline).
    safety_car_multiplier : float
        Scales safety-car deployment probability (1.0 = baseline).
    development_drift : float
        Standard deviation of the per-team performance drift that grows
        linearly across the season to model in-season development.
    team_uncertainty : float
        Standard deviation of persistent per-team package uncertainty. This
        widens the true-car-performance distribution when only a few rounds
        have been completed.
    """

    simulations: int = 4000
    seed: int = DEFAULT_SEASON
    start_round: int = 4
    include_sprints: bool = True
    driver_weight: float = 0.42
    constructor_weight: float = 0.48
    form_weight: float = 0.06
    form_decay_races: float = 2.5
    qualifying_weight: float = 0.72
    chaos: float = 5.5
    reliability_multiplier: float = 1.0
    weather_multiplier: float = 1.0
    safety_car_multiplier: float = 1.0
    development_drift: float = 2.4
    team_uncertainty: float = 6.0
