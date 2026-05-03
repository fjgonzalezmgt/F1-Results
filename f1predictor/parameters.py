"""Simulation parameters for the F1 Monte Carlo engine.

Exposes a single frozen dataclass, ``SimParams``, whose fields control
every tuneable knob of the simulation: iteration count, random seed,
driver/constructor/form weights, noise level and external-factor multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass


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
        Relative weight of the recent-form component in base strength.
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
    """

    simulations: int = 4000
    seed: int = 2026
    start_round: int = 4
    include_sprints: bool = True
    driver_weight: float = 0.42
    constructor_weight: float = 0.48
    form_weight: float = 0.10
    qualifying_weight: float = 0.72
    chaos: float = 5.5
    reliability_multiplier: float = 1.0
    weather_multiplier: float = 1.0
    safety_car_multiplier: float = 1.0
    development_drift: float = 2.4
