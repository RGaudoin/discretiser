"""
discretiser - Competing risks simulation for patient journey data.
"""

from .survival import (
    SurvivalModel,
    Weibull,
    Exponential,
    LogNormal,
    Gamma,
    DeltaMass,
    PointMassPlusContinuous,
    PointMasses,
    Mixture,
    NeverOccurs,
    CompoundWeibull,
    MinSurvival,
    TruncatedSurvival,
    StateDependentWeibull,
    TrainedModelSurvival,
)

from .events import (
    TriggerRule,
    EventType,
    EventRegistry,
    make_censoring_event,
    make_terminal_event,
    make_triggering_event,
    make_conditional_event,
)

from .state import (
    EventRecord,
    Subject,
    State,
    EmbeddingState,
    generate_subjects,
)

from .simulator import (
    SimulationResult,
    Simulator,
    simulate_journey,
    simulate_cohort_simple,
    extract_training_data,
)

__all__ = [
    # Survival models
    "SurvivalModel",
    "Weibull",
    "Exponential",
    "LogNormal",
    "Gamma",
    "DeltaMass",
    "PointMassPlusContinuous",
    "PointMasses",
    "Mixture",
    "NeverOccurs",
    "CompoundWeibull",
    "MinSurvival",
    "TruncatedSurvival",
    "StateDependentWeibull",
    "TrainedModelSurvival",
    # Events
    "TriggerRule",
    "EventType",
    "EventRegistry",
    "make_censoring_event",
    "make_terminal_event",
    "make_triggering_event",
    "make_conditional_event",
    # State
    "EventRecord",
    "Subject",
    "State",
    "EmbeddingState",
    "generate_subjects",
    # Simulator
    "SimulationResult",
    "Simulator",
    "simulate_journey",
    "simulate_cohort_simple",
    "extract_training_data",
]
