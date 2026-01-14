# Synthetic scenarios for model validation
#
# Each scenario defines:
# - Events and their survival models
# - Baseline policy (for data generation)
# - Cost structure
# - Validation criteria
#
# See docs/scenarios/ for detailed documentation.

from .base import (
    Scenario,
    CostStructure,
    summarise_cohort_costs,
)

from .basic_bathtub import (
    # Scenario class
    BasicBathtubScenario,
    # Survival model
    EffectiveAgeBathtub,
    # Subject generation
    generate_bathtub_subjects,
    # Events
    create_bathtub_events,
    # Baseline policy
    baseline_service_interval,
    BaselineSchedulingState,
    make_baseline_state_factory,
    # Cost structure
    BathtubCosts,
    # Data generation
    generate_bathtub_data,
)

__all__ = [
    # Base
    'Scenario',
    'CostStructure',
    'summarise_cohort_costs',
    # Basic Bathtub
    'BasicBathtubScenario',
    'EffectiveAgeBathtub',
    'generate_bathtub_subjects',
    'create_bathtub_events',
    'baseline_service_interval',
    'BaselineSchedulingState',
    'make_baseline_state_factory',
    'BathtubCosts',
    'generate_bathtub_data',
]
