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

from .degradation import (
    # Scenario class
    DegradationScenario,
    # Survival model
    DegradationDependentFailure,
    # Subject generation
    generate_degradation_subjects,
    # Events
    create_degradation_events,
    # Baseline policy
    baseline_degradation_service_interval,
    DegradationBaselineState,
    make_degradation_state_factory,
    # Cost structure
    DegradationCosts,
    # Data generation
    generate_degradation_data,
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
    # Degradation
    'DegradationScenario',
    'DegradationDependentFailure',
    'generate_degradation_subjects',
    'create_degradation_events',
    'baseline_degradation_service_interval',
    'DegradationBaselineState',
    'make_degradation_state_factory',
    'DegradationCosts',
    'generate_degradation_data',
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
