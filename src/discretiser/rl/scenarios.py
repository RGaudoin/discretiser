"""
Shared scenario setup for RL experiments.

Provides default scenario configuration and baseline policies for comparison.
"""

from typing import Dict

from ..scenarios.basic_bathtub import BasicBathtubScenario
from ..policy import LinearIntervalPolicy, FixedIntervalPolicy, NoOpPolicy, Policy


# Default scenario parameters (matching DQN notebook)
DEFAULT_SCENARIO_PARAMS = dict(
    scale1=100.0,
    scale2=200.0,
    service_cost=0.5,
    failure_cost=150.0,
    revenue_per_time=1.50,
)

# Default max episode time
DEFAULT_MAX_TIME = 500.0

# Optimal policy parameters (from policy_optimisation notebook)
# Best from multi-start: c=48.4, r=0.50 -> a=24.2, b=24.2
OPTIMAL_POLICY_PARAMS = (24.2, 24.2)


def get_default_scenario() -> BasicBathtubScenario:
    """Create the default BasicBathtubScenario for RL experiments."""
    return BasicBathtubScenario(**DEFAULT_SCENARIO_PARAMS)


def get_baseline_policies() -> Dict[str, Policy]:
    """Get dictionary of baseline policies for comparison."""
    return {
        'no_service': NoOpPolicy(),
        'fixed_25': FixedIntervalPolicy(interval=25.0),
        'fixed_50': FixedIntervalPolicy(interval=50.0),
        'linear_15_10': LinearIntervalPolicy(a=15.0, b=10.0),
        'optimised_linear': LinearIntervalPolicy(a=OPTIMAL_POLICY_PARAMS[0], b=OPTIMAL_POLICY_PARAMS[1]),
    }


def print_scenario_info(scenario: BasicBathtubScenario) -> None:
    """Print scenario parameters for reference."""
    print("Scenario parameters:")
    print(f"  Bathtub shape: shape1={scenario.failure_model.shape1}, shape2={scenario.failure_model.shape2}")
    print(f"  Scales: scale1={scenario.failure_model.scale1}, scale2={scenario.failure_model.scale2}")
    print(f"  Delta_t (age reduction per service): {scenario.failure_model.delta_t}")
    print(f"  Costs: service={scenario.costs.service_cost}, failure={scenario.costs.failure_cost}")
    print(f"  Revenue per time: {scenario.costs.revenue_per_time}")
