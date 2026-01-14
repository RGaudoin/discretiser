"""
Generic simulation runner with policy integration.

Separates scenario (dynamics) from policy (decisions) for:
- Testing different policies on same scenario
- Comparing ground truth vs learned model scenarios
- Policy optimisation
"""

from typing import List, Optional, Dict, Any, Type
from dataclasses import dataclass
import numpy as np

from .state import State, Subject
from .events import EventRegistry
from .simulator import Simulator, SimulationResult
from .policy import Policy, Action
from .scenarios.base import Scenario, CostStructure


class PolicyState(State):
    """
    State subclass that integrates a policy.

    After each event, gives the policy a chance to schedule actions.
    """

    def __init__(
        self,
        subject: Subject,
        policy: Policy,
        action_source: str = 'policy'
    ):
        super().__init__(subject)
        self.policy = policy
        self.action_source = action_source
        self.policy.reset()

        # Schedule initial action
        self._apply_policy_action()

    def record_event(self, event_name: str, time: float, **kwargs) -> None:
        """Record event and let policy decide next action."""
        super().record_event(event_name, time, **kwargs)

        # After recording, let policy decide
        self._apply_policy_action()

    def _apply_policy_action(self) -> None:
        """Query policy and schedule any action it returns."""
        action = self.policy.get_action(self, self.subject)
        if action is not None:
            scheduled_time = self.time + action.delay
            self.add_pending_event(
                action.event_name,
                scheduled_time,
                triggered_by=self.action_source
            )


def make_policy_state_factory(policy: Policy) -> Type[State]:
    """
    Create a State class factory that uses the given policy.

    Returns a class (not instance) that can be passed to Simulator.
    """
    class ConfiguredPolicyState(PolicyState):
        def __init__(self, subject: Subject):
            super().__init__(subject, policy)

    return ConfiguredPolicyState


@dataclass
class RunResult:
    """Result of running a scenario with a policy."""
    results: List[SimulationResult]
    costs: List[Dict[str, Any]]
    total_net_value: float
    mean_net_value: float
    std_net_value: float


def run_scenario(
    scenario: Scenario,
    policy: Policy,
    n_subjects: int,
    max_time: float = 200.0,
    seed: Optional[int] = None
) -> RunResult:
    """
    Run a scenario with a policy.

    Args:
        scenario: Scenario defining dynamics and costs
        policy: Policy defining decisions
        n_subjects: Number of subjects to simulate
        max_time: Maximum simulation time
        seed: Random seed for reproducibility

    Returns:
        RunResult with simulation results and cost analysis
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate subjects
    subjects = scenario.generate_subjects(n_subjects)

    # Create event registry
    registry = scenario.create_event_registry()

    # Create simulator
    simulator = Simulator(registry, max_time=max_time)

    # Create state factory with policy
    state_class = make_policy_state_factory(policy)

    # Run simulations
    results = simulator.simulate_cohort(subjects, state_class=state_class)

    # Calculate costs
    costs = [scenario.calculate_costs(r) for r in results]

    # Aggregate
    net_values = [c['net_value'] for c in costs]

    return RunResult(
        results=results,
        costs=costs,
        total_net_value=sum(net_values),
        mean_net_value=np.mean(net_values),
        std_net_value=np.std(net_values)
    )


def compare_policies(
    scenario: Scenario,
    policies: Dict[str, Policy],
    n_subjects: int = 1000,
    max_time: float = 200.0,
    n_repeats: int = 5,
    seed: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple policies on the same scenario.

    Args:
        scenario: Scenario to test on
        policies: Dict mapping policy name to Policy instance
        n_subjects: Subjects per run
        max_time: Simulation max time
        n_repeats: Repetitions for confidence intervals
        seed: Base random seed

    Returns:
        Dict mapping policy name to stats (mean, std, min, max)
    """
    results = {}

    for name, policy in policies.items():
        values = []
        for i in range(n_repeats):
            rep_seed = None if seed is None else seed + i * 1000
            run_result = run_scenario(
                scenario, policy, n_subjects, max_time, rep_seed
            )
            values.append(run_result.mean_net_value)

        results[name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }

    return results
