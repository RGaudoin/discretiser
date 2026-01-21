"""
Basic Bathtub Scenario - Minimal scenario for pipeline validation.

This scenario models widget maintenance with:
- Bathtub-shaped failure hazard (infant mortality + wear-out)
- Effective age model (service rejuvenates widgets)
- Single subject feature (durability)
- Linear baseline policy for training data generation

See docs/scenarios/basic_bathtub.md for full specification.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Callable
from functools import partial

import numpy as np
import pandas as pd

from ..survival import SurvivalModel, CompoundWeibull, NeverOccurs
from ..events import EventType, EventRegistry
from ..state import Subject, State
from ..simulator import Simulator, SimulationResult
from .base import Scenario, CostStructure, summarise_cohort_costs


# =============================================================================
# Survival Model: Effective Age Bathtub
# =============================================================================

class EffectiveAgeBathtub(SurvivalModel):
    """
    Bathtub-shaped failure hazard based on effective age.

    The effective age accounts for rejuvenation from service:
        effective_age = time - service_count * delta_t

    Service "turns back the clock" by delta_t, but real time keeps advancing.
    This creates sequential dependence - optimal strategy changes with age.

    The hazard at real time t is computed from the bathtub at effective_age,
    and sampling conditions on having survived to that effective age.

    Args:
        shape1: Weibull shape for early hazard (typically < 1 for infant mortality)
        scale1: Weibull scale for early hazard
        shape2: Weibull shape for late hazard (typically > 1 for wear-out)
        scale2: Weibull scale for late hazard (base value, scaled by durability)
        delta_t: Effective age reduction per service
        durability_feature: Name of subject feature for durability scaling

    Subject durability scales the base scales (higher durability = longer life).
    """

    def __init__(
        self,
        shape1: float = 0.5,   # <1 = infant mortality (high early hazard)
        scale1: float = 100.0,
        shape2: float = 3.0,   # >1 = wear-out (increasing late hazard)
        scale2: float = 200.0,
        delta_t: float = 15.0,
        durability_feature: str = 'durability'
    ):
        self.shape1 = shape1
        self.scale1 = scale1
        self.shape2 = shape2
        self.scale2 = scale2
        self.delta_t = delta_t
        self.durability_feature = durability_feature

    def _get_effective_age(self, state: State) -> float:
        """Compute effective age from state."""
        service_count = state.event_count('service')
        effective_age = state.time - service_count * self.delta_t
        return max(0.0, effective_age)

    def _get_scaled_bathtub(self, subject: Subject) -> CompoundWeibull:
        """Get bathtub model scaled by subject durability."""
        durability = subject.get_feature(self.durability_feature, 1.0)
        return CompoundWeibull(
            shape1=self.shape1,
            scale1=self.scale1 * durability,
            shape2=self.shape2,
            scale2=self.scale2 * durability
        )

    def sample(self, state: Any = None, subject: Any = None) -> float:
        """
        Sample time to failure from conditional distribution.

        The failure time is sampled from the bathtub distribution conditioned
        on having survived to the current effective age.
        """
        if state is None or subject is None:
            # Fallback to unconditional sampling
            bathtub = CompoundWeibull(
                self.shape1, self.scale1,
                self.shape2, self.scale2
            )
            return bathtub.sample()

        effective_age = self._get_effective_age(state)
        bathtub = self._get_scaled_bathtub(subject)

        if effective_age > 0:
            # Sample from conditional distribution given survival to effective_age
            bathtub = bathtub.truncate(effective_age)

        return bathtub.sample()

    def survival(self, t: float) -> float:
        """
        Unconditional survival function.

        Note: This uses base parameters without state/subject conditioning.
        For state-dependent survival, use the bathtub directly.
        """
        bathtub = CompoundWeibull(
            self.shape1, self.scale1,
            self.shape2, self.scale2
        )
        return bathtub.survival(t)

    def hazard(self, t: float) -> float:
        """Unconditional hazard function."""
        bathtub = CompoundWeibull(
            self.shape1, self.scale1,
            self.shape2, self.scale2
        )
        return bathtub.hazard(t)

    def get_conditional_survival(
        self,
        state: State,
        subject: Subject,
        t: float
    ) -> float:
        """
        Get survival probability at time t given current state.

        This accounts for effective age and durability.
        """
        effective_age = self._get_effective_age(state)
        bathtub = self._get_scaled_bathtub(subject)

        if effective_age > 0:
            bathtub = bathtub.truncate(effective_age)

        return bathtub.survival(t)


# =============================================================================
# Subject Generation
# =============================================================================

def generate_bathtub_subjects(
    n: int,
    durability_mean: float = 1.0,
    durability_std: float = 0.3,
    seed: Optional[int] = None
) -> List[Subject]:
    """
    Generate subjects with durability feature.

    Durability is drawn from a log-normal distribution to ensure:
    - Always positive
    - Right-skewed (some very durable subjects)
    - Mean approximately equals durability_mean

    Args:
        n: Number of subjects to generate
        durability_mean: Target mean durability
        durability_std: Standard deviation of durability
        seed: Random seed for reproducibility

    Returns:
        List of Subject instances with 'durability' feature
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert to log-normal parameters
    # For log-normal: E[X] = exp(mu + sigma^2/2), Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
    # Solve for mu, sigma given mean and std
    variance = durability_std ** 2
    mu = np.log(durability_mean ** 2 / np.sqrt(variance + durability_mean ** 2))
    sigma = np.sqrt(np.log(1 + variance / durability_mean ** 2))

    durabilities = np.random.lognormal(mu, sigma, n)

    subjects = []
    for i, durability in enumerate(durabilities):
        subjects.append(Subject(
            id=i,
            features={'durability': float(durability)},
            feature_vector=np.array([durability])
        ))

    return subjects


# =============================================================================
# Event Definitions
# =============================================================================

def create_bathtub_events(
    failure_model: Optional[EffectiveAgeBathtub] = None
) -> EventRegistry:
    """
    Create event registry for basic bathtub scenario.

    Events:
    - failure: Terminal event with effective age bathtub hazard
    - service: Non-terminal action, scheduled externally by policy

    Args:
        failure_model: Custom failure model (default creates standard one)

    Returns:
        EventRegistry with failure and service events
    """
    if failure_model is None:
        failure_model = EffectiveAgeBathtub()

    failure = EventType(
        name='failure',
        survival_model=failure_model,
        terminal=True,
        metadata={'type': 'terminal'}
    )

    service = EventType(
        name='service',
        survival_model=NeverOccurs(),  # Only occurs via pending
        triggers=[],  # No self-triggering
        metadata={'type': 'action'}
    )

    registry = EventRegistry()
    registry.register(failure)
    registry.register(service)

    return registry


# =============================================================================
# Baseline Policy
# =============================================================================

def baseline_service_interval(
    subject: Subject,
    a: float = 20.0,
    b: float = 10.0
) -> float:
    """
    Linear baseline policy: interval = a + b * durability.

    High-durability subjects get serviced less frequently.
    This creates data diversity tied to features.

    Args:
        subject: Subject to compute interval for
        a: Base interval (intercept)
        b: Durability coefficient (slope)

    Returns:
        Service interval in time units
    """
    durability = subject.get_feature('durability', 1.0)
    return a + b * durability


class BaselineSchedulingState(State):
    """
    State subclass that auto-schedules service using baseline policy.

    This cleanly separates scenario dynamics from policy by using the
    existing state_class extension point in the Simulator.
    """

    def __init__(
        self,
        subject: Subject,
        baseline_a: float = 20.0,
        baseline_b: float = 10.0,
        start_time: float = 0.0
    ):
        super().__init__(subject, start_time)
        self.baseline_a = baseline_a
        self.baseline_b = baseline_b

        # Schedule first service
        interval = baseline_service_interval(subject, baseline_a, baseline_b)
        self.add_pending_event('service', interval, 'baseline')

    def record_event(
        self,
        event_name: str,
        time: float,
        triggered_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record event and reschedule service if needed."""
        super().record_event(event_name, time, triggered_by, metadata)

        if event_name == 'service':
            # Schedule next service
            interval = baseline_service_interval(
                self.subject, self.baseline_a, self.baseline_b
            )
            self.add_pending_event('service', time + interval, 'baseline')


def make_baseline_state_factory(
    baseline_a: float = 20.0,
    baseline_b: float = 10.0
) -> Callable[[Subject], BaselineSchedulingState]:
    """
    Create a state factory with bound baseline parameters.

    Usage:
        factory = make_baseline_state_factory(a=20, b=10)
        result = simulator.simulate_subject(subject, state_class=factory)
    """
    def factory(subject: Subject) -> BaselineSchedulingState:
        return BaselineSchedulingState(subject, baseline_a, baseline_b)
    return factory


# =============================================================================
# Cost Structure
# =============================================================================

@dataclass
class BathtubCosts(CostStructure):
    """
    Cost structure for basic bathtub scenario.

    Trade-off:
    - Service costs money but extends operational life
    - Failure has high terminal cost
    - Revenue accrues while operational

    Optimal policy balances service frequency vs failure risk.
    """
    service_cost: float = 50.0
    failure_cost: float = 500.0
    revenue_per_time: float = 1.0

    def __post_init__(self):
        # Set up parent class fields
        self.action_costs = {'service': self.service_cost}
        self.terminal_costs = {'failure': self.failure_cost}


# =============================================================================
# Data Generation
# =============================================================================

def generate_bathtub_data(
    n_subjects: int,
    max_time: float = 200.0,
    baseline_a: float = 20.0,
    baseline_b: float = 10.0,
    scenario: Optional['BasicBathtubScenario'] = None,
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, List[SimulationResult], List[Dict[str, float]]]:
    """
    Generate training data using baseline policy.

    Uses BasicBathtubScenario for dynamics and costs - demonstrating
    how generic code can work with the Scenario interface.

    Args:
        n_subjects: Number of subjects to simulate
        max_time: Maximum simulation time per subject
        baseline_a: Baseline interval intercept
        baseline_b: Baseline interval slope
        scenario: Optional pre-configured scenario (default creates standard one)
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - DataFrame with event history (subject_id, event, time, triggered_by)
        - List of SimulationResult objects
        - List of cost dictionaries per subject
    """
    if seed is not None:
        np.random.seed(seed)

    # Use provided scenario or create default
    if scenario is None:
        scenario = BasicBathtubScenario()

    # Generate subjects via scenario
    subjects = scenario.generate_subjects(n_subjects)

    # Create event registry via scenario
    registry = scenario.create_event_registry()

    # Create simulator
    simulator = Simulator(
        event_registry=registry,
        max_time=max_time,
        mode='competing'
    )

    # Create state factory for baseline policy
    state_factory = make_baseline_state_factory(baseline_a, baseline_b)

    # Simulate all subjects
    results = []
    for subject in subjects:
        result = simulator.simulate_subject(
            subject,
            state_class=lambda s, factory=state_factory: factory(s)
        )
        results.append(result)

    # Combine to DataFrame
    df = simulator.cohort_to_dataframe(results)

    # Add subject features to DataFrame
    if not df.empty:
        subject_features = {s.id: s.features for s in subjects}
        df['durability'] = df['subject_id'].map(
            lambda sid: subject_features.get(sid, {}).get('durability', 1.0)
        )

    # Calculate costs via scenario
    cost_results = [scenario.calculate_costs(r) for r in results]

    return df, results, cost_results


# =============================================================================
# Scenario Class
# =============================================================================

class BasicBathtubScenario(Scenario):
    """
    Basic bathtub maintenance scenario.

    Implements the Scenario interface for integration with generic
    validation and RL pipelines.
    """

    def __init__(
        self,
        shape1: float = 0.5,   # <1 = infant mortality (high early hazard)
        scale1: float = 100.0,
        shape2: float = 3.0,   # >1 = wear-out (increasing late hazard)
        scale2: float = 200.0,
        delta_t: float = 15.0,
        durability_mean: float = 1.0,
        durability_std: float = 0.3,
        service_cost: float = 20.0,
        failure_cost: float = 500.0,
        revenue_per_time: float = 3.0
    ):
        self.failure_model = EffectiveAgeBathtub(
            shape1=shape1,
            scale1=scale1,
            shape2=shape2,
            scale2=scale2,
            delta_t=delta_t
        )
        self.durability_mean = durability_mean
        self.durability_std = durability_std
        self.costs = BathtubCosts(
            service_cost=service_cost,
            failure_cost=failure_cost,
            revenue_per_time=revenue_per_time
        )

    def create_event_registry(self) -> EventRegistry:
        return create_bathtub_events(self.failure_model)

    def generate_subjects(
        self,
        n: int,
        seed: Optional[int] = None
    ) -> List[Subject]:
        return generate_bathtub_subjects(
            n,
            durability_mean=self.durability_mean,
            durability_std=self.durability_std,
            seed=seed
        )

    def calculate_costs(self, result: SimulationResult) -> Dict[str, float]:
        return self.costs.calculate(result)
