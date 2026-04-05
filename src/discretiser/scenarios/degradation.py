"""
Degradation scenario: multiple competing events with history dependence.

Three events:
- degradation: non-terminal Weibull, occurs naturally
- failure: terminal Weibull, scale decreases with degradation count
- service: non-terminal, scheduled by policy, resets degradation count

This creates genuine sequential dependence — the failure hazard at any
point depends on how many degradation events have occurred since the last
service. Unlike the basic bathtub (where effective age is deterministic),
here the degradation process is stochastic.

Optimisation problem: when to service, balancing service cost against
increasing failure risk from accumulated degradation.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..events import EventType, EventRegistry
from ..state import State, Subject
from ..survival import SurvivalModel, Weibull, NeverOccurs
from .base import Scenario, CostStructure


# =============================================================================
# Survival models
# =============================================================================

class DegradationDependentFailure(SurvivalModel):
    """Failure model where hazard increases with degradation count.

    effective_scale = base_scale * durability / (1 + deg_count * degradation_factor)

    More degradation events → lower scale → higher hazard → earlier failure.
    Service resets the degradation count (tracked via state).

    Args:
        base_shape: Weibull shape parameter.
        base_scale: Weibull scale at zero degradation (before durability scaling).
        degradation_factor: How much each degradation event reduces scale.
            At n degradation events, scale is divided by (1 + n * factor).
        durability_feature: Subject feature name for durability scaling.
    """

    def __init__(
        self,
        base_shape: float = 2.0,
        base_scale: float = 200.0,
        degradation_factor: float = 0.3,
        durability_feature: str = 'durability',
    ):
        self.base_shape = base_shape
        self.base_scale = base_scale
        self.degradation_factor = degradation_factor
        self.durability_feature = durability_feature
        # Cache for survival/hazard calls (set during sample)
        self._current_weibull: Optional[Weibull] = None

    def _get_degradation_count(self, state: State) -> int:
        """Count degradation events since last service."""
        # Walk history backwards to find last service
        for record in reversed(state.history):
            if record.event_name == 'service':
                # Count degradation events after this service
                count = 0
                found_service = False
                for r in state.history:
                    if r is record:
                        found_service = True
                        continue
                    if found_service and r.event_name == 'degradation':
                        count += 1
                return count
        # No service ever — count all degradation events
        return state.event_count('degradation')

    def _build_weibull(self, state: State, subject: Subject) -> Weibull:
        """Build Weibull with effective parameters."""
        deg_count = self._get_degradation_count(state)
        durability = subject.get_feature(self.durability_feature, 1.0)
        effective_scale = (
            self.base_scale * durability / (1 + deg_count * self.degradation_factor)
        )
        return Weibull(self.base_shape, effective_scale)

    def sample(self, state: Any = None, subject: Any = None) -> float:
        if state is None or subject is None:
            self._current_weibull = Weibull(self.base_shape, self.base_scale)
            return self._current_weibull.sample()
        self._current_weibull = self._build_weibull(state, subject)
        return self._current_weibull.sample()

    def survival(self, t: float) -> float:
        if self._current_weibull is None:
            return Weibull(self.base_shape, self.base_scale).survival(t)
        return self._current_weibull.survival(t)

    def hazard(self, t: float) -> float:
        if self._current_weibull is None:
            return Weibull(self.base_shape, self.base_scale).hazard(t)
        return self._current_weibull.hazard(t)

    def log_pdf(self, t: float) -> float:
        if self._current_weibull is None:
            return Weibull(self.base_shape, self.base_scale).log_pdf(t)
        return self._current_weibull.log_pdf(t)

    def get_effective_params(
        self, state: State, subject: Subject
    ) -> Dict[str, float]:
        """Get current effective Weibull parameters (for diagnostics/NLL)."""
        w = self._build_weibull(state, subject)
        return {'shape': w.shape, 'scale': w.scale}


# =============================================================================
# Event definitions
# =============================================================================

def create_degradation_events(
    degradation_model: Optional[SurvivalModel] = None,
    failure_model: Optional[DegradationDependentFailure] = None,
) -> EventRegistry:
    """Create event registry for degradation scenario.

    Events:
    - degradation: Non-terminal, Weibull. Increases failure risk.
    - failure: Terminal, state-dependent Weibull.
    - service: Non-terminal, NeverOccurs (scheduled by policy).
    """
    if degradation_model is None:
        degradation_model = Weibull(shape=1.5, scale=30.0)

    if failure_model is None:
        failure_model = DegradationDependentFailure()

    degradation = EventType(
        name='degradation',
        survival_model=degradation_model,
        terminal=False,
        metadata={'type': 'natural'},
    )

    failure = EventType(
        name='failure',
        survival_model=failure_model,
        terminal=True,
        metadata={'type': 'terminal'},
    )

    service = EventType(
        name='service',
        survival_model=NeverOccurs(),
        terminal=False,
        metadata={'type': 'action'},
    )

    registry = EventRegistry()
    registry.register(degradation)
    registry.register(failure)
    registry.register(service)
    return registry


# =============================================================================
# Subject generation
# =============================================================================

def generate_degradation_subjects(
    n: int,
    durability_mean: float = 1.0,
    durability_std: float = 0.3,
    seed: Optional[int] = None,
) -> List[Subject]:
    """Generate subjects with durability feature (same as bathtub)."""
    if seed is not None:
        np.random.seed(seed)

    variance = durability_std ** 2
    mu = np.log(durability_mean ** 2 / np.sqrt(variance + durability_mean ** 2))
    sigma = np.sqrt(np.log(1 + variance / durability_mean ** 2))

    durabilities = np.random.lognormal(mu, sigma, n)

    subjects = []
    for i, durability in enumerate(durabilities):
        subjects.append(Subject(
            id=i,
            features={'durability': float(durability)},
            feature_vector=np.array([durability]),
        ))
    return subjects


# =============================================================================
# Cost structure
# =============================================================================

@dataclass
class DegradationCosts(CostStructure):
    """Cost structure for degradation scenario.

    Trade-off: service resets degradation but costs money.
    Ignoring degradation risks expensive failure.
    """
    service_cost: float = 20.0
    failure_cost: float = 500.0
    revenue_per_time: float = 3.0

    def __post_init__(self):
        self.action_costs = {'service': self.service_cost}
        self.terminal_costs = {'failure': self.failure_cost}


# =============================================================================
# Scenario class
# =============================================================================

class DegradationScenario(Scenario):
    """Degradation scenario with history-dependent failure.

    Three competing events where degradation accumulates and service
    resets it. Creates genuine sequential dependence for learning.

    Args:
        deg_shape: Weibull shape for degradation events.
        deg_scale: Weibull scale for degradation events.
        fail_shape: Weibull shape for failure.
        fail_base_scale: Base Weibull scale for failure (at zero degradation).
        degradation_factor: How much each degradation worsens failure.
        durability_mean: Mean subject durability.
        durability_std: Std of subject durability.
        service_cost: Cost per service action.
        failure_cost: Terminal failure cost.
        revenue_per_time: Revenue rate while operational.
    """

    def __init__(
        self,
        deg_shape: float = 1.5,
        deg_scale: float = 30.0,
        fail_shape: float = 2.0,
        fail_base_scale: float = 200.0,
        degradation_factor: float = 0.3,
        durability_mean: float = 1.0,
        durability_std: float = 0.3,
        service_cost: float = 20.0,
        failure_cost: float = 500.0,
        revenue_per_time: float = 3.0,
    ):
        self.degradation_model = Weibull(shape=deg_shape, scale=deg_scale)
        self.failure_model = DegradationDependentFailure(
            base_shape=fail_shape,
            base_scale=fail_base_scale,
            degradation_factor=degradation_factor,
        )
        self.durability_mean = durability_mean
        self.durability_std = durability_std
        self.costs = DegradationCosts(
            service_cost=service_cost,
            failure_cost=failure_cost,
            revenue_per_time=revenue_per_time,
        )

    def create_event_registry(self) -> EventRegistry:
        return create_degradation_events(
            degradation_model=self.degradation_model,
            failure_model=self.failure_model,
        )

    def generate_subjects(
        self, n: int, seed: Optional[int] = None
    ) -> List[Subject]:
        return generate_degradation_subjects(
            n,
            durability_mean=self.durability_mean,
            durability_std=self.durability_std,
            seed=seed,
        )

    def calculate_costs(self, result) -> Dict[str, float]:
        return self.costs.calculate(result)


# =============================================================================
# Baseline policy and data generation
# =============================================================================

def baseline_degradation_service_interval(
    subject: Subject,
    a: float = 20.0,
    b: float = 10.0,
) -> float:
    """Baseline service interval: a + b * durability."""
    durability = subject.get_feature('durability', 1.0)
    return a + b * durability


class DegradationBaselineState(State):
    """State that auto-schedules service using a baseline policy.

    Service is rescheduled after each service event. Degradation and
    failure events don't trigger rescheduling.
    """

    def __init__(
        self,
        subject: Subject,
        baseline_a: float = 20.0,
        baseline_b: float = 10.0,
        start_time: float = 0.0,
    ):
        super().__init__(subject, start_time)
        self.baseline_a = baseline_a
        self.baseline_b = baseline_b

        # Schedule first service
        interval = baseline_degradation_service_interval(
            subject, baseline_a, baseline_b
        )
        self.add_pending_event('service', interval, 'baseline')

    def record_event(
        self,
        event_name: str,
        time: float,
        triggered_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().record_event(event_name, time, triggered_by, metadata)

        if event_name == 'service':
            interval = baseline_degradation_service_interval(
                self.subject, self.baseline_a, self.baseline_b
            )
            self.add_pending_event('service', time + interval, 'baseline')


def make_degradation_state_factory(
    baseline_a: float = 20.0,
    baseline_b: float = 10.0,
) -> Callable[[Subject], DegradationBaselineState]:
    """Create a state factory for degradation baseline policy."""
    def factory(subject: Subject) -> DegradationBaselineState:
        return DegradationBaselineState(subject, baseline_a, baseline_b)
    return factory


def generate_degradation_data(
    n_subjects: int,
    max_time: float = 300.0,
    baseline_a: float = 20.0,
    baseline_b: float = 10.0,
    scenario: Optional[DegradationScenario] = None,
    seed: Optional[int] = None,
) -> Tuple[List, List[Dict[str, float]]]:
    """Generate simulation data using baseline policy.

    Args:
        n_subjects: Number of subjects to simulate.
        max_time: Maximum simulation time per subject.
        baseline_a: Baseline interval intercept.
        baseline_b: Baseline interval slope.
        scenario: Optional pre-configured scenario.
        seed: Random seed.

    Returns:
        Tuple of (SimulationResult list, cost dict list).
    """
    import pandas as pd
    from ..simulator import Simulator

    if seed is not None:
        np.random.seed(seed)

    if scenario is None:
        scenario = DegradationScenario()

    subjects = scenario.generate_subjects(n_subjects)
    registry = scenario.create_event_registry()
    simulator = Simulator(event_registry=registry, max_time=max_time, mode='competing')
    state_factory = make_degradation_state_factory(baseline_a, baseline_b)

    results = []
    for subject in subjects:
        result = simulator.simulate_subject(
            subject,
            state_class=lambda s, factory=state_factory: factory(s),
        )
        results.append(result)

    cost_results = [scenario.calculate_costs(r) for r in results]
    return results, cost_results
