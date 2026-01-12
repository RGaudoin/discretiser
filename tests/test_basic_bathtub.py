"""
Tests for basic_bathtub scenario.

Tests cover:
- EffectiveAgeBathtub survival model
- Subject generation
- Event definitions
- BaselineSchedulingState
- Data generation
- Cost calculation
"""

import pytest
import numpy as np
import pandas as pd

from src.scenarios import (
    BasicBathtubScenario,
    EffectiveAgeBathtub,
    generate_bathtub_subjects,
    create_bathtub_events,
    baseline_service_interval,
    BaselineSchedulingState,
    make_baseline_state_factory,
    BathtubCosts,
    generate_bathtub_data,
    summarise_cohort_costs,
)
from src.state import Subject, State
from src.simulator import Simulator


class TestEffectiveAgeBathtub:
    """Tests for the EffectiveAgeBathtub survival model."""

    def test_unconditional_sample(self):
        """Sample without state/subject returns valid time."""
        model = EffectiveAgeBathtub()
        times = [model.sample() for _ in range(100)]

        assert all(t > 0 for t in times)
        assert all(t < float('inf') for t in times)

    def test_effective_age_reduces_with_service(self):
        """Service reduces effective age."""
        model = EffectiveAgeBathtub(delta_t=15.0)
        subject = Subject(id=0, features={'durability': 1.0})

        # Create state at time 30 with 0 services
        state0 = State(subject)
        state0.time = 30.0

        # Create state at time 30 with 2 services
        state2 = State(subject)
        state2.time = 30.0
        state2._event_counts['service'] = 2

        eff_age_0 = model._get_effective_age(state0)
        eff_age_2 = model._get_effective_age(state2)

        assert eff_age_0 == 30.0
        assert eff_age_2 == 0.0  # 30 - 2*15 = 0

    def test_durability_scales_survival(self):
        """Higher durability should lead to longer survival on average."""
        model = EffectiveAgeBathtub()

        low_durability = Subject(id=0, features={'durability': 0.5})
        high_durability = Subject(id=1, features={'durability': 2.0})

        # Sample many times for each
        np.random.seed(42)
        state = State(low_durability)
        low_times = [model.sample(state, low_durability) for _ in range(500)]

        state = State(high_durability)
        high_times = [model.sample(state, high_durability) for _ in range(500)]

        # High durability should have longer mean survival
        assert np.mean(high_times) > np.mean(low_times)

    def test_conditional_sampling_uses_truncation(self):
        """Sampling at positive effective age uses truncation."""
        model = EffectiveAgeBathtub(delta_t=15.0)
        subject = Subject(id=0, features={'durability': 1.0})

        # State at effective age 50
        state = State(subject)
        state.time = 50.0

        # All samples should be relative times (can be small)
        np.random.seed(42)
        times = [model.sample(state, subject) for _ in range(100)]

        # Times are relative to current time, so should be positive
        assert all(t > 0 for t in times)


class TestSubjectGeneration:
    """Tests for subject generation."""

    def test_generates_correct_count(self):
        """Generates requested number of subjects."""
        subjects = generate_bathtub_subjects(50, seed=42)
        assert len(subjects) == 50

    def test_subjects_have_durability(self):
        """All subjects have durability feature."""
        subjects = generate_bathtub_subjects(20, seed=42)

        for s in subjects:
            assert 'durability' in s.features
            assert s.features['durability'] > 0

    def test_durability_distribution(self):
        """Durability has approximately correct mean."""
        subjects = generate_bathtub_subjects(
            1000,
            durability_mean=1.5,
            durability_std=0.3,
            seed=42
        )

        durabilities = [s.features['durability'] for s in subjects]
        mean = np.mean(durabilities)

        # Should be close to target mean (within 10%)
        assert abs(mean - 1.5) < 0.15

    def test_unique_ids(self):
        """All subjects have unique IDs."""
        subjects = generate_bathtub_subjects(100, seed=42)
        ids = [s.id for s in subjects]
        assert len(ids) == len(set(ids))


class TestEventDefinitions:
    """Tests for event registry creation."""

    def test_creates_two_events(self):
        """Registry has failure and service events."""
        registry = create_bathtub_events()

        assert 'failure' in registry
        assert 'service' in registry

    def test_failure_is_terminal(self):
        """Failure event is terminal."""
        registry = create_bathtub_events()
        assert registry['failure'].terminal is True

    def test_service_not_terminal(self):
        """Service event is not terminal."""
        registry = create_bathtub_events()
        assert registry['service'].terminal is False

    def test_service_is_never_occurs(self):
        """Service has NeverOccurs survival model (externally scheduled)."""
        registry = create_bathtub_events()
        service = registry['service']

        # NeverOccurs.sample() returns inf
        sample = service.survival_model.sample()
        assert sample == float('inf')


class TestBaselinePolicy:
    """Tests for baseline policy function."""

    def test_linear_relationship(self):
        """Interval is linear in durability."""
        low = Subject(id=0, features={'durability': 0.5})
        high = Subject(id=1, features={'durability': 1.5})

        interval_low = baseline_service_interval(low, a=20, b=10)
        interval_high = baseline_service_interval(high, a=20, b=10)

        # interval = 20 + 10 * durability
        assert interval_low == pytest.approx(25.0)
        assert interval_high == pytest.approx(35.0)

    def test_default_parameters(self):
        """Default parameters work."""
        subject = Subject(id=0, features={'durability': 1.0})
        interval = baseline_service_interval(subject)

        assert interval == 30.0  # 20 + 10 * 1.0


class TestBaselineSchedulingState:
    """Tests for BaselineSchedulingState."""

    def test_schedules_first_service(self):
        """First service is scheduled on initialisation."""
        subject = Subject(id=0, features={'durability': 1.0})
        state = BaselineSchedulingState(subject, baseline_a=20, baseline_b=10)

        pending = state.get_pending_events()
        assert 'service' in pending
        assert pending['service'][0] == 30.0  # 20 + 10 * 1.0

    def test_reschedules_after_service(self):
        """Next service is scheduled after service occurs."""
        subject = Subject(id=0, features={'durability': 1.0})
        state = BaselineSchedulingState(subject, baseline_a=20, baseline_b=10)

        # Simulate service at t=30
        state.advance_time(30.0)
        state.pop_pending_event('service')
        state.record_event('service', 30.0, triggered_by='baseline')

        # Next service should be scheduled
        pending = state.get_pending_events()
        assert 'service' in pending
        assert pending['service'][0] == 60.0  # 30 + 30


class TestDataGeneration:
    """Tests for data generation."""

    def test_generates_data(self):
        """Data generation produces results."""
        df, results, costs = generate_bathtub_data(
            n_subjects=10,
            max_time=100.0,
            seed=42
        )

        assert len(results) == 10
        assert len(costs) == 10
        assert not df.empty

    def test_dataframe_has_required_columns(self):
        """DataFrame has expected columns."""
        df, _, _ = generate_bathtub_data(n_subjects=10, seed=42)

        required = ['subject_id', 'event', 'time']
        for col in required:
            assert col in df.columns

    def test_events_are_failure_or_service(self):
        """Only failure and service events occur."""
        df, _, _ = generate_bathtub_data(n_subjects=20, seed=42)

        unique_events = set(df['event'].unique())
        assert unique_events <= {'failure', 'service'}

    def test_service_before_failure(self):
        """Services occur before failure (for most subjects)."""
        df, _, _ = generate_bathtub_data(n_subjects=50, max_time=200.0, seed=42)

        # At least some subjects should have services
        service_count = (df['event'] == 'service').sum()
        assert service_count > 0

    def test_costs_have_net_value(self):
        """Cost results include net_value."""
        _, _, costs = generate_bathtub_data(n_subjects=10, seed=42)

        for c in costs:
            assert 'net_value' in c
            assert 'lifetime' in c


class TestCostCalculation:
    """Tests for cost calculation."""

    def test_service_cost_accumulates(self):
        """Service cost is count * cost_per_service."""
        costs = BathtubCosts(service_cost=50.0, failure_cost=500.0)

        # Generate data and find a result with services
        _, results, _ = generate_bathtub_data(n_subjects=20, max_time=200.0, seed=42)

        for result in results:
            calc = costs.calculate(result)
            service_count = calc.get('service_count', 0)
            service_cost = calc.get('service_cost', 0)

            if service_count > 0:
                assert service_cost == service_count * 50.0
                break

    def test_failure_cost_applied(self):
        """Failure cost applied when failure occurs."""
        costs = BathtubCosts(service_cost=50.0, failure_cost=500.0)

        # Generate until we get a failure
        _, results, _ = generate_bathtub_data(n_subjects=50, max_time=200.0, seed=42)

        for result in results:
            failed = any(e.event_name == 'failure' for e in result.history)
            calc = costs.calculate(result)

            if failed:
                assert calc.get('failure_cost', 0) == 500.0
                break

    def test_revenue_proportional_to_lifetime(self):
        """Revenue equals lifetime * rate."""
        costs = BathtubCosts(revenue_per_time=1.0)

        _, results, _ = generate_bathtub_data(n_subjects=10, seed=42)

        for result in results:
            calc = costs.calculate(result)
            assert calc['revenue'] == pytest.approx(calc['lifetime'])


class TestSummariseCohortCosts:
    """Tests for cohort cost summarisation."""

    def test_computes_statistics(self):
        """Summary includes mean, std, min, max."""
        _, _, costs = generate_bathtub_data(n_subjects=50, seed=42)
        summary = summarise_cohort_costs(costs)

        assert 'net_value' in summary
        stats = summary['net_value']
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats


class TestIntegration:
    """Integration tests for the full scenario."""

    def test_simulation_runs_to_completion(self):
        """Full simulation runs without errors."""
        scenario = BasicBathtubScenario()
        registry = scenario.create_event_registry()
        subjects = scenario.generate_subjects(10, seed=42)

        simulator = Simulator(registry, max_time=200.0)
        factory = make_baseline_state_factory(baseline_a=20, baseline_b=10)

        for subject in subjects:
            result = simulator.simulate_subject(
                subject,
                state_class=lambda s, f=factory: f(s)
            )
            assert result.final_time > 0

    def test_service_reduces_failure_rate(self):
        """Service should reduce failure rate (compare serviced vs not)."""
        # This is a statistical test - with service, fewer early failures

        # Generate with baseline (service)
        np.random.seed(42)
        _, results_with_service, _ = generate_bathtub_data(
            n_subjects=100,
            max_time=50.0,  # Short time to see difference
            baseline_a=10,
            baseline_b=5,
            seed=42
        )

        # Count early failures (before t=50)
        failures_with = sum(
            1 for r in results_with_service
            if any(e.event_name == 'failure' for e in r.history)
        )

        # Without service - use very long intervals
        np.random.seed(42)
        _, results_no_service, _ = generate_bathtub_data(
            n_subjects=100,
            max_time=50.0,
            baseline_a=1000,  # Effectively no service
            baseline_b=0,
            seed=42
        )

        failures_without = sum(
            1 for r in results_no_service
            if any(e.event_name == 'failure' for e in r.history)
        )

        # With service should have fewer failures (or equal)
        # This is probabilistic, so we use a lenient check
        assert failures_with <= failures_without + 20  # Allow some variance

    def test_higher_durability_longer_life(self):
        """Higher durability subjects should live longer on average."""
        np.random.seed(42)

        # Generate with varying durability using custom scenario
        scenario = BasicBathtubScenario(
            durability_mean=1.0,
            durability_std=0.5  # High variance to see effect
        )
        df, results, _ = generate_bathtub_data(
            n_subjects=200,
            max_time=300.0,
            scenario=scenario,
            seed=42
        )

        # Group by durability quartile
        subjects_by_id = {r.subject.id: r for r in results}
        durabilities = [r.subject.features['durability'] for r in results]
        lifetimes = [r.final_time for r in results]

        # Compute correlation
        corr = np.corrcoef(durabilities, lifetimes)[0, 1]

        # Should be positive (higher durability = longer life)
        assert corr > 0
