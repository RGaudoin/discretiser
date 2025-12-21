"""
Main competing risks simulation loop.

Simulates subject journeys by:
1. Sampling time-to-event for all active competing risks
2. Selecting the winner (minimum time)
3. Updating state and processing triggered events
4. Repeating until termination or max time
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
import numpy as np

from .survival import SurvivalModel
from .events import EventType, EventRegistry
from .state import Subject, State, EventRecord


@dataclass
class SimulationResult:
    """Result of simulating a single subject."""
    subject: Subject
    history: List[EventRecord]
    final_time: float
    terminated: bool
    censored: bool

    def to_dataframe(self) -> pd.DataFrame:
        """Convert event history to DataFrame."""
        if not self.history:
            return pd.DataFrame(columns=[
                "subject_id", "event", "time", "triggered_by"
            ])

        records = []
        for event in self.history:
            records.append({
                "subject_id": self.subject.id,
                "event": event.event_name,
                "time": event.time,
                "triggered_by": event.triggered_by,
                **event.metadata
            })
        return pd.DataFrame(records)


class Simulator:
    """
    Competing risks simulator.

    Simulates event sequences for subjects according to registered
    event types and their survival models.
    """

    def __init__(
        self,
        event_registry: EventRegistry,
        max_time: float = float('inf'),
        max_events: int = 1000
    ):
        """
        Args:
            event_registry: Registry of event types
            max_time: Maximum simulation time
            max_events: Maximum events per subject (safety limit)
        """
        self.registry = event_registry
        self.max_time = max_time
        self.max_events = max_events

        # Validate registry
        errors = self.registry.validate()
        if errors:
            raise ValueError(f"Invalid event registry: {errors}")

    def simulate_subject(
        self,
        subject: Subject,
        initial_events: Optional[List[Tuple[str, float]]] = None,
        state_class: type = State
    ) -> SimulationResult:
        """
        Simulate a single subject's journey.

        Args:
            subject: The subject to simulate
            initial_events: Optional list of (event_name, time) to seed history
            state_class: State class to use (State or EmbeddingState)

        Returns:
            SimulationResult with event history
        """
        state = state_class(subject)

        # Apply initial events
        if initial_events:
            for event_name, time in initial_events:
                state.record_event(event_name, time)
                state.advance_time(time)

        event_count = 0

        while not state.terminated and state.time < self.max_time:
            # Safety limit
            event_count += 1
            if event_count > self.max_events:
                break

            # Get active events and sample times
            active_events = self.registry.get_active_events(state, subject)
            if not active_events:
                break

            # Sample time to each competing event
            times: Dict[str, float] = {}
            triggered_by: Dict[str, str] = {}  # Track which events are pending triggers

            for event in active_events:
                dt = event.sample_time(state, subject)
                if dt < float('inf'):
                    times[event.name] = state.time + dt

            # Include pending triggered events in competition
            for event_name, (scheduled_time, source) in state.get_pending_events().items():
                # Pending event competes if its time is still in the future
                if scheduled_time > state.time:
                    # Only include if no sampled time or pending is earlier
                    if event_name not in times or scheduled_time < times[event_name]:
                        times[event_name] = scheduled_time
                        triggered_by[event_name] = source

            if not times:
                break

            # Find winner (minimum time)
            winner_name = min(times, key=times.get)
            winner_time = times[winner_name]

            # Check max time
            if winner_time >= self.max_time:
                break

            # Get the winning event type
            winner_event = self.registry[winner_name]

            # Check if winner was a pending triggered event
            winner_triggered_by = triggered_by.get(winner_name)
            if winner_triggered_by:
                state.pop_pending_event(winner_name)

            # Advance time and record event
            state.advance_time(winner_time)
            state.record_event(winner_name, winner_time, triggered_by=winner_triggered_by)

            # Handle termination
            if winner_event.terminal:
                state.mark_terminated(is_censoring=winner_event.is_censoring)
                state.clear_pending_events()  # Cancel any pending triggered events
                break

            # Process triggered events
            self._process_triggers(state, subject, winner_event)

        return SimulationResult(
            subject=subject,
            history=state.history.copy(),
            final_time=state.time,
            terminated=state.terminated,
            censored=state.censored
        )

    def _process_triggers(
        self,
        state: State,
        subject: Subject,
        source_event: EventType
    ) -> None:
        """
        Process events triggered by the source event.

        - Simultaneous events (dt=0) are recorded immediately
        - Future events (dt>0) are added to pending and compete with other events
        """
        triggered = source_event.get_triggered_events(state, subject)

        for target_name, dt in triggered:
            if target_name not in self.registry:
                continue

            target_time = state.time + dt

            # Only process if within max_time
            if target_time >= self.max_time:
                continue

            if dt == 0:
                # Simultaneous event - record immediately at current time
                state.record_event(
                    target_name,
                    state.time,
                    triggered_by=source_event.name
                )

                # Check if triggered event is terminal
                target_event = self.registry[target_name]
                if target_event.terminal:
                    state.mark_terminated(is_censoring=target_event.is_censoring)
                    state.clear_pending_events()
                    return
            else:
                # Future event - add to pending to compete with other events
                state.add_pending_event(
                    target_name,
                    target_time,
                    triggered_by=source_event.name
                )

    def simulate_cohort(
        self,
        subjects: List[Subject],
        initial_events: Optional[List[Tuple[str, float]]] = None,
        state_class: type = State
    ) -> List[SimulationResult]:
        """
        Simulate a cohort of subjects.

        Args:
            subjects: List of subjects to simulate
            initial_events: Optional initial events (applied to all)
            state_class: State class to use

        Returns:
            List of SimulationResult objects
        """
        results = []
        for subject in subjects:
            result = self.simulate_subject(
                subject,
                initial_events=initial_events,
                state_class=state_class
            )
            results.append(result)
        return results

    def cohort_to_dataframe(
        self,
        results: List[SimulationResult]
    ) -> pd.DataFrame:
        """Convert cohort results to a single DataFrame."""
        dfs = [r.to_dataframe() for r in results]
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------

def simulate_journey(
    subject: Subject,
    events: List[EventType],
    max_time: float = 365.0,
    initial_events: Optional[List[Tuple[str, float]]] = None
) -> SimulationResult:
    """
    Convenience function for simple simulations.

    Args:
        subject: Subject to simulate
        events: List of event types
        max_time: Maximum simulation time
        initial_events: Optional initial events

    Returns:
        SimulationResult
    """
    registry = EventRegistry()
    registry.register_all(events)

    simulator = Simulator(registry, max_time=max_time)
    return simulator.simulate_subject(subject, initial_events=initial_events)


def simulate_cohort_simple(
    n_subjects: int,
    events: List[EventType],
    max_time: float = 365.0,
    subject_dim: int = 4
) -> pd.DataFrame:
    """
    Convenience function for cohort simulation with generated subjects.

    Args:
        n_subjects: Number of subjects
        events: List of event types
        max_time: Maximum simulation time
        subject_dim: Dimension of subject feature vectors

    Returns:
        DataFrame with all events
    """
    from .state import generate_subjects

    subjects = generate_subjects(n_subjects, feature_dim=subject_dim)

    registry = EventRegistry()
    registry.register_all(events)

    simulator = Simulator(registry, max_time=max_time)
    results = simulator.simulate_cohort(subjects)

    return simulator.cohort_to_dataframe(results)
