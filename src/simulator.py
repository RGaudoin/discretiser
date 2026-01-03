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

    Modes:
        'competing': (default) Pending events persist until they win or are cancelled.
                     Events that don't win continue to compete in future rounds.

        'autoregressive': Pending events get one chance to compete. After each event,
                          all pending is cleared. Triggers add fresh pending for the
                          next round only. More natural for sequence models that
                          predict "next event | history".
    """

    VALID_MODES = {'competing', 'autoregressive'}

    def __init__(
        self,
        event_registry: EventRegistry,
        max_time: float = float('inf'),
        max_events: int = 1000,
        mode: str = 'competing'
    ):
        """
        Args:
            event_registry: Registry of event types
            max_time: Maximum simulation time
            max_events: Maximum events per subject (safety limit)
            mode: 'competing' (default) or 'autoregressive'
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.registry = event_registry
        self.max_time = max_time
        self.max_events = max_events
        self.mode = mode

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

            # Process cancellations - remove specified events from pending
            for cancelled_event in winner_event.cancels:
                state.pop_pending_event(cancelled_event)

            # Clear all pending if event requests it or in autoregressive mode
            if winner_event.clears_all_pending or self.mode == 'autoregressive':
                state.clear_pending_events()

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
        - Trigger-level cancellations are processed when the trigger fires
        """
        triggered = source_event.get_triggered_events(state, subject)

        for target_name, dt, trigger_cancels in triggered:
            if target_name not in self.registry:
                continue

            # Process trigger-level cancellations
            for cancelled_event in trigger_cancels:
                state.pop_pending_event(cancelled_event)

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


# -----------------------------------------------------------------------------
# Training data extraction
# -----------------------------------------------------------------------------

def extract_training_data(
    df: pd.DataFrame,
    event_names: List[str],
    target_event: str,
    include_censored: bool = True
) -> pd.DataFrame:
    """
    Extract training data from simulation output.

    For each subject, computes features at each event and the time to
    the target event (or censoring). Suitable for supervised learning
    of survival models.

    Args:
        df: DataFrame from simulate_cohort_simple or cohort_to_dataframe
            Must have columns: subject_id, event, time
        event_names: Events to use for feature extraction (counts, time-since)
        target_event: The event we're trying to predict time-to
        include_censored: Whether to include censored observations

    Returns:
        DataFrame with columns:
        - subject_id
        - observation_time: Time at which features are observed
        - {event}_count: Count of each event up to observation_time
        - {event}_time_since: Time since last occurrence (NaN if never)
        - time_to_target: Time from observation to target event
        - censored: Whether observation is censored (target not observed)

    Example:
        df = simulate_cohort_simple(n_subjects=1000, events=events, max_time=365)
        training_df = extract_training_data(df, ['diagnosis', 'treatment'], 'outcome')
    """
    records = []

    for subject_id, subject_df in df.groupby('subject_id'):
        subject_df = subject_df.sort_values('time')

        # Find target event time (if any)
        target_rows = subject_df[subject_df['event'] == target_event]
        if len(target_rows) > 0:
            target_time = target_rows['time'].iloc[0]
            is_censored = False
        else:
            # Subject didn't experience target event - use max observed time
            target_time = subject_df['time'].max()
            is_censored = True

        if is_censored and not include_censored:
            continue

        # Build features at each observation point (each event)
        event_counts = {name: 0 for name in event_names}
        last_event_times = {name: np.nan for name in event_names}

        for _, row in subject_df.iterrows():
            obs_time = row['time']
            event = row['event']

            # Skip if this is the target event or after it
            if event == target_event or obs_time >= target_time:
                break

            # Update counts for observed event
            if event in event_names:
                event_counts[event] += 1
                last_event_times[event] = obs_time

            # Create feature record at this observation point
            record = {
                'subject_id': subject_id,
                'observation_time': obs_time,
                'time_to_target': target_time - obs_time,
                'censored': is_censored,
            }

            # Add event counts
            for name in event_names:
                record[f'{name}_count'] = event_counts[name]

            # Add time-since features
            for name in event_names:
                if np.isnan(last_event_times[name]):
                    record[f'{name}_time_since'] = np.nan
                else:
                    record[f'{name}_time_since'] = obs_time - last_event_times[name]

            records.append(record)

        # Also create a baseline record at t=0 if subject has events before target
        if len(subject_df[subject_df['time'] < target_time]) > 0:
            baseline = {
                'subject_id': subject_id,
                'observation_time': 0.0,
                'time_to_target': target_time,
                'censored': is_censored,
            }
            for name in event_names:
                baseline[f'{name}_count'] = 0
                baseline[f'{name}_time_since'] = np.nan
            records.insert(0, baseline)  # Add at beginning

    return pd.DataFrame(records)
