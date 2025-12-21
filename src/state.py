"""
State representation for competing risks simulation.

The State tracks:
- Current simulation time
- Event history (what happened and when)
- Derived features (counts, time-since-last, embeddings)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
import numpy as np


@dataclass
class EventRecord:
    """Record of a single event occurrence."""
    event_name: str
    time: float
    triggered_by: Optional[str] = None  # which event caused this (if any)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Subject:
    """
    Subject (e.g., patient) with static features.

    Attributes:
        id: Unique identifier
        features: Dict of feature name -> value
        feature_vector: Optional numpy array representation
    """
    id: Any
    features: Dict[str, Any] = field(default_factory=dict)
    feature_vector: Optional[np.ndarray] = None

    def get_feature(self, name: str, default: Any = None) -> Any:
        """Get a feature value by name."""
        return self.features.get(name, default)


class State:
    """
    Simulation state for a single subject.

    Tracks event history and provides derived features.
    """

    def __init__(self, subject: Subject, start_time: float = 0.0):
        self.subject = subject
        self.time = start_time
        self.history: List[EventRecord] = []
        self._occurred_events: Set[str] = set()
        self._event_counts: Dict[str, int] = {}
        self._last_event_time: Dict[str, float] = {}
        self.terminated = False
        self.censored = False
        # Pending triggered events: event_name -> (absolute_time, triggered_by)
        self._pending_events: Dict[str, tuple] = {}

    @property
    def occurred_events(self) -> Set[str]:
        """Set of event names that have occurred."""
        return self._occurred_events

    def event_count(self, event_name: str) -> int:
        """Number of times an event has occurred."""
        return self._event_counts.get(event_name, 0)

    def time_since_last(self, event_name: str) -> Optional[float]:
        """Time since last occurrence of an event (None if never occurred)."""
        if event_name not in self._last_event_time:
            return None
        return self.time - self._last_event_time[event_name]

    def time_since_start(self) -> float:
        """Total time elapsed since simulation start."""
        return self.time

    def record_event(
        self,
        event_name: str,
        time: float,
        triggered_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an event occurrence and update derived features.

        Args:
            event_name: Name of the event
            time: Absolute time of occurrence
            triggered_by: Name of event that triggered this (if any)
            metadata: Optional additional data
        """
        record = EventRecord(
            event_name=event_name,
            time=time,
            triggered_by=triggered_by,
            metadata=metadata or {}
        )
        self.history.append(record)

        # Update derived features
        self._occurred_events.add(event_name)
        self._event_counts[event_name] = self._event_counts.get(event_name, 0) + 1
        self._last_event_time[event_name] = time

    def advance_time(self, new_time: float) -> None:
        """Advance simulation time."""
        if new_time < self.time:
            raise ValueError(f"Cannot go backwards in time: {new_time} < {self.time}")
        self.time = new_time

    def mark_terminated(self, is_censoring: bool = False) -> None:
        """Mark the simulation as terminated."""
        self.terminated = True
        self.censored = is_censoring

    def get_last_n_events(self, n: int) -> List[EventRecord]:
        """Get the last n events (or all if fewer)."""
        return self.history[-n:]

    def get_events_since(self, time: float) -> List[EventRecord]:
        """Get all events after a given time."""
        return [e for e in self.history if e.time > time]

    def get_events_of_type(self, event_name: str) -> List[EventRecord]:
        """Get all occurrences of a specific event type."""
        return [e for e in self.history if e.event_name == event_name]

    # -------------------------------------------------------------------------
    # Pending triggered events (compete with other events)
    # -------------------------------------------------------------------------

    def add_pending_event(
        self,
        event_name: str,
        scheduled_time: float,
        triggered_by: str
    ) -> None:
        """
        Schedule a triggered event to compete with other events.

        If the same event is already pending, keeps the earlier time.
        """
        if event_name in self._pending_events:
            existing_time, _ = self._pending_events[event_name]
            if scheduled_time >= existing_time:
                return  # Keep the earlier scheduled time
        self._pending_events[event_name] = (scheduled_time, triggered_by)

    def get_pending_events(self) -> Dict[str, tuple]:
        """Get all pending triggered events."""
        return self._pending_events.copy()

    def pop_pending_event(self, event_name: str) -> Optional[tuple]:
        """Remove and return a pending event (time, triggered_by)."""
        return self._pending_events.pop(event_name, None)

    def clear_pending_events(self) -> None:
        """Clear all pending events (e.g., after terminal event)."""
        self._pending_events.clear()

    # -------------------------------------------------------------------------
    # Feature extraction for state-dependent survival models
    # -------------------------------------------------------------------------

    def to_feature_dict(self) -> Dict[str, Any]:
        """
        Extract features from state for use in survival models.

        Returns dict with:
        - time: current simulation time
        - event_counts: dict of event name -> count
        - time_since_last: dict of event name -> time since last (or None)
        - subject_features: subject's feature dict
        """
        return {
            "time": self.time,
            "event_counts": dict(self._event_counts),
            "time_since_last": {
                name: self.time_since_last(name)
                for name in self._occurred_events
            },
            "subject_features": self.subject.features,
            "n_events": len(self.history),
        }

    def to_feature_vector(
        self,
        event_names: List[str],
        include_subject: bool = True
    ) -> np.ndarray:
        """
        Convert state to numerical feature vector.

        Args:
            event_names: List of event names to include counts/times for
            include_subject: Whether to include subject feature vector

        Returns:
            Numpy array suitable for model input
        """
        features = []

        # Current time
        features.append(self.time)

        # Event counts (one per event type)
        for name in event_names:
            features.append(float(self.event_count(name)))

        # Time since last (0 if never occurred)
        for name in event_names:
            t = self.time_since_last(name)
            features.append(t if t is not None else 0.0)

        # Subject features
        if include_subject and self.subject.feature_vector is not None:
            features.extend(self.subject.feature_vector.flatten())

        return np.array(features, dtype=np.float32)


# -----------------------------------------------------------------------------
# State with decay-weighted event embeddings
# -----------------------------------------------------------------------------

class EmbeddingState(State):
    """
    State that maintains decay-weighted event embeddings.

    Each event type has an embedding vector. The state maintains a
    weighted sum of embeddings from past events, with exponential
    decay based on time since each event.
    """

    def __init__(
        self,
        subject: Subject,
        event_embeddings: Dict[str, np.ndarray],
        decay_rates: Dict[str, float],
        start_time: float = 0.0
    ):
        """
        Args:
            subject: The subject
            event_embeddings: Dict mapping event name -> embedding vector
            decay_rates: Dict mapping event name -> decay rate (1/timescale)
            start_time: Initial simulation time
        """
        super().__init__(subject, start_time)
        self.event_embeddings = event_embeddings
        self.decay_rates = decay_rates

        # Determine embedding dimension from first embedding
        if event_embeddings:
            first = next(iter(event_embeddings.values()))
            self.embedding_dim = len(first)
        else:
            self.embedding_dim = 0

    def get_current_embedding(self) -> np.ndarray:
        """
        Compute current embedding as decay-weighted sum of past events.

        Each past event contributes:
            weight = exp(-decay_rate * time_since_event)
        """
        if self.embedding_dim == 0:
            return np.array([])

        embedding = np.zeros(self.embedding_dim)

        for record in self.history:
            if record.event_name in self.event_embeddings:
                emb = self.event_embeddings[record.event_name]
                decay = self.decay_rates.get(record.event_name, 1.0)
                dt = self.time - record.time
                weight = np.exp(-decay * dt)
                embedding += weight * emb

        return embedding

    def to_feature_vector(
        self,
        event_names: List[str],
        include_subject: bool = True,
        include_embedding: bool = True
    ) -> np.ndarray:
        """Extended to include decay-weighted embedding."""
        base = super().to_feature_vector(event_names, include_subject)

        if include_embedding:
            emb = self.get_current_embedding()
            return np.concatenate([base, emb])

        return base


# -----------------------------------------------------------------------------
# Subject generators
# -----------------------------------------------------------------------------

def generate_subjects(
    n: int,
    feature_dim: int = 4,
    dirichlet_alpha: float = 0.5,
    id_prefix: str = "subject"
) -> List[Subject]:
    """
    Generate subjects with Dirichlet-distributed feature vectors.

    Args:
        n: Number of subjects
        feature_dim: Dimension of feature vector
        dirichlet_alpha: Concentration parameter (< 1 for sparse)
        id_prefix: Prefix for subject IDs

    Returns:
        List of Subject objects
    """
    feature_vectors = np.random.dirichlet(
        [dirichlet_alpha] * feature_dim,
        size=n
    )

    subjects = []
    for i, fv in enumerate(feature_vectors):
        subjects.append(Subject(
            id=f"{id_prefix}_{i}",
            feature_vector=fv,
            features={f"f{j}": fv[j] for j in range(feature_dim)}
        ))

    return subjects
