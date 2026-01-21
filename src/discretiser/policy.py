"""
Policy definitions for decision-making in simulations.

Policies are separate from scenarios - same policy can be used with
ground truth or learned model scenarios.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from dataclasses import dataclass

from .state import State, Subject


@dataclass
class Action:
    """An action to take in the simulation."""
    event_name: str  # Which event to schedule
    delay: float     # Time from now until event occurs


class Policy(ABC):
    """
    Abstract base class for policies.

    A policy decides what action to take given current state and subject.
    """

    @abstractmethod
    def get_action(self, state: State, subject: Subject) -> Optional[Action]:
        """
        Decide what action to take.

        Args:
            state: Current simulation state
            subject: The subject being simulated

        Returns:
            Action to take, or None if no action
        """
        pass

    def reset(self) -> None:
        """Reset any internal state (called at start of each episode)."""
        pass


class NoOpPolicy(Policy):
    """Policy that never takes any action."""

    def get_action(self, state: State, subject: Subject) -> Optional[Action]:
        return None


class LinearIntervalPolicy(Policy):
    """
    Schedule events at intervals linear in a subject feature.

    interval = a + b * feature_value

    Used for baseline service scheduling in bathtub scenario.
    """

    def __init__(
        self,
        a: float,
        b: float,
        event_name: str = 'service',
        feature: str = 'durability',
        default_feature_value: float = 1.0
    ):
        """
        Args:
            a: Base interval (intercept)
            b: Feature coefficient (slope)
            event_name: Event to schedule
            feature: Subject feature to use
            default_feature_value: Default if feature missing
        """
        self.a = a
        self.b = b
        self.event_name = event_name
        self.feature = feature
        self.default_feature_value = default_feature_value
        self._next_scheduled: Optional[float] = None

    def get_interval(self, subject: Subject) -> float:
        """Calculate interval for this subject."""
        feature_value = subject.get_feature(self.feature, self.default_feature_value)
        return self.a + self.b * feature_value

    def get_action(self, state: State, subject: Subject) -> Optional[Action]:
        """
        Schedule next event if none pending.

        Schedules at regular intervals determined by subject features.
        """
        # Check if we already have this event pending
        pending = state.get_pending_events()
        if self.event_name in pending:
            return None

        # Schedule next event
        interval = self.get_interval(subject)
        return Action(event_name=self.event_name, delay=interval)

    def reset(self) -> None:
        self._next_scheduled = None

    @property
    def params(self) -> tuple:
        """Return policy parameters as tuple."""
        return (self.a, self.b)

    @classmethod
    def from_params(cls, params: tuple, **kwargs) -> 'LinearIntervalPolicy':
        """Create policy from parameter tuple."""
        return cls(a=params[0], b=params[1], **kwargs)


class FixedIntervalPolicy(Policy):
    """
    Schedule events at fixed intervals (ignores subject features).

    Simpler than LinearIntervalPolicy - useful for baselines.
    """

    def __init__(self, interval: float, event_name: str = 'service'):
        self.interval = interval
        self.event_name = event_name

    def get_action(self, state: State, subject: Subject) -> Optional[Action]:
        pending = state.get_pending_events()
        if self.event_name in pending:
            return None
        return Action(event_name=self.event_name, delay=self.interval)

    @property
    def params(self) -> tuple:
        return (self.interval,)

    @classmethod
    def from_params(cls, params: tuple, **kwargs) -> 'FixedIntervalPolicy':
        return cls(interval=params[0], **kwargs)
