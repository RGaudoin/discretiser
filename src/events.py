"""
Event type definitions and triggering rules for competing risks simulation.

An EventType defines:
- A survival model for time-to-event
- Optional triggering rules (what other events it can cause)
- Optional activation conditions (when this event can occur)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any, Dict

from .survival import SurvivalModel, PointMassPlusContinuous, NeverOccurs


@dataclass
class TriggerRule:
    """
    Rule for one event triggering another.

    When the source event occurs, the target event may be triggered
    according to the specified survival model (which may include
    a point mass at t=0 for simultaneous events).

    Attributes:
        target_event: Name of the event to trigger
        survival_model: How the triggered event's time is determined
        condition: Optional callable(state, subject) -> bool
        cancels: List of event names to remove from pending when this trigger fires
    """
    target_event: str
    survival_model: SurvivalModel
    condition: Optional[Callable[[Any, Any], bool]] = None
    cancels: List[str] = field(default_factory=list)

    def should_trigger(self, state: Any, subject: Any) -> bool:
        """Check if this trigger rule should fire."""
        if self.condition is None:
            return True
        return self.condition(state, subject)


@dataclass
class EventType:
    """
    Definition of an event type in the competing risks model.

    Attributes:
        name: Unique identifier for this event type
        survival_model: Base survival model for time-to-event
        triggers: List of events this event can trigger when it occurs
        cancels: List of event names to remove from pending when this occurs
        terminal: If True, this event ends the simulation (e.g., death)
        is_censoring: If True, this is a censoring event (observation ends)
        active_condition: Optional callable(state, subject) -> bool
                         If provided, event only competes when condition is True
        metadata: Optional dict for additional event-specific data
    """
    name: str
    survival_model: SurvivalModel
    triggers: List[TriggerRule] = field(default_factory=list)
    cancels: List[str] = field(default_factory=list)
    terminal: bool = False
    is_censoring: bool = False
    active_condition: Optional[Callable[[Any, Any], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self, state: Any, subject: Any) -> bool:
        """Check if this event type is currently active (can occur)."""
        if self.active_condition is None:
            return True
        return self.active_condition(state, subject)

    def sample_time(self, state: Any, subject: Any) -> float:
        """Sample time to this event occurring."""
        if not self.is_active(state, subject):
            return float('inf')
        return self.survival_model.sample(state, subject)

    def get_triggered_events(self, state: Any, subject: Any) -> List[tuple]:
        """
        Get list of events triggered by this event occurring.

        Returns:
            List of (event_name, time_delta, cancels) tuples.
            time_delta = 0 means simultaneous occurrence.
            cancels is a list of event names to remove from pending.
        """
        triggered = []
        for rule in self.triggers:
            if rule.should_trigger(state, subject):
                dt = rule.survival_model.sample(state, subject)
                triggered.append((rule.target_event, dt, rule.cancels))
        return triggered


# -----------------------------------------------------------------------------
# Factory functions for common event patterns
# -----------------------------------------------------------------------------

def make_censoring_event(
    name: str = "censoring",
    survival_model: Optional[SurvivalModel] = None
) -> EventType:
    """Create a censoring event (administrative end of observation)."""
    if survival_model is None:
        survival_model = NeverOccurs()  # Must be explicitly set
    return EventType(
        name=name,
        survival_model=survival_model,
        is_censoring=True,
        terminal=True
    )


def make_terminal_event(
    name: str,
    survival_model: SurvivalModel
) -> EventType:
    """Create a terminal event (e.g., death) that ends the simulation."""
    return EventType(
        name=name,
        survival_model=survival_model,
        terminal=True
    )


def make_triggering_event(
    name: str,
    survival_model: SurvivalModel,
    triggers: Dict[str, SurvivalModel],
    trigger_conditions: Optional[Dict[str, Callable]] = None
) -> EventType:
    """
    Create an event that can trigger other events.

    Args:
        name: Event name
        survival_model: Base survival model
        triggers: Dict mapping target event names to their survival models
        trigger_conditions: Optional dict mapping target names to condition functions

    Example:
        # Diagnosis triggers treatment (30% immediate, 70% within weeks)
        diagnosis = make_triggering_event(
            name="diagnosis",
            survival_model=Weibull(1.5, 30),
            triggers={
                "treatment": PointMassPlusContinuous(0.3, Weibull(1.2, 14))
            }
        )
    """
    trigger_conditions = trigger_conditions or {}
    rules = [
        TriggerRule(
            target_event=target,
            survival_model=model,
            condition=trigger_conditions.get(target)
        )
        for target, model in triggers.items()
    ]
    return EventType(
        name=name,
        survival_model=survival_model,
        triggers=rules
    )


def make_conditional_event(
    name: str,
    survival_model: SurvivalModel,
    condition: Callable[[Any, Any], bool]
) -> EventType:
    """
    Create an event that only becomes active under certain conditions.

    Example:
        # Relapse only possible after treatment
        relapse = make_conditional_event(
            name="relapse",
            survival_model=Weibull(0.8, 365),
            condition=lambda state, _: "treatment" in state.occurred_events
        )
    """
    return EventType(
        name=name,
        survival_model=survival_model,
        active_condition=condition
    )


# -----------------------------------------------------------------------------
# Event Registry
# -----------------------------------------------------------------------------

class EventRegistry:
    """
    Registry of all event types in a simulation.

    Provides lookup by name and validation of trigger references.
    """

    def __init__(self):
        self._events: Dict[str, EventType] = {}

    def register(self, event: EventType) -> None:
        """Register an event type."""
        if event.name in self._events:
            raise ValueError(f"Event '{event.name}' already registered")
        self._events[event.name] = event

    def register_all(self, events: List[EventType]) -> None:
        """Register multiple event types."""
        for event in events:
            self.register(event)

    def get(self, name: str) -> EventType:
        """Get event type by name."""
        if name not in self._events:
            raise KeyError(f"Event '{name}' not found in registry")
        return self._events[name]

    def __getitem__(self, name: str) -> EventType:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._events

    def __iter__(self):
        return iter(self._events.values())

    def __len__(self) -> int:
        return len(self._events)

    @property
    def names(self) -> List[str]:
        """List of all registered event names."""
        return list(self._events.keys())

    def validate(self) -> List[str]:
        """
        Validate that all trigger references point to registered events.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        for event in self._events.values():
            for rule in event.triggers:
                if rule.target_event not in self._events:
                    errors.append(
                        f"Event '{event.name}' triggers unknown event '{rule.target_event}'"
                    )
        return errors

    def get_active_events(self, state: Any, subject: Any) -> List[EventType]:
        """Get list of currently active (competing) events."""
        return [e for e in self._events.values() if e.is_active(state, subject)]
