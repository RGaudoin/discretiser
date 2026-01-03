# Simulation Framework

This document explains how events, triggers, state, and the simulation loop fit together.

## Overview

The framework simulates **competing risks** - multiple events compete to occur, and the one with the shortest time wins. When an event occurs, it may trigger other events (immediately or after a delay), and the state is updated.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Simulation Loop                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Subject ──► State ──► Active Events ──► Sample Times          │
│     │          │              │                │                │
│     │          │              ▼                ▼                │
│     │          │        ┌──────────┐    ┌──────────────┐       │
│     │          │        │ Pending  │    │  Competing   │       │
│     │          │        │ Triggers │    │   Risks      │       │
│     │          │        └────┬─────┘    └──────┬───────┘       │
│     │          │             │                 │                │
│     │          │             └────────┬────────┘                │
│     │          │                      ▼                         │
│     │          │              Winner (min time)                 │
│     │          │                      │                         │
│     │          ◄──────────────────────┤                         │
│     │          │              Update State                      │
│     │          │                      │                         │
│     │          │              Process Triggers                  │
│     │          │                      │                         │
│     │          │              ┌───────┴───────┐                 │
│     │          │              ▼               ▼                 │
│     │          │         Immediate       Scheduled              │
│     │          │         (record)      (add pending)            │
│     │          │                                                │
│     │          └────────────► Repeat until terminal/max_time    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### Subject

Static features that don't change during simulation. Think of this as patient demographics or baseline characteristics.

```python
subject = Subject(
    id="patient_001",
    features={"age": 65, "sex": "M", "risk_score": 0.7},
    feature_vector=np.array([0.65, 1.0, 0.7])  # For ML models
)
```

### State

Accumulated history and derived features. Updates as events occur.

```python
state.time                          # Current simulation time
state.event_count('treatment')      # How many treatments so far
state.time_since_last('diagnosis')  # Time since last diagnosis
state.occurred_events               # Set of event types that have occurred
state.to_feature_vector(events)     # Numerical features for ML
```

The `EmbeddingState` variant adds decay-weighted event embeddings for richer state representation.

### EventType

Defines what can happen and when.

```python
EventType(
    name='treatment',
    survival_model=Weibull(shape=1.2, scale=14),  # Time distribution
    triggers=[...],           # What this event can cause
    terminal=False,           # Does this end the simulation?
    is_censoring=False,       # Is this administrative censoring?
    active_condition=None     # When can this event occur?
)
```

### Survival Model

Determines **when** an event occurs. See [survival_models.md](survival_models.md) for available models.

### TriggerRule

Defines causal relationships between events.

```python
TriggerRule(
    target_event='side_effect',
    survival_model=PointMassPlusContinuous(p0=0.3, continuous=Weibull(1.5, 7)),
    condition=lambda state, subject: state.event_count('treatment') < 3
)
```

## The Simulation Loop

### Step 1: Get Active Events

Not all events compete at all times. An event is active if:
- Its `active_condition` returns True (or is None)
- Example: "relapse" only active after "treatment" has occurred

```python
active_events = registry.get_active_events(state, subject)
```

### Step 2: Sample Times

For each active event, sample time-to-event from its survival model:

```python
for event in active_events:
    dt = event.survival_model.sample(state, subject)
    times[event.name] = state.time + dt
```

**State-dependent models** can use the current state to adjust parameters (e.g., hazard increases with number of prior events).

### Step 3: Include Pending Triggers

Triggered events that were scheduled for the future also compete:

```python
for event_name, (scheduled_time, source) in state.get_pending_events().items():
    if scheduled_time < times.get(event_name, inf):
        times[event_name] = scheduled_time
```

### Step 4: Winner Takes All

The event with minimum time wins:

```python
winner_name = min(times, key=times.get)
winner_time = times[winner_name]
```

### Step 5: Update State

```python
state.advance_time(winner_time)
state.record_event(winner_name, winner_time, triggered_by=...)
```

### Step 6: Process Triggers

When an event occurs, check its trigger rules:

```python
for rule in winner_event.triggers:
    if rule.should_trigger(state, subject):
        dt = rule.survival_model.sample(state, subject)

        if dt == 0:
            # Immediate: record now
            state.record_event(target, state.time, triggered_by=winner)
        else:
            # Delayed: add to pending, will compete with other events
            state.add_pending_event(target, state.time + dt, triggered_by=winner)
```

### Step 7: Repeat or Terminate

Continue until:
- Terminal event occurs (death, cure, etc.)
- Censoring event occurs (end of observation)
- `max_time` reached
- `max_events` safety limit hit

## Triggering Patterns

### Immediate Triggers (dt = 0)

Event B happens at the exact same time as event A.

```python
# When diagnosed, 30% chance a comorbidity is discovered simultaneously
diagnosis = make_triggering_event(
    name='diagnosis',
    survival_model=Weibull(1.5, 30),
    triggers={
        'comorbidity': PointMassPlusContinuous(
            p0=0.3,  # 30% immediate
            continuous=NeverOccurs()  # Never happens later
        )
    }
)
```

### Delayed Triggers (dt > 0)

Event B is scheduled to happen after event A, but competes with other events.

```python
# Treatment triggers possible side effect within days
treatment = make_triggering_event(
    name='treatment',
    survival_model=...,
    triggers={
        'side_effect': Weibull(shape=2, scale=7)  # Peaks around day 7
    }
)
```

**Important**: Delayed triggers enter the competition. Another event (e.g., death) might occur first, cancelling the pending trigger.

### Conditional Triggers

Triggers can have conditions:

```python
TriggerRule(
    target_event='relapse',
    survival_model=Weibull(0.8, 180),
    condition=lambda state, _: state.event_count('treatment') >= 2
)
```

### Cancelling Pending Events

When an event occurs, it can cancel (remove) other events from the pending list. Both `EventType` and `TriggerRule` support cancellations.

**EventType.cancels** - unconditional, fires whenever the event occurs:

```python
# New diagnosis always cancels scheduled routine checkups
EventType(
    name='new_diagnosis',
    survival_model=Weibull(1.5, 60),
    cancels=['routine_checkup', 'annual_screening']
)
```

**TriggerRule.cancels** - conditional, only fires when the trigger fires:

```python
# Treatment only cancels disease_progression if it triggers remission
EventType(
    name='treatment',
    survival_model=Weibull(1.2, 14),
    triggers=[
        TriggerRule(
            target_event='remission',
            survival_model=PointMassPlusContinuous(0.3, Weibull(2, 30)),
            cancels=['disease_progression', 'symptom_flare']
        )
    ]
)
```

**Processing order:**
1. Event occurs → EventType.cancels processed
2. Triggers checked → TriggerRule.cancels processed for each firing trigger

**Note**: Terminal events automatically clear all pending events.

## Event Types by Role

### Regular Events

Occur, update state, possibly trigger others, simulation continues.

```python
EventType('treatment', Weibull(1.2, 14))
```

### Terminal Events

End the simulation when they occur.

```python
EventType('death', Weibull(2, 365), terminal=True)
EventType('cure', Exponential(0.01), terminal=True)
```

### Censoring Events

Special terminal events marking end of observation (not a real outcome).

```python
make_censoring_event('admin_censoring', Exponential(rate=1/365))
```

### Conditional Events

Only compete when a condition is met.

```python
make_conditional_event(
    'relapse',
    Weibull(0.8, 365),
    condition=lambda state, _: 'treatment' in state.occurred_events
)
```

## Complete Example

```python
from discretiser.src import (
    EventType, TriggerRule, EventRegistry, Simulator,
    Subject, State,
    Weibull, Exponential, PointMassPlusContinuous, NeverOccurs,
    make_censoring_event, make_conditional_event
)

# Define events
events = [
    # Diagnosis happens first
    EventType(
        name='diagnosis',
        survival_model=Weibull(1.5, 30),
        triggers=[
            TriggerRule(
                target_event='treatment',
                survival_model=PointMassPlusContinuous(
                    p0=0.2,  # 20% immediate treatment
                    continuous=Weibull(1.2, 14)  # Rest within ~2 weeks
                )
            )
        ]
    ),

    # Treatment (only occurs if triggered by diagnosis)
    EventType(
        name='treatment',
        survival_model=NeverOccurs(),  # Only via trigger
        triggers=[
            TriggerRule(
                target_event='outcome',
                survival_model=Weibull(2, 90)
            )
        ]
    ),

    # Outcome
    EventType('outcome', survival_model=NeverOccurs(), terminal=True),

    # Censoring
    make_censoring_event('censoring', Exponential(rate=1/365))
]

# Setup and run
registry = EventRegistry()
registry.register_all(events)

simulator = Simulator(registry, max_time=730)

subject = Subject(id='patient_001')
result = simulator.simulate_subject(subject)

# Result contains event history
for event in result.history:
    print(f"t={event.time:.1f}: {event.event_name} (triggered by: {event.triggered_by})")
```

## Simulation Modes

The simulator supports two modes for handling pending events:

### Competing Mode (default)

Pending events persist until they win or are explicitly cancelled. Events that don't win continue competing in future rounds.

```python
simulator = Simulator(registry, max_time=730, mode='competing')
```

**Behaviour:**
- Triggered event scheduled at t=50
- Another event wins at t=30
- Triggered event still pending, competes again next round

### Autoregressive Mode

Pending events get one chance to compete. After each event, all remaining pending events are cleared. Triggers add fresh pending for the next round only.

```python
simulator = Simulator(registry, max_time=730, mode='autoregressive')
```

**Behaviour:**
- Triggered event scheduled at t=50
- Another event wins at t=30
- Triggered event is cleared (didn't win this round)
- Only newly triggered events compete next round

**Use cases:**
- Sequence models that predict "next event | history"
- Transformer/RNN architectures
- Simpler state representation (no persistent pending to track)

## State-Dependent Survival

Survival models receive `(state, subject)` and can adapt their behaviour:

```python
class TreatmentDependentOutcome(Weibull):
    """Each treatment improves survival by 20%."""

    def sample(self, state=None, subject=None):
        n_tx = state.event_count('treatment') if state else 0
        adjusted_scale = self.scale * (1.2 ** n_tx)  # Longer survival
        return adjusted_scale * np.random.weibull(self.shape)
```

See [trained_model_integration.md](trained_model_integration.md) for using neural networks to predict survival parameters.
