# Synthetic Scenarios for Model Validation

Design notes for synthetic data generators with known ground-truth dynamics.

## Purpose

Build synthetic scenarios where:
1. Ground truth dynamics are fully specified and known
2. Data can be generated for training generative models
3. RL can optimise policies on both ground truth and learned models
4. Policies can be validated by comparing learned-model performance to ground-truth performance

## Validation Loop

```
Ground Truth Simulator
        │
        ▼
   Generate Data ──────────────────────────┐
        │                                  │
        ▼                                  │
   Train Generative Model                  │
        │                                  │
        ▼                                  │
   RL on Learned Model ───► Policy A       │
        │                                  │
        │                     Compare      │
        │                       │          │
        ▼                       ▼          │
   RL on Ground Truth ────► Policy B       │
        │                       │          │
        ▼                       ▼          │
   Evaluate Both Policies on Ground Truth ◄┘
```

**Key metrics:**
- Does learned model recover true dynamics? (distribution matching)
- Does Policy A (from learned) perform similarly to Policy B (from truth)?
- How much does the learned model's imperfections degrade policy quality?

---

## Scenario: Widget Maintenance

A neutral domain (avoiding medical/clinical complexity) that captures the core dynamics of interest.

### Events

| Event | Type | Description |
|-------|------|-------------|
| `degradation` | Recurring | Widget quality decreases |
| `quick_fix` | Action | Cheap, short-term maintenance |
| `full_service` | Action | Expensive, long-term maintenance |
| `failure` | Terminal | Widget fails, high cost |

### Ground Truth Dynamics

#### Failure Hazard

Failure hazard depends on:
- **Time since last maintenance** (either type)
- **Degradation count** since last full service
- **Type of last maintenance** (quick fix provides less protection)

```python
def failure_hazard_modifier(state, subject):
    """
    Failure hazard increases with:
    - Time since any maintenance (quick_fix or full_service)
    - Accumulated degradations since last full_service
    """
    # Time since last maintenance (any type)
    t_quick = state.time_since('quick_fix') or float('inf')
    t_full = state.time_since('full_service') or float('inf')
    t_since_maint = min(t_quick, t_full)

    # Degradations since last full service
    # (quick_fix doesn't reset degradation effects)
    degradations = state.events_since('degradation', 'full_service')

    # Hazard multiplier
    time_factor = 1.0 + 0.02 * t_since_maint  # Linear increase
    degrad_factor = 1.5 ** degradations        # Exponential with degradations

    return time_factor * degrad_factor
```

#### Maintenance Effects

| Maintenance | Cost | Effect |
|-------------|------|--------|
| `quick_fix` | Low (C_q = 10) | Resets time-since-maintenance clock only |
| `full_service` | High (C_f = 100) | Resets both time clock AND degradation count |

#### Failure Cost

`failure` incurs high cost C_fail = 1000 and terminates the journey.

### Baseline Heuristic Policy

The ground truth simulator can run with a simple heuristic:

```python
# Heuristic: quick_fix every 30 days, full_service every 90 days
def baseline_policy(state):
    t_quick = state.time_since('quick_fix') or state.time
    t_full = state.time_since('full_service') or state.time

    if t_full >= 90:
        return 'full_service'
    elif t_quick >= 30:
        return 'quick_fix'
    else:
        return None  # No action
```

This provides:
- A sensible baseline for comparison
- Training data with "reasonable" maintenance patterns
- A benchmark for RL to beat

### What RL Should Discover

An optimal policy should learn:
1. **State-dependent timing** - maintain sooner when degradation count is high
2. **Action selection** - prefer full_service when degradations accumulate
3. **Risk-cost trade-off** - balance maintenance cost vs failure probability

### Recoverable Structure (Validation Targets)

| What to Recover | Validation Method |
|-----------------|-------------------|
| Failure depends on maintenance recency | Learned model's failure hazard should increase with time-since-maintenance |
| Degradation accelerates failure | Learned model should show higher failure rate with more degradations |
| Two maintenance types differ | Embeddings should separate quick_fix from full_service histories |
| Optimal maintenance timing | RL policy on learned model should match RL on ground truth |

### Embedding Validation

A trained generative model may produce embeddings at different levels:

**Subject-level embeddings:**
- Represent inherent widget characteristics (from subject features)
- Should correlate with ground-truth durability factors
- Validate via: clustering should separate high/low durability subjects, regression should predict failure rates

**Journey-level embeddings:**
- Represent accumulated state (maintenance history, degradation count, etc.)
- Should encode "risk state" at any point in time
- Validate via: classification of "needs maintenance soon" vs "healthy", regression to predict time-to-failure

**Downstream tasks for validation:**
- Clustering: Do embeddings group subjects/journeys with similar ground-truth characteristics?
- Classification: Can embeddings predict categorical outcomes (will fail within X days)?
- Regression: Can embeddings predict continuous outcomes (expected time to failure, total cost)?

This provides an additional validation axis beyond policy performance - even if RL policies match, embeddings should also recover meaningful structure.

---

## Implementation Sketch

### State Features Needed

```python
# Extend State class or use EmbeddingState
features = {
    'time': state.time,
    'time_since_quick_fix': state.time_since('quick_fix'),
    'time_since_full_service': state.time_since('full_service'),
    'degradation_count': state.event_count('degradation'),
    'degradations_since_full': state.events_since('degradation', 'full_service'),
    'quick_fix_count': state.event_count('quick_fix'),
    'full_service_count': state.event_count('full_service'),
}
```

### Event Definitions

```python
from discretiser import (
    EventType, EventRegistry, Simulator,
    Weibull, Exponential, StateDependentWeibull,
    make_censoring_event
)

# Degradation - happens periodically
degradation = EventType(
    name='degradation',
    survival_model=Exponential(rate=1/20),  # ~every 20 days on average
)

# Quick fix - ACTION (agent-controlled in RL, heuristic in baseline)
quick_fix = EventType(
    name='quick_fix',
    survival_model=...,  # Depends on policy mode
)

# Full service - ACTION
full_service = EventType(
    name='full_service',
    survival_model=...,  # Depends on policy mode
)

# Failure - state-dependent hazard
failure = EventType(
    name='failure',
    survival_model=StateDependentFailure(...),  # Custom class
    terminal=True
)

# Censoring at max time
censoring = make_censoring_event('censoring', Exponential(rate=1/365))
```

### Action Representation Options

**Option A: Actions as events with policy-controlled timing**
- Actions are EventTypes with survival models that depend on policy
- Policy outputs "time until next action" for each action type
- Natural fit with competing risks framework

**Option B: Decision points**
- At each event, agent chooses whether to trigger an action
- Actions fire immediately (dt=0) when chosen
- More like traditional RL discrete actions

**Option C: Continuous action space**
- Agent outputs parameters (e.g., "maintain every X days")
- Could be optimised with policy gradient methods

### Cost Function

```python
def journey_cost(result: SimulationResult) -> float:
    """Total cost of a simulated journey."""
    cost = 0.0

    for event in result.history:
        if event.event_name == 'quick_fix':
            cost += C_QUICK_FIX
        elif event.event_name == 'full_service':
            cost += C_FULL_SERVICE
        elif event.event_name == 'failure':
            cost += C_FAILURE

    # Optional: running cost per time unit
    # cost += result.final_time * C_PER_DAY

    return cost
```

---

## Open Questions

1. **Action timing**: How does the RL agent specify when to perform maintenance?
   - Continuous time output?
   - Decision at each event?
   - Fixed decision intervals?

2. **Observation space**: What does the agent observe?
   - Full state (time since each event, counts)?
   - Partial observation (realistic scenario)?
   - Raw event sequence (for sequence models)?

3. **Training data generation**:
   - Use baseline heuristic for all training data?
   - Mix of policies for diversity?
   - Random policies initially?

4. **Complexity level**:
   - Start simple (one maintenance type) then add complexity?
   - Or build full scenario from start?

---

## Implementation Notes

### State Methods Needed

The scenario requires methods not yet in `State`:
- `time_since(event_name)` - time since last occurrence of event (or None/inf)
- `events_since(event_name, reset_event)` - count of `event_name` since last `reset_event`

These are straightforward to add based on existing `_event_counts` and `history`.

### Failure Model Options

**Option A: StateDependentWeibull with modifier function**
- Use existing `StateDependentWeibull` with a custom `scale_modifier`
- Scale increases with time-since-maintenance and degradation count

**Option B: CompoundWeibull / bathtub with subject dependence**
- Use `CompoundWeibull(shape1, scale1, shape2, scale2)` for bathtub hazard
- Subject features modulate the scales: `scale_i = base_scale_i * f(subject.features)`
- Natural interpretation: some widgets are inherently more durable
- Could create `SubjectDependentCompoundWeibull` or use feature-based scale modifiers

The bathtub model is attractive because:
- Realistic failure pattern (early defects + wear-out)
- Subject heterogeneity is natural (manufacturing variation)
- Already have `CompoundWeibull` implemented

---

## Next Steps

- [ ] Implement `time_since()` and `events_since()` methods in State
- [ ] Create failure model (StateDependentWeibull or subject-dependent bathtub)
- [ ] Build baseline heuristic policy
- [ ] Generate sample data and visualise
- [ ] Define RL interface (action space, observation space, reward)
