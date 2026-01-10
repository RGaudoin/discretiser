# Scenario: Widget Maintenance (Full)

A neutral domain (avoiding medical/clinical complexity) that captures the core dynamics of interest. This extends [basic_bathtub](basic_bathtub.md) with additional complexity.

## Events

| Event | Type | Description |
|-------|------|-------------|
| `degradation` | Recurring | Widget quality decreases (see options below) |
| `quick_fix` | Action | Cheap, short-term maintenance |
| `full_service` | Action | Expensive, long-term maintenance |
| `failure` | Terminal | Widget fails, high cost |

## Ground Truth Dynamics

### Failure Hazard

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

### Maintenance Effects

| Maintenance | Cost | Effect |
|-------------|------|--------|
| `quick_fix` | Low (C_q = 10) | Resets time-since-maintenance clock only |
| `full_service` | High (C_f = 100) | Resets both time clock AND degradation count |

### Failure Cost

`failure` incurs high cost C_fail = 1000 and terminates the journey.

## Optional: Monitor Event

An additional `monitor` event adds an observation/decision layer:

| Event | Type | Description |
|-------|------|-------------|
| `monitor` | Action | Inspect widget state, small cost |

**Mechanistic triggering:**
- `monitor` observes current degradation count
- If degradation ≥ threshold_high → trigger `full_service`
- If degradation ≥ threshold_low → trigger `quick_fix`
- Otherwise → no action

This creates a more realistic flow: inspect → decide → act

**Optimisation levels:**
1. **Parameter optimisation** (simple baseline): tune monitor interval and thresholds
2. **Learning on monitor timing**: when to inspect (state-dependent)
3. **Learning on full action space**: when to inspect AND what action to take after

## Baseline Heuristic Policy

With monitor:
```python
# Heuristic: monitor every 15 days, act based on degradation count
def baseline_policy(state):
    t_monitor = state.time_since('monitor') or state.time

    if t_monitor >= 15:
        return 'monitor'  # Triggers maintenance mechanistically
    else:
        return None  # No action

# Monitor triggers (defined in event setup):
# - degradation_count >= 3 → full_service
# - degradation_count >= 1 → quick_fix
```

Without monitor:
```python
# Direct heuristic: quick_fix every 30 days, full_service every 90 days
def baseline_policy_direct(state):
    t_quick = state.time_since('quick_fix') or state.time
    t_full = state.time_since('full_service') or state.time

    if t_full >= 90:
        return 'full_service'
    elif t_quick >= 30:
        return 'quick_fix'
    else:
        return None  # No action
```

## Degradation Modelling Options

**Option A: Random arrivals (Poisson-like)**
```python
degradation = EventType('degradation', Exponential(rate=1/20))  # ~every 20 days avg
```
- Simple, stochastic
- Harder to interpret/verify recovery

**Option B: Fixed intervals**
```python
degradation = EventType('degradation', DeltaMass(interval))  # Exactly every X days
# interval could be subject-dependent: interval = 15 + 10 * subject.features[0]
```
- Deterministic, interpretable
- Can be reset by full_service (re-trigger with fresh interval)
- Subject features modulate interval (some widgets degrade faster)

**Option C: Observed only at monitor**
- No independent degradation events
- `monitor` event reveals current "degradation level" (could be time-based or latent)
- Adds partial observability aspect

**Option D: Implicit via time**
- No explicit degradation events
- Failure hazard depends directly on time-since-full-service
- Simplest, but less rich state representation
- **Note:** No point having monitor event if degradation is implicit - nothing to observe

**Coupling with monitor event:**
- Options A/B (explicit degradation): Monitor makes sense - observe degradation count, decide action
- Options C/D (no explicit degradation): Monitor is redundant - just use time-based policy

Option B (fixed intervals, subject-dependent) may be cleanest for validation.

## What Learning Can Do That Linear Cannot

1. **React to events**: "Just had a degradation → maintain sooner" - dynamic response to journey history
2. **Non-linearity**: Feature interactions, thresholds (e.g., "if durability < 0.3 AND time > 20...")
3. **Bathtub phases**: Different strategies for different life phases:
   - Early: Watch for infant mortality defects
   - Mid-life: Standard intervals
   - Late: Aggressive maintenance as wear-out accelerates

## Validation Targets

| What to Recover | Validation Method |
|-----------------|-------------------|
| Failure depends on maintenance recency | Learned model's failure hazard should increase with time-since-maintenance |
| Degradation accelerates failure | Learned model should show higher failure rate with more degradations |
| Two maintenance types differ | Embeddings should separate quick_fix from full_service histories |
| Optimal maintenance timing | Policy on learned model should match policy on ground truth |

## State Features

```python
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

## Implementation Notes

### State Methods Needed

The scenario requires methods not yet in `State`:
- `time_since(event_name)` - time since last occurrence of event (or None/inf)
- `events_since(event_name, reset_event)` - count of `event_name` since last `reset_event`

### Failure Model Options

**Option A: StateDependentWeibull with modifier function**
- Use existing `StateDependentWeibull` with a custom `scale_modifier`
- Scale increases with time-since-maintenance and degradation count

**Option B: CompoundWeibull / bathtub with subject dependence**
- Use `CompoundWeibull(shape1, scale1, shape2, scale2)` for bathtub hazard
- Subject features modulate the scales
- Natural interpretation: some widgets are inherently more durable

**Note on subject heterogeneity:**
No need for discrete widget "types" - the subject feature vector already provides continuous heterogeneity. Each subject is a point in feature space; ground-truth dynamics are parameterised by features.
