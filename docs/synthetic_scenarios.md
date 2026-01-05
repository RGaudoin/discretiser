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

## Simple POC (Proof of Concept)

Minimal version to validate the pipeline end-to-end before adding complexity.

### POC Specification

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Degradation | None (implicit via time) | Simplest option |
| Monitor event | None | Not needed without explicit degradation |
| Maintenance | Single `service` action | One action type to start |
| Failure model | Bathtub with subject-dependent scale | Gives heterogeneity for RL |
| Baseline | Linear: `interval = a + b·durability` | Adds data diversity |
| State | `(time, time_since_service, features)` | Minimal state space |

### POC Open Issues

**Action space representation:**
- How does RL specify when to service?
- Options: continuous time output, decision at fixed intervals, decision at each event
- For POC: suggest fixed decision intervals (e.g., every 10 days, choose service/wait)

**Observation space:**
- What does the agent see? Full state vs partial?
- For POC: full state (time, time_since_service, subject features)

**Reward/cost structure:**
- Service cost: C_service
- Failure cost: C_failure
- For POC: simple immediate costs, no discounting initially

### POC Validation

1. Generate training data with linear baseline
2. Train generative model on data
3. Run RL on ground truth → Policy_truth
4. Run RL on learned model → Policy_learned
5. Compare both policies on ground truth

**Success criteria:**
- RL beats linear baseline on ground truth (proves RL adds value)
- Policy_learned performs close to Policy_truth (proves model is useful)

### After POC

Once POC works, add complexity incrementally:
- Add second maintenance type (quick_fix vs full_service)
- Add explicit degradation events
- Add monitor event with mechanistic triggering
- Explore continuous action spaces

---

## Scenario: Widget Maintenance (Full)

A neutral domain (avoiding medical/clinical complexity) that captures the core dynamics of interest. This section describes the full scenario with all options; see Simple POC above for minimal starting point.

### Events

| Event | Type | Description |
|-------|------|-------------|
| `degradation` | Recurring | Widget quality decreases (see options below) |
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

### Optional: Monitor Event

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
2. **RL on monitor timing**: when to inspect (state-dependent)
3. **RL on full action space**: when to inspect AND what action to take after

### Baseline Heuristic Policy

The ground truth simulator can run with a simple heuristic:

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

Alternative without monitor:
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

This provides:
- A sensible baseline for comparison
- Training data with "reasonable" maintenance patterns
- A benchmark for RL to beat
- Parameter optimisation as intermediate step before full RL

### Policy Hierarchy

Three levels of policy sophistication:

**1. Global parameter optimisation:**
- "Maintain every X days" → search for best X
- Same rule for all subjects
- Works if optimal policy is a simple fixed rule
- Produces homogeneous training data (problem for learning)

**2. Linear feature-based baseline:**
- `interval = a + b·features` - simple mapping from features to timing
- Different subjects get different intervals
- Adds data diversity tied to features (needed for learning)
- But can't capture non-linear effects or state-dependent decisions

**3. RL (full policy learning):**
- π(state, features) → action
- Adapts to accumulated history, not just initial features
- Captures non-linear feature interactions
- Phase-dependent behaviour (early-life vs wear-out)

### What RL Can Do That Linear Cannot

1. **React to events**: "Just had a degradation → maintain sooner" - dynamic response to journey history
2. **Non-linearity**: Feature interactions, thresholds (e.g., "if durability < 0.3 AND time > 20...")
3. **Bathtub phases**: Different strategies for different life phases:
   - Early: Watch for infant mortality defects
   - Mid-life: Standard intervals
   - Late: Aggressive maintenance as wear-out accelerates

### Data Diversity for Learning

A fixed-interval baseline produces homogeneous data - learner can't discover what happens at other intervals.

**Solution**: Use linear feature-based baseline for training data generation:
- `interval = base + weight·durability_feature`
- Creates variation tied to features
- Learner sees different intervals for different subjects
- Can discover feature→outcome relationships

### Validation Logic

If RL on ground truth significantly beats linear baseline, AND RL on learned model approximates the ground-truth RL performance, the pipeline is validated:
- Learned model captures enough dynamics for good policy learning
- Gap between RL-on-learned vs RL-on-truth measures model quality

### What RL Should Discover

An optimal policy should learn:
1. **Feature-dependent timing** - high-durability subjects need less frequent maintenance
2. **State-dependent adjustments** - maintain sooner if recent events indicate problems
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

### Degradation Modelling Options

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

Option B (fixed intervals, subject-dependent) may be cleanest for validation: clear ground truth, easy to verify learned model recovers the interval structure, and monitor event has meaningful role.

### Event Definitions

```python
from discretiser import (
    EventType, EventRegistry, Simulator,
    Weibull, Exponential, StateDependentWeibull,
    make_censoring_event
)

# Degradation - fixed interval (Option B)
# Could make interval subject-dependent
degradation = EventType(
    name='degradation',
    survival_model=DeltaMass(20),  # Exactly every 20 days
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

**Note on subject heterogeneity:**
No need for discrete widget "types" - the subject feature vector already provides continuous heterogeneity. Each subject is a point in feature space; ground-truth dynamics are parameterised by features. Different failure/degradation profiles emerge naturally from the feature distribution.

---

## Next Steps

- [ ] Implement `time_since()` and `events_since()` methods in State
- [ ] Create failure model (StateDependentWeibull or subject-dependent bathtub)
- [ ] Build baseline heuristic policy
- [ ] Generate sample data and visualise
- [ ] Define RL interface (action space, observation space, reward)
