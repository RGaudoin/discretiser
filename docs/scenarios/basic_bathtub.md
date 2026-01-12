# Scenario: Basic Bathtub

Minimal scenario to validate the pipeline end-to-end before adding complexity.

## Specification

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Degradation | None (implicit via time) | Simplest option |
| Monitor event | None | Not needed without explicit degradation |
| Maintenance | Single `service` action | One action type to start |
| Failure model | Bathtub with subject-dependent scale | Gives heterogeneity |
| Baseline | Linear: `interval = a + b·durability` | Adds data diversity |
| State | `(time, time_since_service, service_count, features)` | Includes service history for effective age |

## Why This Is True RL (Not Just a Bandit)

Without careful design, each service cycle is independent → contextual bandit, not RL. To create sequential dependence, need state that doesn't fully reset.

**Options for sequential structure:**
1. **Cumulative wear**: Service doesn't fully restore; permanent degradation accumulates
2. **Service effectiveness degrades**: Repeated services become less effective over time
3. **Budget/resource constraints**: Limited services available across lifetime
4. **Bathtub + effective age model** (chosen): Service reduces effective age but can't stop time

### Effective Age Model

```
effective_age = total_age - cumulative_service_benefit
service adds delta_t to cumulative_service_benefit (capped)
failure_hazard = bathtub(effective_age)
```

- Service "rejuvenates" widget by delta_t
- But total_age keeps advancing → bathtub phases still matter
- Optimal strategy changes with age: less service early, more late
- **State that carries across cycles**: cumulative_service_benefit (or equivalently, service_count)

**This is an MDP:** The effective_age is fully computable from state (`total_age - service_count * delta_t`). Subject heterogeneity comes from features (observed). Uncertainty comes from:
- Stochastic failure times (bathtub survival model)
- Optionally stochastic service effects (`delta_t ~ Distribution`)

No hidden state → standard MDP, not POMDP.

## Implementation

The scenario is implemented in `src/scenarios/basic_bathtub.py`:

```python
from src.scenarios import (
    BasicBathtubScenario,      # Full scenario class
    EffectiveAgeBathtub,       # Survival model
    generate_bathtub_subjects,  # Subject generation
    generate_bathtub_data,      # Data generation with baseline
    BathtubCosts,              # Cost structure
)

# Generate training data
df, results, costs = generate_bathtub_data(
    n_subjects=1000,
    max_time=200.0,
    baseline_a=20.0,  # Base interval
    baseline_b=10.0,  # Durability coefficient
)
```

**Cost structure** (defined in scenario):
- Service cost: 50 per service
- Failure cost: 500 terminal
- Revenue: 1 per time unit while operational

## Open Issues (RL Layer)

These issues belong to the **RL layer**, not the scenario definition. The scenario provides dynamics and costs; RL decides how to act.

**Action space representation:**
- How does the agent specify when to service?
- Options: continuous time output, decision at fixed intervals, decision at each event
- Suggestion: fixed decision intervals (e.g., every 10 days, choose service/wait)

**Observation space:**
- What does the agent see? Full state vs partial?
- Suggestion: full state (time, time_since_service, subject features)

## Validation

1. Generate training data with linear baseline
2. Train generative model on data
3. Run policy learning on ground truth → Policy_truth
4. Run policy learning on learned model → Policy_learned
5. Compare both policies on ground truth

**Success criteria:**
- Learning algorithm beats linear baseline on ground truth (proves it adds value)
- Policy_learned performs close to Policy_truth (proves model is useful)

## What Learning Should Discover

An optimal policy should learn:
1. **Feature-dependent timing** - high-durability subjects need less frequent maintenance
2. **Phase-dependent behaviour** - different strategies for bathtub phases:
   - Early: Less service (can't prevent infant mortality defects anyway)
   - Mid-life: Standard intervals
   - Late: More aggressive maintenance (wear-out accelerating)
3. **Risk-cost trade-off** - balance maintenance cost vs failure probability

## Next Scenarios

Once this works, add complexity incrementally:
- [widget_maintenance_full](widget_maintenance_full.md): Add second maintenance type (quick_fix vs full_service)
- Explicit degradation events
- Monitor event with mechanistic triggering
- Continuous action spaces

## Future Investigation: Effective Age Model

**TODO (deep dive later):** The effective age bathtub model has several aspects worth investigating:

### 1. Relationship to Truncation

- Currently using `truncate(effective_age)` to condition on survival
- Is this mathematically correct when effective_age ≠ real time?
- Alternative: sample from base distribution, reject if < effective_age
- **Learning vs knowing:** Ground truth has exact formula for truncation. A learned model must discover this structure from rollouts:
  - Learner sees: (state, action, outcome) trajectories
  - Must learn: P(failure | effective_age, history, ...)
  - The conditional/truncation structure is implicit in the data
  - This is exactly what validation tests - can learner recover these dynamics?

### 2. Implications for RL/Learning

- Action space: How does effective age interact with continuous vs discrete actions?
- State representation: Should RL see effective_age directly or compute it?
- Temporal abstraction: Does effective age enable options/macro-actions?
- **Rejuvenation as action dimension:** Can delta_t (service strength) be part of the action?
  - Action = (when, how_much) instead of just (when)
  - Different service levels → different rejuvenation amounts → different costs
  - Creates trade-off: aggressive service (high delta_t, high cost) vs light service (low delta_t, low cost)
  - The truncation offset becomes a function of cumulative action history
  - Feasibility: Mathematically straightforward, but may complicate learning

### 3. Alternatives Achieving Similar POC Goals

- Simpler: Just scale hazard by service_count (no effective age concept)
- More complex: Explicit degradation state that service partially resets
- Different: Service affects one bathtub component but not the other

### 4. Model Identifiability

- Can a learner distinguish effective age from time-based hazard?
- What data diversity is needed to learn the delta_t parameter?

These questions don't block the POC but should be revisited when designing RL integration.
