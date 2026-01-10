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

**Observability note:** If `effective_age = total_age - service_count * delta_t`, it's deterministic and fully computable from state → standard MDP.

**Options for adding uncertainty:**
1. **Stochastic service effect** (MDP with noisy transitions): `delta_t ~ Normal(mean, var)` - service benefit varies, but cumulative effect is tracked
2. **Hidden quality** (true POMDP): subject has unobserved quality factor affecting effective_age
3. **Keep deterministic** for simplicity in first scenario

## Open Issues

**Action space representation:**
- How does the agent specify when to service?
- Options: continuous time output, decision at fixed intervals, decision at each event
- Suggestion: fixed decision intervals (e.g., every 10 days, choose service/wait)

**Observation space:**
- What does the agent see? Full state vs partial?
- Suggestion: full state (time, time_since_service, subject features)

**Reward/cost structure:**
- Service cost: C_service (immediate cost)
- Failure cost: C_failure (terminal cost)
- **Revenue: R per time unit while operational** (critical - otherwise "fail fast" is optimal)
- Example: R=1/day, C_service=50, C_failure=500
- Trade-off: service extends revenue-generating life but costs money

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
