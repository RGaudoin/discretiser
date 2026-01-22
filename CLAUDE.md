# CLAUDE.md - discretiser

## Purpose

Competing risks simulation for synthetic patient journeys. See [README](README.md) for overview.

## Coding Guidelines

- Prefer efficient code even during development (pre-compute where sensible)
- Survival models should be composable (mixtures, point masses, conditionals)
- Keep simulation loop simple; complexity goes in model definitions
- Use British spelling (behaviour, organisation, etc.)

## Architecture

```
Subject (static features)
    |
State (accumulated history + derived features)
    |
Competing Risks: min(time_to_event for all active events)
    |
Winner updates state -> may trigger other events (t=0 or later)
    |
Repeat until censoring or max_time
```

## Key Design Decisions

- `NeverOccurs()` is the canonical way to represent events that never happen (cure fractions, disabled events). `DeltaMass(inf)` emits a deprecation warning and returns `NeverOccurs()`.

- `Mixture.hazard()` uses the correct formula `h(t) = [sum w_i h_i(t) S_i(t)] / S(t)`, NOT the weighted average of component hazards.

- Trained model integration patterns in `docs/trained_model_integration.md` show the interface contract. Actual training implementations live in separate (private) repositories.

- `TrainedModelSurvival` is the generic wrapper for externally trained models. Any predictor with a `predict(features) -> params` method can be wrapped.

- `truncate(elapsed_time)` creates conditional distributions for re-triggering cleared events in autoregressive mode. Closed-form for Weibull; mixtures get updated weights.

- `MinSurvival` is a general competing risks model (min of any survival distributions). More flexible than `CompoundWeibull`.

- **ServiceEnv truncation at max_time**: `max_time` is a truncation trigger, NOT a reward clipping boundary. When the agent chooses a delay that would cross `max_time`, the step earns its full reward (for the complete delay), then truncates if survived. This ensures correct Bellman updates. The "penalise failure" reward formulation applies: service costs service_cost, failure costs failure_cost. Episode times can exceed max_time; it just indicates when to bootstrap.

## TODO

- [ ] Validation test: autoregressive + truncated re-triggering ≡ competing mode
  - Compare distributions of event times/counts between competing mode (pending events persist) vs autoregressive with `model.truncate(elapsed)`
  - Both should be statistically equivalent — confirms truncation math is correct

## Future Additions

- **Pending event modes**: Currently `add_pending_event()` keeps earliest time. Could add `pending_mode`: `'reschedule'` (replace with new time), `'queue'` (allow multiple pending of same type). Useful for richer synthetic scenarios.
