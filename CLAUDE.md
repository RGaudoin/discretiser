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

## Public/Private Repository Split

This repo is designed to be **public**. Model training code lives in separate **private** repositories.

```
┌─────────────────────────────────────────────────────────────────┐
│                    discretiser (public)                         │
├─────────────────────────────────────────────────────────────────┤
│  • Simulation framework (events, state, survival models)        │
│  • Synthetic data generation with known ground-truth dynamics   │
│  • Interfaces for trained models (TrainedModelSurvival)         │
│  • Generic RL for policy optimisation                           │
│  • Validation (compare policy on trained model vs ground truth) │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                         imports from
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    training repo (private)                      │
├─────────────────────────────────────────────────────────────────┤
│  • Neural network architectures                                 │
│  • Training loops for generative models                         │
│  • Model export (pickle, ONNX, etc.)                            │
└─────────────────────────────────────────────────────────────────┘
```

**Workflow:**
1. Generate synthetic data here with known ground-truth dynamics
2. Train generative model externally (private repo)
3. Import trained model here via `TrainedModelSurvival` wrapper
4. Run generic RL to find optimal policy on trained model
5. Validate policy against ground-truth simulation

**Private repo also exports embeddings:**
- Subject-level embeddings → validate via risk stratification vs actual outcomes
- Journey-level embeddings → validate intervention optimality

**Installing discretiser in training repo:**
```bash
pip install git+https://github.com/RGaudoin/discretiser.git
```

## File Structure

```
discretiser/
+-- src/
|   +-- __init__.py
|   +-- survival.py       # Survival model base + implementations
|   +-- events.py         # EventType definitions + triggering rules
|   +-- state.py          # State, Subject, EmbeddingState, generators
|   +-- simulator.py      # Competing risks loop + DataFrame export
+-- docs/
|   +-- simulation_framework.md
|   +-- survival_models.md
|   +-- trained_model_integration.md
+-- notebooks/
|   +-- journey_simulation_examples.ipynb
+-- README.md
+-- CLAUDE.md
```

## Key Design Decisions

- `NeverOccurs()` is the canonical way to represent events that never happen (cure fractions, disabled events). `DeltaMass(inf)` emits a deprecation warning and returns `NeverOccurs()`.

- `Mixture.hazard()` uses the correct formula `h(t) = [sum w_i h_i(t) S_i(t)] / S(t)`, NOT the weighted average of component hazards.

- Trained model integration patterns in `docs/trained_model_integration.md` show the interface contract. Actual training implementations live in separate (private) repositories.

- `TrainedModelSurvival` is the generic wrapper for externally trained models. Any predictor with a `predict(features) -> params` method can be wrapped.

- `truncate(elapsed_time)` creates conditional distributions for re-triggering cleared events in autoregressive mode. Closed-form for Weibull; mixtures get updated weights.

- `MinSurvival` is a general competing risks model (min of any survival distributions). More flexible than `CompoundWeibull`.

## Dependencies

- numpy
- scipy (for distributions)
- pandas (for output)

## TODO

- [ ] Build synthetic data generator with known ground-truth dynamics
- [ ] Model on synthetic data:
  - [ ] Embeddings for journeys and patients - validate recovery of known structure
  - [ ] Generative modelling - test in RL setting, compare optimised policy (based on learnt model) against true synthetic data generator
- [ ] Pending event modes: Currently `add_pending_event()` keeps earliest time. Add `pending_mode` to TriggerRule:
  - `'earliest'` (current default) - keep whichever is earlier
  - `'reschedule'` - replace with new time
  - `'queue'` - allow multiple pending of same type
    - Needs instance IDs for cancellation (e.g., `'checkup_001'`)
    - Change `_pending_events` from `Dict[str, tuple]` to support multiple instances
- [ ] Validation test: autoregressive + truncated re-triggering ≡ competing mode
  - Compare distributions of event times/counts between:
    1. Competing mode (baseline) - pending events persist
    2. Autoregressive mode with truncated re-triggering - sample from `model.truncate(elapsed)`
  - Both should be statistically equivalent (same conditional distribution)
