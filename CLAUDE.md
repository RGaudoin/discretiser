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

This repo is designed to be **public**. Advanced algorithms live in `discretiser-learn` (private).

```
┌─────────────────────────────────────────────────────────────────┐
│                    discretiser (public)                         │
├─────────────────────────────────────────────────────────────────┤
│  • Simulation framework (events, state, survival models)        │
│  • Synthetic data generation with known ground-truth dynamics   │
│  • RL environment (ServiceEnv) and evaluation utilities         │
│  • Notebooks showing results (may import from discretiser-learn)│
│  • Interfaces for trained models (TrainedModelSurvival)         │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                         imports from
                              │
┌─────────────────────────────────────────────────────────────────┐
│                 discretiser-learn (private)                     │
├─────────────────────────────────────────────────────────────────┤
│  • Advanced RL algorithms (expected-value RL, custom AC)        │
│  • Generative models that learn from synthetic data             │
│  • Surrogate environments based on learned models               │
│  • Design documents and research notes                          │
└─────────────────────────────────────────────────────────────────┘
```

**Workflow:**
1. Generate synthetic data here with known ground-truth dynamics
2. Train advanced models via discretiser-learn
3. Use learned models as surrogate environments
4. Run RL to find optimal policy
5. Validate policy against ground-truth simulation

**Installing discretiser:**
```bash
pip install git+https://github.com/RGaudoin/discretiser.git
```

**For advanced algorithms:** Contact for access to discretiser-learn.

## File Structure

```
discretiser/
+-- src/
|   +-- __init__.py
|   +-- survival.py       # Survival model base + implementations
|   +-- events.py         # EventType definitions + triggering rules
|   +-- state.py          # State, Subject, EmbeddingState, generators
|   +-- simulator.py      # Competing risks loop + DataFrame export
|   +-- rl/
|       +-- environment.py   # ServiceEnv (Gymnasium wrapper)
|       +-- evaluation.py    # evaluate_model, compare_with_baselines
|       +-- scenarios.py     # get_default_scenario, baseline policies
|       +-- callbacks.py     # Training callbacks for logging
+-- docs/
|   +-- simulation_framework.md
|   +-- survival_models.md
|   +-- trained_model_integration.md
+-- notebooks/
|   +-- journey_simulation_examples.ipynb
|   +-- rl_quickstart.ipynb   # Minimal RL example (runnable)
|   +-- rl_dqn.ipynb          # DQN experiments (results, needs discretiser-learn)
|   +-- rl_sac.ipynb          # SAC experiments (results, needs discretiser-learn)
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
