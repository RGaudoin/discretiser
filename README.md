# discretiser

Competing risks simulation framework for discrete event sequences.

Turns continuous time into discrete event simulations with:
- **Competing risks**: Multiple events compete; minimum time wins
- **Autoregressive triggering**: Events can trigger other events
- **State-dependent survival**: Event rates depend on journey history
- **Defective distributions**: Point masses for simultaneous events

Primary use case: generating synthetic event sequences with known ground-truth dynamics for:
- **Training generative models**: Learn to predict event timing from sequences
- **Embedding validation**: Verify that learned embeddings recover meaningful structure
- **Policy optimisation**: Use RL or other learning algorithms to find optimal intervention strategies
- **Model validation**: Compare policies learned on trained models vs ground truth

## Purpose

This framework generates synthetic data where the true dynamics are known, enabling rigorous validation of learned models:

1. **Generate** synthetic journeys with known ground-truth dynamics
2. **Train** generative models (externally) on the synthetic data
3. **Validate embeddings**: Do subject/journey embeddings cluster by true risk factors?
4. **Optimise policies**: Use learning algorithms on learned models to find intervention strategies
5. **Compare**: Evaluate learned-model policies against ground-truth optimal

The gap between policy-on-learned-model and policy-on-ground-truth measures how well the model captures actionable dynamics.

**Architecture**: This repo provides simulation, synthetic data generation, and policy evaluation (including basic RL). Generative model training (learning surrogate models from data) happens externally - implementations may come from various sources (proprietary, open-source, academic).

## Installation

```bash
pip install numpy scipy pandas
```

## Quick Start

```python
from discretiser.src import (
    Weibull, Exponential, EventType,
    make_censoring_event, simulate_cohort_simple
)

events = [
    EventType('diagnosis', Weibull(1.5, 30)),
    EventType('treatment', Weibull(1.2, 14)),
    EventType('outcome', Weibull(2.0, 180), terminal=True),
    make_censoring_event('censoring', Exponential(rate=1/365))
]

df = simulate_cohort_simple(n_subjects=100, events=events, max_time=365)
```

## Documentation

- [Simulation Framework](docs/simulation_framework.md) - How events, triggers, and state fit together
- [Survival Models](docs/survival_models.md) - Available distributions and composite models
- [Trained Model Integration](docs/trained_model_integration.md) - Interface for externally trained models
- [Synthetic Scenarios](docs/synthetic_scenarios.md) - Design notes for validation scenarios
- [Example notebook](notebooks/journey_simulation_examples.ipynb) - Detailed usage examples

## License

Apache 2.0 - see [LICENSE](LICENSE)
