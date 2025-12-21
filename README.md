# discretiser

Competing risks simulation framework for discrete event sequences.

Turns continuous time into discrete event simulations with:
- **Competing risks**: Multiple events compete; minimum time wins
- **Autoregressive triggering**: Events can trigger other events
- **State-dependent survival**: Event rates depend on journey history
- **Defective distributions**: Point masses for simultaneous events

Primary use case: patient journey simulation for training generative models,
counterfactual evaluation, and treatment optimisation.

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

See `notebooks/journey_simulation_examples.ipynb` for detailed examples.
