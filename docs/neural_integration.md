# Neural Network Integration

This document describes patterns for integrating neural networks with the survival model framework. These are **example patterns** - not implemented classes - to guide your own implementations.

## State Features

Every `SurvivalModel.sample(state, subject)` call receives the full journey history:

```python
state.event_count('treatment')           # Number of occurrences
state.time_since_last('treatment')       # Time since last occurrence
state.to_feature_vector(event_names)     # Numerical summary for ML
state.get_current_embedding()            # Decay-weighted embeddings (EmbeddingState)
```

## Neural Wrapper Pattern

Wrap a trained model to predict survival parameters from state:

```python
class NeuralSurvival(SurvivalModel):
    """
    Example pattern - not an implemented class.
    Wraps a trained neural network to predict Weibull parameters.
    """
    def __init__(self, trained_model, event_names):
        self.model = trained_model
        self.event_names = event_names

    def sample(self, state=None, subject=None):
        features = state.to_feature_vector(self.event_names)
        shape, scale = self.model.predict(features[np.newaxis, :])[0]
        return scale * np.random.weibull(shape)

    def survival(self, t: float) -> float:
        # Would need to store last predicted params or recompute
        raise NotImplementedError("Survival function requires state context")
```

## Validation Workflow

To test whether your neural architecture can capture state-dependent dynamics:

### 1. Hand-craft ground-truth dynamics

Create a survival model with known, recoverable parameters:

```python
class GroundTruthOutcome(Weibull):
    """
    Example pattern - not an implemented class.
    Each treatment multiplies scale by 1.2 (known ground truth).
    """
    def sample(self, state=None, subject=None):
        n_tx = state.event_count('treatment') if state else 0
        adjusted_scale = self.scale * (1.2 ** n_tx)  # Known: 1.2x per treatment
        return adjusted_scale * np.random.weibull(self.shape)
```

### 2. Generate synthetic journeys

```python
events = [
    EventType('treatment', Exponential(rate=1/30)),
    EventType('outcome', GroundTruthOutcome(shape=1.5, scale=100), terminal=True),
    make_censoring_event('censoring', Exponential(rate=1/365))
]
df = simulate_cohort_simple(n_subjects=10000, events=events, max_time=730)
```

### 3. Train neural network

Train your model to predict survival parameters from state features extracted from the simulated journeys.

### 4. Verify recovery

Does the network learn the 1.2x scaling per treatment? If not, the architecture may lack capacity or the training data may be insufficient.

This validates whether your architecture can capture state-dependent dynamics before applying to real data where ground truth is unknown.

## Alternative Approaches

### Embedding-based prediction

Use `EmbeddingState` for richer state representation:

```python
embedding = state.get_current_embedding()  # Decay-weighted event embeddings
# Feed to neural network alongside static subject features
```

### Hazard prediction

Instead of predicting distribution parameters, predict hazard directly:

```python
class NeuralHazard(SurvivalModel):
    """Predict instantaneous hazard, integrate for survival."""

    def hazard(self, t: float, state=None) -> float:
        features = self._build_features(t, state)
        return self.model.predict(features)[0]

    def sample(self, state=None, subject=None):
        # Use inverse transform sampling with numerical integration
        ...
```
