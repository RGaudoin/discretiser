# Trained Model Integration

This document describes how to integrate externally trained models with the simulation framework.

## Architecture Overview

This repository provides the **interface** for trained models. Actual model training lives in separate (private) repositories.

```
┌─────────────────────────────────────────────────────────────────┐
│                    discretiser (public)                         │
├─────────────────────────────────────────────────────────────────┤
│  • Simulation with ground-truth dynamics                        │
│  • TrainedModelSurvival wrapper                                 │
│  • Generic RL for policy optimisation                           │
│  • Validation against ground truth                              │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                    trained model artifact
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    training repo (private)                      │
├─────────────────────────────────────────────────────────────────┤
│  • pip install discretiser (from git)                           │
│  • Train on simulated data                                      │
│  • Export model (pickle, ONNX, weights, etc.)                   │
└─────────────────────────────────────────────────────────────────┘
```

## The Interface Contract

Any trained model that predicts survival parameters can be used. The model must implement:

```python
# Minimal interface - just needs predict()
class TrainedPredictor:
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Args:
            features: Shape (batch_size, n_features) from State.to_feature_vector()

        Returns:
            Shape (batch_size, n_params) - e.g., (batch, 2) for Weibull shape/scale
        """
        ...
```

This works with sklearn models, Keras/PyTorch models, ONNX runtime, or any custom predictor.

## Using TrainedModelSurvival

The `TrainedModelSurvival` wrapper connects any predictor to the simulation:

```python
from discretiser.src import TrainedModelSurvival, EventType, Simulator
import joblib

# Load your trained model (from private training repo)
predictor = joblib.load('path/to/trained_model.pkl')

# Wrap it for use in simulation
trained_survival = TrainedModelSurvival(
    predictor=predictor,
    event_names=['diagnosis', 'treatment'],  # Features to extract
    distribution='weibull'  # What the model predicts
)

# Use in event definition
events = [
    EventType('outcome', trained_survival, terminal=True),
    # ... other events
]
```

## State Features

The wrapper uses `State.to_feature_vector()` to extract features:

```python
# Feature vector structure (in order):
# 1. Current time (1 value)
# 2. Event counts (1 per event in event_names)
# 3. Time since last occurrence (1 per event, 0 if never occurred)
# 4. Subject features (if subject.feature_vector exists)

features = state.to_feature_vector(['diagnosis', 'treatment'])
# Example: [time, n_diagnosis, n_treatment, t_since_diagnosis, t_since_treatment, ...]
```

For richer representations, use `EmbeddingState`:

```python
embedding = state.get_current_embedding()  # Decay-weighted event embeddings
```

## Supported Distributions

`TrainedModelSurvival` supports:

| Distribution | Predicted params | Sampling |
|--------------|------------------|----------|
| `weibull` | (shape, scale) | `scale * np.random.weibull(shape)` |
| `exponential` | (rate,) | `np.random.exponential(1/rate)` |
| `lognormal` | (mu, sigma) | `np.random.lognormal(mu, sigma)` |

## Training Data Extraction

To prepare simulation output for training:

```python
from discretiser.src import extract_training_data, simulate_cohort_simple

# Generate synthetic data
df = simulate_cohort_simple(n_subjects=10000, events=events, max_time=730)

# Extract features at each event for supervised learning
training_df = extract_training_data(
    df,
    event_names=['diagnosis', 'treatment'],
    target_event='outcome'
)
# Columns: features..., time_to_target, censored
```

## Validation Workflow

### 1. Define ground-truth dynamics

```python
class GroundTruthOutcome(Weibull):
    """Known: each treatment multiplies survival by 1.2"""
    def sample(self, state=None, subject=None):
        n_tx = state.event_count('treatment') if state else 0
        adjusted_scale = self.scale * (1.2 ** n_tx)
        return adjusted_scale * np.random.weibull(self.shape)
```

### 2. Generate synthetic data

```python
ground_truth_events = [
    EventType('treatment', Exponential(rate=1/30)),
    EventType('outcome', GroundTruthOutcome(shape=1.5, scale=100), terminal=True),
    make_censoring_event('censoring', Exponential(rate=1/365))
]
df = simulate_cohort_simple(n_subjects=10000, events=ground_truth_events, max_time=730)
```

### 3. Train externally (private repo)

```python
# In your private training repo:
from discretiser.src import extract_training_data
# ... train your model on the extracted data
# ... export trained model
```

### 4. Import and wrap

```python
predictor = joblib.load('trained_model.pkl')
learned_outcome = TrainedModelSurvival(predictor, ['treatment'], 'weibull')
```

### 5. Optimise policy (RL)

```python
# Use learned model as environment
learned_events = [
    EventType('treatment', ...),  # Policy controls this
    EventType('outcome', learned_outcome, terminal=True),
]
# Run RL to find optimal treatment policy
optimal_policy = run_rl(learned_events, ...)
```

### 6. Validate against ground truth

```python
# Apply learned policy to ground-truth simulation
results_on_ground_truth = evaluate_policy(optimal_policy, ground_truth_events)

# Compare: Does the policy that's optimal on the learned model
# also perform well on the true underlying dynamics?
```

## Alternative Approaches

### Direct hazard prediction

Instead of predicting distribution parameters:

```python
class HazardPredictor(SurvivalModel):
    """Predict h(t|state) directly, sample via inverse transform."""

    def __init__(self, hazard_model, event_names):
        self.model = hazard_model
        self.event_names = event_names

    def hazard(self, t: float, state=None) -> float:
        features = self._build_features(t, state)
        return self.model.predict(features)[0]

    def sample(self, state=None, subject=None) -> float:
        # Inverse transform sampling with numerical integration
        ...
```

### Embedding-based prediction

```python
# Use EmbeddingState for richer state representation
state = EmbeddingState(subject, event_embeddings, decay_rates)
embedding = state.get_current_embedding()
# Concatenate with other features for prediction
```
