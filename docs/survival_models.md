# Survival Models

This document describes the available survival model classes for competing risks simulation.

## Model Interface

Every model implements:

```python
class SurvivalModel(ABC):
    def sample(self, state=None, subject=None) -> float:
        """Sample time to event, optionally conditioned on state/subject."""

    def survival(self, t: float) -> float:
        """Survival function S(t) = P(T > t)."""

    def hazard(self, t: float) -> float:
        """Hazard function h(t) = f(t) / S(t)."""
```

## Parametric Models

| Model | Parameters | Use case |
|-------|------------|----------|
| `Weibull(shape, scale)` | shape < 1: decreasing hazard, = 1: constant (exponential), > 1: increasing | Most event times |
| `Exponential(rate)` | Memoryless; special case of Weibull with shape=1 | Constant hazard events |
| `LogNormal(mu, sigma)` | log(T) ~ Normal(mu, sigma) | Skewed positive times |
| `Gamma(shape, rate)` | Alternative to Weibull | Flexible positive times |

## Point Masses and Defective Distributions

| Model | Use case |
|-------|----------|
| `DeltaMass(t0)` | Scheduled events at fixed times (e.g., checkups at 30, 60, 90 days) |
| `NeverOccurs()` | Cure fraction component, disabled events (S(t) = 1 always) |
| `PointMassPlusContinuous(p0, continuous)` | Simultaneous + later events (defective distribution) |
| `PointMasses(point_masses, continuous)` | Multiple scheduled times plus optional continuous tail |

### Defective Distributions

For triggered events where:
- With probability p0: event occurs immediately (t=0)
- With probability (1-p0): event time follows continuous distribution

The survival function has a step at t=0 but doesn't drop to zero:

```
S(t) = 1                           for t < 0
S(0+) = 1 - p0                     (immediate drop by p0)
S(t) = (1 - p0) * S_continuous(t)  for t > 0
```

Analogous to zero-inflated models or cure-fraction models.

**Use cases:**
- **Simultaneous discovery**: When diagnosed with A, 30% chance B is found immediately
- **Cure fraction**: Some subjects never experience the event
- **Triggered + delayed**: Treatment might immediately resolve (20%) or take time (80%)

```python
# Example: 30% immediate trigger, 70% follows Weibull
trigger_model = PointMassPlusContinuous(
    p0=0.3,
    continuous=Weibull(shape=1.5, scale=10.0)
)
```

## Composite Models

| Model | Use case |
|-------|----------|
| `Mixture(models, weights)` | Cure fractions, multimodal populations |
| `CompoundWeibull(shape1, scale1, shape2, scale2)` | Bathtub hazard (additive hazards from two Weibulls) |

### Mixture Models

```python
# Cure fraction: 20% never experience event, 80% follow Weibull
cure_model = Mixture(
    models=[NeverOccurs(), Weibull(shape=1.5, scale=30)],
    weights=[0.2, 0.8]
)
```

**Important**: The mixture hazard is NOT the weighted average of component hazards:

```
h(t) = [sum_i w_i * h_i(t) * S_i(t)] / S(t)
```

For cure models, the hazard decreases over time as the uncured population is depleted.

### Bathtub Hazard (CompoundWeibull)

Combines two Weibulls with additive hazards:

```python
# High early hazard (infant mortality) + increasing late hazard (ageing)
bathtub = CompoundWeibull(
    shape1=0.5, scale1=10,   # Decreasing early hazard
    shape2=2.0, scale2=100   # Increasing late hazard
)
```

## Simultaneous Events - Design Options

| Approach | Mechanism | When to use |
|----------|-----------|-------------|
| Point mass at t=0 | Defective distribution with P(T=0) = p0 | Probabilistic triggering |
| Composite event | "A+B" as distinct event type | Co-occurrence has own statistics |
| Conditional trigger | Separate Bernoulli after A occurs | Complex conditional logic |

## State-Dependent Models

See [Neural Integration](neural_integration.md) for using neural networks to predict survival parameters from journey state.

The built-in `StateDependentWeibull` provides a simple callback-based approach:

```python
def scale_by_treatment_count(base_scale, state, subject):
    n_tx = state.event_count('treatment') if state else 0
    return base_scale * (1.2 ** n_tx)

model = StateDependentWeibull(
    base_shape=1.5,
    base_scale=30.0,
    scale_modifier=scale_by_treatment_count
)
```

## Truncated Distributions

For autoregressive modelling, events may need to be re-triggered after being cleared. The `truncate()` method creates a conditional distribution given survival up to a certain time.

### Interface

```python
# Base method available on all SurvivalModel classes
truncated = model.truncate(elapsed_time)

# With explicit sampling method
truncated = model.truncate(elapsed_time, sampling_method='numerical')
```

The `sampling_method` parameter controls how `inverse_survival()` is computed:
- `'auto'` (default): use analytic if available, else numerical
- `'analytic'`: use closed-form (error if not available)
- `'numerical'`: always use root-finding (Brent's method)

Returns a new `SurvivalModel` where:
- `S'(t) = S(t + t₀) / S(t₀)` - conditional survival
- `h'(t) = h(t + t₀)` - hazard just shifts
- `sample()` draws from the conditional distribution

### Closed-Form Implementations

| Model | Truncation |
|-------|------------|
| `Weibull` | Closed-form inverse via `λ * ((t₀/λ)^k - ln(U))^(1/k) - t₀` |
| `Exponential` | Memoryless - truncated exponential is just exponential |
| `LogNormal`, `Gamma` | Via scipy's quantile function |
| `NeverOccurs` | Returns itself (still never occurs) |
| `Mixture` | Mixture with updated weights + truncated components |
| `MinSurvival`, `CompoundWeibull` | Min of truncated components |

### Mixture Truncation

For mixtures, the component weights are updated based on conditional survival:

```
w'_i = w_i * S_i(t₀) / S(t₀)
```

The component with higher survival at t₀ gets more weight - if you've survived this long, you're more likely from the longer-lived component.

### Example: Re-triggering Cleared Events

```python
# In autoregressive mode, events are cleared after each occurrence
# To re-trigger with correct conditional distribution:

pending = state.get_pending_summary()
for event_name, info in pending.items():
    elapsed = current_time - triggered_time  # time since original trigger
    original_model = event_registry[event_name].survival_model
    truncated_model = original_model.truncate(elapsed)
    new_dt = truncated_model.sample()
    # new_dt is sampled from S(t | T > elapsed)
```

### MinSurvival Class

A general competing risks model (more flexible than `CompoundWeibull`):

```python
# Minimum of any survival distributions (additive hazards)
min_model = MinSurvival([
    Weibull(0.5, 10),   # Early hazard
    Weibull(2.0, 100),  # Late hazard
    Exponential(0.01)   # Constant background hazard
])
```

Properties:
- `S(t) = prod_i S_i(t)`
- `h(t) = sum_i h_i(t)`
- `sample() = min(sample_i for all i)`
