"""
Survival model classes for competing risks simulation.

Each model implements:
- sample(state, subject) -> float: sample time to event
- survival(t) -> float: S(t) = P(T > t)
- hazard(t) -> float: h(t) = f(t) / S(t)  (optional)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
from scipy import stats


class SurvivalModel(ABC):
    """Base class for survival models."""

    @abstractmethod
    def sample(self, state: Any = None, subject: Any = None) -> float:
        """Sample time to event, optionally conditioned on state/subject."""
        pass

    @abstractmethod
    def survival(self, t: float) -> float:
        """Survival function S(t) = P(T > t)."""
        pass

    def cdf(self, t: float) -> float:
        """Cumulative distribution function F(t) = P(T <= t) = 1 - S(t)."""
        return 1.0 - self.survival(t)

    def hazard(self, t: float) -> float:
        """Hazard function h(t). Default uses numerical approximation."""
        eps = 1e-6
        s_t = self.survival(t)
        if s_t < eps:
            return float('inf')
        s_t_eps = self.survival(t + eps)
        return -(s_t_eps - s_t) / (eps * s_t)


# -----------------------------------------------------------------------------
# Parametric Models
# -----------------------------------------------------------------------------

class Weibull(SurvivalModel):
    """
    Weibull distribution.

    shape (k): < 1 decreasing hazard, = 1 constant (exponential), > 1 increasing
    scale (λ): characteristic time
    """

    def __init__(self, shape: float, scale: float):
        self.shape = shape  # k
        self.scale = scale  # λ

    def sample(self, state=None, subject=None) -> float:
        return self.scale * np.random.weibull(self.shape)

    def survival(self, t: float) -> float:
        if t < 0:
            return 1.0
        return np.exp(-((t / self.scale) ** self.shape))

    def hazard(self, t: float) -> float:
        if t <= 0:
            return 0.0 if self.shape > 1 else float('inf') if self.shape < 1 else 1.0 / self.scale
        return (self.shape / self.scale) * ((t / self.scale) ** (self.shape - 1))


class Exponential(Weibull):
    """Exponential distribution (memoryless). Special case of Weibull with shape=1."""

    def __init__(self, rate: float):
        super().__init__(shape=1.0, scale=1.0 / rate)
        self.rate = rate

    def sample(self, state=None, subject=None) -> float:
        return np.random.exponential(1.0 / self.rate)

    def hazard(self, t: float) -> float:
        return self.rate if t >= 0 else 0.0


class LogNormal(SurvivalModel):
    """Log-normal distribution. log(T) ~ Normal(mu, sigma)."""

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        self._dist = stats.lognorm(s=sigma, scale=np.exp(mu))

    def sample(self, state=None, subject=None) -> float:
        return self._dist.rvs()

    def survival(self, t: float) -> float:
        if t <= 0:
            return 1.0
        return 1.0 - self._dist.cdf(t)


class Gamma(SurvivalModel):
    """Gamma distribution."""

    def __init__(self, shape: float, rate: float):
        self.shape = shape  # k (shape)
        self.rate = rate    # β (rate = 1/scale)
        self._dist = stats.gamma(a=shape, scale=1.0 / rate)

    def sample(self, state=None, subject=None) -> float:
        return self._dist.rvs()

    def survival(self, t: float) -> float:
        if t <= 0:
            return 1.0
        return 1.0 - self._dist.cdf(t)


# -----------------------------------------------------------------------------
# Point Masses and Defective Distributions
# -----------------------------------------------------------------------------

class DeltaMass(SurvivalModel):
    """
    Deterministic event at fixed time t0.

    Useful for scheduled events (e.g., checkups at 30, 60, 90 days).
    """

    def __init__(self, t0: float = 0.0):
        self.t0 = t0

    def sample(self, state=None, subject=None) -> float:
        return self.t0

    def survival(self, t: float) -> float:
        return 1.0 if t < self.t0 else 0.0


class PointMassPlusContinuous(SurvivalModel):
    """
    Defective distribution: point mass at t=0 plus continuous tail.

    Models simultaneous events (p0 probability) plus delayed events.
    Analogous to zero-inflated or cure-fraction models.

    S(t) = 1                           for t < 0
    S(0+) = 1 - p0                     (step down at t=0)
    S(t) = (1 - p0) * S_continuous(t)  for t > 0
    """

    def __init__(self, p0: float, continuous: SurvivalModel):
        if not 0 <= p0 <= 1:
            raise ValueError(f"p0 must be in [0, 1], got {p0}")
        self.p0 = p0
        self.continuous = continuous

    def sample(self, state=None, subject=None) -> float:
        if np.random.random() < self.p0:
            return 0.0  # Simultaneous event
        return self.continuous.sample(state, subject)

    def survival(self, t: float) -> float:
        if t < 0:
            return 1.0
        if t == 0:
            return 1.0  # Just before the point mass
        # For t > 0: those who didn't get immediate event follow continuous
        return (1 - self.p0) * self.continuous.survival(t)

    def cdf(self, t: float) -> float:
        if t < 0:
            return 0.0
        # F(t) = p0 + (1-p0) * F_continuous(t)
        return self.p0 + (1 - self.p0) * self.continuous.cdf(t)


class PointMasses(SurvivalModel):
    """
    Multiple point masses at specified times plus optional continuous tail.

    Useful for scheduled events (checkups, medication refills) combined
    with random events.
    """

    def __init__(
        self,
        point_masses: Dict[float, float],
        continuous: Optional[SurvivalModel] = None
    ):
        """
        Args:
            point_masses: {time: probability} e.g. {0: 0.2, 30: 0.1, 60: 0.1}
            continuous: optional continuous distribution for remaining probability
        """
        self.point_masses = dict(sorted(point_masses.items()))  # sorted by time
        self.total_point_mass = sum(point_masses.values())

        if self.total_point_mass > 1.0:
            raise ValueError(f"Total point mass {self.total_point_mass} exceeds 1.0")

        self.continuous = continuous
        self.continuous_weight = 1.0 - self.total_point_mass

        # Pre-compute for sampling
        self._times = list(self.point_masses.keys())
        self._probs = list(self.point_masses.values())
        if self.continuous is not None and self.continuous_weight > 0:
            self._times.append(None)  # sentinel for continuous
            self._probs.append(self.continuous_weight)

    def sample(self, state=None, subject=None) -> float:
        idx = np.random.choice(len(self._times), p=self._probs)
        t = self._times[idx]
        if t is None:
            # Sample from continuous
            return self.continuous.sample(state, subject)
        return t

    def survival(self, t: float) -> float:
        if t < 0:
            return 1.0

        # Subtract all point masses at times <= t
        mass_below = sum(p for time, p in self.point_masses.items() if time <= t)

        # Add continuous contribution
        if self.continuous is not None and self.continuous_weight > 0:
            continuous_surv = self.continuous.survival(t)
            return (1 - mass_below) - self.continuous_weight * (1 - continuous_surv)
        else:
            return 1 - mass_below


# -----------------------------------------------------------------------------
# Composite Models
# -----------------------------------------------------------------------------

class Mixture(SurvivalModel):
    """
    Mixture of survival distributions.

    S(t) = sum_i w_i * S_i(t)

    Use cases:
    - Cure fraction: Mixture([DeltaMass(inf), Weibull(...)], [p_cure, 1-p_cure])
    - Multimodal: different subpopulations with different survival
    """

    def __init__(self, models: List[SurvivalModel], weights: List[float]):
        if len(models) != len(weights):
            raise ValueError("Number of models must match number of weights")

        total = sum(weights)
        self.models = models
        self.weights = [w / total for w in weights]  # normalise

    def sample(self, state=None, subject=None) -> float:
        idx = np.random.choice(len(self.models), p=self.weights)
        return self.models[idx].sample(state, subject)

    def survival(self, t: float) -> float:
        return sum(w * m.survival(t) for w, m in zip(self.weights, self.models))


class NeverOccurs(SurvivalModel):
    """Event that never occurs. Useful for cure fractions or disabled events."""

    def sample(self, state=None, subject=None) -> float:
        return float('inf')

    def survival(self, t: float) -> float:
        return 1.0


class CompoundWeibull(SurvivalModel):
    """
    Compound of two Weibulls with additive hazards.

    h(t) = h1(t) + h2(t)
    S(t) = S1(t) × S2(t)
    sample: min(sample1, sample2)

    Useful for bathtub-shaped hazard curves, e.g., human mortality:
    - Weibull 1: shape < 1 (high early hazard, declining)
    - Weibull 2: shape > 1 (low early hazard, increasing with age)

    The combined hazard starts high, decreases, then increases again.

    Parameters: shape1, scale1, shape2, scale2 (4 total)
    """

    def __init__(
        self,
        shape1: float,
        scale1: float,
        shape2: float,
        scale2: float
    ):
        self.shape1 = shape1
        self.scale1 = scale1
        self.shape2 = shape2
        self.scale2 = scale2
        self.weibull1 = Weibull(shape1, scale1)
        self.weibull2 = Weibull(shape2, scale2)

    def sample(self, state=None, subject=None) -> float:
        t1 = self.weibull1.sample(state, subject)
        t2 = self.weibull2.sample(state, subject)
        return min(t1, t2)

    def survival(self, t: float) -> float:
        return self.weibull1.survival(t) * self.weibull2.survival(t)

    def hazard(self, t: float) -> float:
        return self.weibull1.hazard(t) + self.weibull2.hazard(t)


# -----------------------------------------------------------------------------
# State-Dependent Models (placeholders for autoregressive behaviour)
# -----------------------------------------------------------------------------

class StateDependentWeibull(Weibull):
    """
    Weibull with parameters that depend on state/subject.

    Override parameter computation in subclass or provide callables.
    """

    def __init__(
        self,
        base_shape: float,
        base_scale: float,
        shape_modifier: Optional[callable] = None,
        scale_modifier: Optional[callable] = None
    ):
        super().__init__(base_shape, base_scale)
        self.base_shape = base_shape
        self.base_scale = base_scale
        self.shape_modifier = shape_modifier
        self.scale_modifier = scale_modifier

    def sample(self, state=None, subject=None) -> float:
        shape = self.base_shape
        scale = self.base_scale

        if self.shape_modifier is not None:
            shape = self.shape_modifier(shape, state, subject)
        if self.scale_modifier is not None:
            scale = self.scale_modifier(scale, state, subject)

        return scale * np.random.weibull(shape)
