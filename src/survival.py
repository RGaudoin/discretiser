"""
Survival model classes for competing risks simulation.

Each model implements:
- sample(state, subject) -> float: sample time to event
- survival(t) -> float: S(t) = P(T > t)
- hazard(t) -> float: h(t) = f(t) / S(t)  (optional)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, Any, TYPE_CHECKING
import warnings

import numpy as np
from scipy import stats
from scipy.optimize import brentq


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

    def inverse_survival(self, u: float, sampling_method: str = 'auto') -> float:
        """
        Inverse survival function: find t such that S(t) = u.

        Args:
            u: Target survival probability in (0, 1]
            sampling_method: One of:
                - 'auto': use analytic if available, else numerical (default)
                - 'analytic': use closed-form (error if not available)
                - 'numerical': always use root-finding

        Returns:
            Time t where S(t) = u

        Default implementation uses numerical root-finding.
        Subclasses should override with closed-form solutions where available.
        """
        if sampling_method not in ('auto', 'analytic', 'numerical'):
            raise ValueError(
                f"sampling_method must be 'auto', 'analytic', or 'numerical', "
                f"got '{sampling_method}'"
            )

        if sampling_method == 'analytic':
            raise NotImplementedError(
                f"{self.__class__.__name__} does not have an analytic inverse_survival. "
                f"Use sampling_method='auto' or 'numerical'."
            )

        return self._inverse_survival_numerical(u)

    def _inverse_survival_numerical(self, u: float) -> float:
        """Numerical inverse using root-finding. For internal use."""
        if u <= 0:
            return float('inf')
        if u >= 1:
            return 0.0

        t_upper = 1.0
        while self.survival(t_upper) > u and t_upper < 1e10:
            t_upper *= 2
        if t_upper >= 1e10:
            return float('inf')

        return brentq(lambda t: self.survival(t) - u, 0, t_upper)

    def truncate(
        self,
        elapsed_time: float,
        sampling_method: str = 'auto'
    ) -> 'SurvivalModel':
        """
        Return a truncated version conditioned on T > elapsed_time.

        The truncated model represents the remaining time distribution
        given survival up to elapsed_time.

        Args:
            elapsed_time: Time already elapsed (t₀)
            sampling_method: Method for inverse_survival ('auto', 'analytic', 'numerical')

        Returns:
            New SurvivalModel where:
            - S'(t) = S(t + t₀) / S(t₀)
            - h'(t) = h(t + t₀)
            - sample() draws from conditional distribution
        """
        if elapsed_time <= 0:
            return self
        return TruncatedSurvival(self, elapsed_time, sampling_method)


# -----------------------------------------------------------------------------
# Truncated Survival Wrapper
# -----------------------------------------------------------------------------

class TruncatedSurvival(SurvivalModel):
    """
    Truncated survival model conditioned on T > elapsed_time.

    Represents the remaining time distribution given survival up to elapsed_time.
    For a base distribution with survival S(t) and hazard h(t), the truncated
    version has:
    - S'(t) = S(t + t₀) / S(t₀)
    - h'(t) = h(t + t₀)  (hazard just shifts)

    Sampling uses inverse transform with scaled uniform.
    """

    def __init__(
        self,
        base: SurvivalModel,
        elapsed_time: float,
        sampling_method: str = 'auto'
    ):
        """
        Args:
            base: The original survival model
            elapsed_time: Time already elapsed (t₀), must be > 0
            sampling_method: Method for inverse_survival ('auto', 'analytic', 'numerical')
        """
        if elapsed_time <= 0:
            raise ValueError(f"elapsed_time must be positive, got {elapsed_time}")
        if sampling_method not in ('auto', 'analytic', 'numerical'):
            raise ValueError(
                f"sampling_method must be 'auto', 'analytic', or 'numerical', "
                f"got '{sampling_method}'"
            )
        self.base = base
        self.elapsed = elapsed_time
        self.sampling_method = sampling_method
        self._s_elapsed = base.survival(elapsed_time)

        if self._s_elapsed <= 0:
            raise ValueError(
                f"Cannot truncate at elapsed_time={elapsed_time}: "
                f"S({elapsed_time}) = 0, event would have occurred"
            )

    def sample(self, state=None, subject=None) -> float:
        """
        Sample from truncated distribution using inverse transform.

        Scale uniform to (0, S(t₀)] then use base inverse.
        """
        u = np.random.uniform(0, 1)
        u_scaled = u * self._s_elapsed
        t_absolute = self.base.inverse_survival(u_scaled, self.sampling_method)
        return t_absolute - self.elapsed

    def survival(self, t: float) -> float:
        """S'(t) = S(t + t₀) / S(t₀)"""
        if t < 0:
            return 1.0
        return self.base.survival(t + self.elapsed) / self._s_elapsed

    def hazard(self, t: float) -> float:
        """h'(t) = h(t + t₀) - hazard just shifts"""
        return self.base.hazard(t + self.elapsed)

    def inverse_survival(self, u: float, sampling_method: str = None) -> float:
        """
        Inverse of truncated survival.

        S'(t) = u means S(t + t₀) / S(t₀) = u
        So S(t + t₀) = u * S(t₀), and t = base.inverse(u * S(t₀)) - t₀

        Args:
            u: Target survival probability
            sampling_method: Override instance method if provided
        """
        method = sampling_method if sampling_method is not None else self.sampling_method

        if u <= 0:
            return float('inf')
        if u >= 1:
            return 0.0
        u_scaled = u * self._s_elapsed
        return self.base.inverse_survival(u_scaled, method) - self.elapsed

    def truncate(
        self,
        elapsed_time: float,
        sampling_method: str = None
    ) -> 'SurvivalModel':
        """Truncating a truncated model: just add the elapsed times."""
        if elapsed_time <= 0:
            return self
        # Use provided method or inherit from self
        method = sampling_method if sampling_method is not None else self.sampling_method
        return TruncatedSurvival(self.base, self.elapsed + elapsed_time, method)


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

    def inverse_survival(self, u: float, sampling_method: str = 'auto') -> float:
        """Closed-form inverse: t = λ * (-ln(u))^(1/k)"""
        if sampling_method == 'numerical':
            return self._inverse_survival_numerical(u)

        if u <= 0:
            return float('inf')
        if u >= 1:
            return 0.0
        return self.scale * ((-np.log(u)) ** (1.0 / self.shape))


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

    def inverse_survival(self, u: float, sampling_method: str = 'auto') -> float:
        """Use scipy's ppf: S(t) = u means F(t) = 1 - u"""
        if sampling_method == 'numerical':
            return self._inverse_survival_numerical(u)

        if u <= 0:
            return float('inf')
        if u >= 1:
            return 0.0
        return self._dist.ppf(1.0 - u)


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

    def inverse_survival(self, u: float, sampling_method: str = 'auto') -> float:
        """Use scipy's ppf: S(t) = u means F(t) = 1 - u"""
        if sampling_method == 'numerical':
            return self._inverse_survival_numerical(u)

        if u <= 0:
            return float('inf')
        if u >= 1:
            return 0.0
        return self._dist.ppf(1.0 - u)


# -----------------------------------------------------------------------------
# Point Masses and Defective Distributions
# -----------------------------------------------------------------------------

class DeltaMass(SurvivalModel):
    """
    Deterministic event at fixed time t0.

    Useful for scheduled events (e.g., checkups at 30, 60, 90 days).
    """

    def __new__(cls, t0: float = 0.0):
        if np.isinf(t0):
            warnings.warn(
                "DeltaMass(inf) is deprecated; use NeverOccurs() instead",
                DeprecationWarning,
                stacklevel=2
            )
            return NeverOccurs()
        return super().__new__(cls)

    def __init__(self, t0: float = 0.0):
        if np.isinf(t0):
            return  # Already handled by __new__, skip init
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
    h(t) = [sum_i w_i * f_i(t)] / S(t)   (NOT the weighted average of hazards!)

    Use cases:
    - Cure fraction: Mixture([NeverOccurs(), Weibull(...)], [p_cure, 1-p_cure])
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

    def hazard(self, t: float) -> float:
        """
        Mixture hazard: h(t) = f(t) / S(t) = [sum_i w_i h_i(t) S_i(t)] / S(t)

        Note: This is NOT the weighted average of component hazards.
        For a cure model, the hazard decreases over time as the uncured
        population is depleted and only cured (immortal) individuals remain.
        """
        s_t = self.survival(t)
        if s_t < 1e-10:
            return float('inf')

        # f(t) = sum_i w_i * f_i(t) = sum_i w_i * h_i(t) * S_i(t)
        f_t = sum(
            w * m.hazard(t) * m.survival(t)
            for w, m in zip(self.weights, self.models)
        )
        return f_t / s_t

    def truncate(
        self,
        elapsed_time: float,
        sampling_method: str = 'auto'
    ) -> 'SurvivalModel':
        """
        Truncated mixture: mixture with updated weights and truncated components.

        New weights: w'_i = w_i * S_i(t₀) / S(t₀)

        The component with higher survival at t₀ gets more weight, reflecting
        that if you've survived this long, you're more likely from the
        longer-lived component.
        """
        if elapsed_time <= 0:
            return self

        s_total = self.survival(elapsed_time)
        if s_total <= 0:
            raise ValueError(
                f"Cannot truncate mixture at elapsed_time={elapsed_time}: "
                f"S({elapsed_time}) = 0"
            )

        # Compute updated weights: w'_i = w_i * S_i(t₀) / S(t₀)
        new_weights = [
            w * m.survival(elapsed_time) / s_total
            for w, m in zip(self.weights, self.models)
        ]

        # Truncate each component with same sampling method
        truncated_models = [
            m.truncate(elapsed_time, sampling_method) for m in self.models
        ]

        return Mixture(truncated_models, new_weights)


class NeverOccurs(SurvivalModel):
    """Event that never occurs. Useful for cure fractions or disabled events."""

    def sample(self, state=None, subject=None) -> float:
        return float('inf')

    def survival(self, t: float) -> float:
        return 1.0

    def inverse_survival(self, u: float, sampling_method: str = 'auto') -> float:
        """S(t) = 1 for all t, so inverse is 0 for u=1, inf otherwise."""
        # No difference between methods for NeverOccurs
        return 0.0 if u >= 1 else float('inf')

    def truncate(
        self,
        elapsed_time: float,
        sampling_method: str = 'auto'
    ) -> 'SurvivalModel':
        """Truncating NeverOccurs still never occurs."""
        return self


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

    This is a convenience wrapper around MinSurvival([Weibull, Weibull]).
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
        self._min_survival = MinSurvival([self.weibull1, self.weibull2])

    def sample(self, state=None, subject=None) -> float:
        return self._min_survival.sample(state, subject)

    def survival(self, t: float) -> float:
        return self._min_survival.survival(t)

    def hazard(self, t: float) -> float:
        return self._min_survival.hazard(t)

    def truncate(
        self,
        elapsed_time: float,
        sampling_method: str = 'auto'
    ) -> 'SurvivalModel':
        return self._min_survival.truncate(elapsed_time, sampling_method)


class MinSurvival(SurvivalModel):
    """
    Minimum of multiple survival distributions (additive hazards).

    S(t) = prod_i S_i(t)
    h(t) = sum_i h_i(t)
    sample: min(sample_i for all i)

    This is the competing risks model where the first event to occur wins.
    More general than CompoundWeibull (works with any survival models).
    """

    def __init__(self, models: List[SurvivalModel]):
        if not models:
            raise ValueError("MinSurvival requires at least one model")
        self.models = models

    def sample(self, state=None, subject=None) -> float:
        return min(m.sample(state, subject) for m in self.models)

    def survival(self, t: float) -> float:
        result = 1.0
        for m in self.models:
            result *= m.survival(t)
        return result

    def hazard(self, t: float) -> float:
        return sum(m.hazard(t) for m in self.models)

    def truncate(
        self,
        elapsed_time: float,
        sampling_method: str = 'auto'
    ) -> 'SurvivalModel':
        """
        Truncated min: min of truncated components.

        S(t + t₀) / S(t₀) = prod_i [S_i(t + t₀) / S_i(t₀)]
        """
        if elapsed_time <= 0:
            return self
        return MinSurvival([
            m.truncate(elapsed_time, sampling_method) for m in self.models
        ])


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


# -----------------------------------------------------------------------------
# Trained Model Integration
# -----------------------------------------------------------------------------

class TrainedModelSurvival(SurvivalModel):
    """
    Wrapper for externally trained survival models.

    Connects any predictor with a predict(features) -> params interface
    to the simulation framework. The predictor can be:
    - A pickled sklearn/keras model
    - An ONNX runtime session
    - Any object with predict(features) -> array of parameters

    Example:
        predictor = joblib.load('trained_model.pkl')
        survival = TrainedModelSurvival(predictor, ['diagnosis', 'treatment'])
        events = [EventType('outcome', survival, terminal=True)]
    """

    SUPPORTED_DISTRIBUTIONS = {'weibull', 'exponential', 'lognormal'}

    def __init__(
        self,
        predictor,
        event_names: List[str],
        distribution: str = 'weibull'
    ):
        """
        Args:
            predictor: Object with predict(features) -> params method
            event_names: Event names for feature extraction via State.to_feature_vector()
            distribution: One of 'weibull', 'exponential', 'lognormal'
        """
        if distribution not in self.SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of {self.SUPPORTED_DISTRIBUTIONS}, "
                f"got '{distribution}'"
            )
        self.predictor = predictor
        self.event_names = event_names
        self.distribution = distribution
        self._last_params = None  # Cache for survival/hazard calculations

    def sample(self, state=None, subject=None) -> float:
        """Sample time-to-event using predicted parameters."""
        if state is None:
            raise ValueError("TrainedModelSurvival requires state for prediction")

        features = state.to_feature_vector(self.event_names)
        params = self.predictor.predict(features[np.newaxis, :])[0]
        self._last_params = params

        if self.distribution == 'weibull':
            shape, scale = params[0], params[1]
            return scale * np.random.weibull(shape)

        elif self.distribution == 'exponential':
            rate = params[0]
            return np.random.exponential(1.0 / rate)

        elif self.distribution == 'lognormal':
            mu, sigma = params[0], params[1]
            return np.random.lognormal(mu, sigma)

    def survival(self, t: float) -> float:
        """
        Survival function using last predicted parameters.

        Note: This uses cached parameters from the last sample() call.
        For accurate survival curves, re-predict for the specific state.
        """
        if self._last_params is None:
            raise RuntimeError(
                "No parameters available. Call sample() first or use "
                "predict_survival() with explicit state."
            )

        if self.distribution == 'weibull':
            shape, scale = self._last_params[0], self._last_params[1]
            if t < 0:
                return 1.0
            return np.exp(-((t / scale) ** shape))

        elif self.distribution == 'exponential':
            rate = self._last_params[0]
            if t < 0:
                return 1.0
            return np.exp(-rate * t)

        elif self.distribution == 'lognormal':
            mu, sigma = self._last_params[0], self._last_params[1]
            if t <= 0:
                return 1.0
            return 1.0 - stats.lognorm(s=sigma, scale=np.exp(mu)).cdf(t)

    def predict_params(self, state) -> np.ndarray:
        """
        Get predicted parameters for a given state without sampling.

        Useful for inspecting what the model predicts.
        """
        features = state.to_feature_vector(self.event_names)
        return self.predictor.predict(features[np.newaxis, :])[0]
