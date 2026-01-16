"""
Gymnasium environment for service policy learning.

Wraps a scenario (ground-truth or learnt model) in the standard
Gymnasium interface for use with RL algorithms.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Any, Dict

from ..scenarios.base import Scenario
from ..state import State, Subject
from ..events import EventRegistry
from ..simulator import Simulator, StepResult


class ServiceEnv(gym.Env):
    """
    Gymnasium environment for learning service policies.

    The agent observes state features and decides when to schedule
    the next service. Episode ends on failure or truncation (max_time).

    State space (5 features):
        - current_time: Current simulation time
        - last_interval: Time elapsed in the interval that just ended
        - service_count: Number of services so far
        - avg_service_interval: Average time between services
        - durability: Subject durability feature

    Note: Scenario parameters (delta_t, costs) are not included as they're
    constant within a scenario. For multi-scenario training, these could be
    added to enable generalisation across different cost structures.

    Action space (configurable):
        - 0: Immediate service (delay≈0)
        - 1 to N-1: delay in steps (default step=2, giving 0.1, 2, 4, ..., 100)
        - N: No more service (delay=inf)

    Reward (training formulation - "reward survival"):
        Step-wise: (revenue × time_elapsed - service_cost) / reward_scale
        On failure: 0 (no penalty)
        On truncation (max_time): +survival_bonus / reward_scale

    Rewards are scaled by reward_scale (default=failure_cost) to keep them
    in a reasonable range (~[0, 2.5]) for more stable learning.
    The original (unscaled) metric is tracked separately for comparison.
    """

    @staticmethod
    def make_action_delays(max_delay: float = 100.0, step: float = 2.0) -> list:
        """Create action delay list with given step size."""
        # 0.1 for "immediate", then regular steps, then infinity
        delays = [0.1] + [float(d) for d in np.arange(step, max_delay + 0.1, step)] + [float('inf')]
        return delays

    # Default action delays (can be overridden via constructor)
    ACTION_DELAYS = make_action_delays.__func__(100.0, 2.0)  # 52 actions

    metadata = {'render_modes': []}

    def __init__(
        self,
        scenario: Scenario,
        max_time: float = 150.0,
        max_steps: int = 1000,
        seed: Optional[int] = None,
        reward_scale: Optional[float] = None,
        action_step: float = 2.0,
        max_action_delay: float = 100.0,
        fixed_durability: Optional[float] = None,
        use_optimal_policy: bool = False,
        optimal_policy_params: tuple = (24.2, 24.2),
    ):
        """
        Args:
            scenario: Scenario defining dynamics and costs
            max_time: Maximum episode length
            max_steps: Maximum steps per episode (safety limit)
            seed: Random seed
            reward_scale: Scale factor for rewards (default=failure_cost).
                         Rewards are divided by this to keep them in ~[0, 2.5].
            action_step: Step size for action delays (default=2.0).
                        Smaller = denser action space.
            max_action_delay: Maximum delay in action space (default=100.0)
            fixed_durability: If set, override subject durability with this value.
                             Useful for debugging (removes heterogeneity).
            use_optimal_policy: If True, ignore agent actions and use optimal
                               linear policy. For debugging reward machinery.
            optimal_policy_params: (a, b) for interval = a + b * durability
        """
        super().__init__()

        self.scenario = scenario
        self.max_time = max_time
        self.max_steps = max_steps

        # Extract scenario parameters for state
        self.delta_t = getattr(scenario.failure_model, 'delta_t', 0.0)
        self.service_cost = scenario.costs.service_cost
        self.revenue_per_time = scenario.costs.revenue_per_time
        self.failure_cost = scenario.costs.failure_cost

        # Reward scaling (default to failure_cost for ~[0, 2.5] range)
        self.reward_scale = reward_scale if reward_scale is not None else self.failure_cost

        # Fixed durability for debugging (removes heterogeneity)
        self.fixed_durability = fixed_durability

        # Debug: use optimal policy instead of agent actions
        self.use_optimal_policy = use_optimal_policy
        self.optimal_policy_params = optimal_policy_params

        # Build action space
        self.action_delays = self.make_action_delays(max_action_delay, action_step)

        # Define spaces
        # State: 5 normalised features (divided by max for 0-1 range)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([
                1.0,    # current_time / max_time
                1.0,    # last_interval / max_time
                1.0,    # service_count / 100
                1.0,    # avg_service_interval / max_time
                1.0,    # durability / 10 (log-normal, rarely exceeds 5)
            ], dtype=np.float32),
            dtype=np.float32
        )

        # Actions: discrete choices based on action_delays
        self.action_space = spaces.Discrete(len(self.action_delays))

        # Episode state (set in reset)
        self._subject: Optional[Subject] = None
        self._state: Optional[State] = None
        self._registry: Optional[EventRegistry] = None
        self._time_of_last_service: float = 0.0
        self._service_count: int = 0
        self._total_service_time: float = 0.0  # For computing average
        self._last_interval: float = 0.0  # Interval that just ended
        self._cumulative_reward: float = 0.0

        # Seed
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        else:
            self._np_random = np.random.default_rng()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to start new episode."""
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # Generate new subject
        subjects = self.scenario.generate_subjects(1, seed=self._np_random.integers(0, 2**31))
        self._subject = subjects[0]

        # Override durability if fixed (for debugging)
        if self.fixed_durability is not None:
            self._subject.features['durability'] = self.fixed_durability

        # Create event registry, simulator, and state
        self._registry = self.scenario.create_event_registry()
        self._simulator = Simulator(self._registry, max_time=self.max_time)
        self._state = State(self._subject)

        # Reset tracking
        self._time_of_last_service = 0.0
        self._service_count = 0
        self._total_service_time = 0.0
        self._last_interval = 0.0  # No interval yet at episode start
        self._cumulative_reward = 0.0  # Training reward (reward survival)
        self._cumulative_reward_original = 0.0  # Original metric (penalise failure)
        self._step_count = 0
        self._failed = False  # Track if episode ended in failure

        return self._get_observation(), self._get_info()

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute action and advance simulation.

        Args:
            action: Index into ACTION_DELAYS

        Returns:
            observation, reward, terminated, truncated, info
        """
        self._step_count += 1

        # Safety limit
        if self._step_count >= self.max_steps:
            return self._get_observation(), 0.0, False, True, self._get_info()

        # Get delay from action (or override with optimal policy for debugging)
        if self.use_optimal_policy:
            a, b = self.optimal_policy_params
            durability = self._subject.get_feature('durability', 1.0)
            delay = a + b * durability
        else:
            delay = self.action_delays[action]

        current_time = self._state.time

        # Schedule service (unless delay is inf)
        if delay < float('inf'):
            scheduled_time = current_time + delay
            if scheduled_time < self.max_time:
                self._state.add_pending_event('service', scheduled_time, triggered_by='policy')

        # Run simulation until next decision point (service, failure, or max_time)
        # Note: This "run until decision event" pattern is extensible to:
        # - Multiple decision event types (not just service)
        # - Multiple action types scheduling different events
        # - Multi-agent RL where different agents handle different decision events
        reward = 0.0
        terminated = False
        truncated = False
        prev_time = self._state.time

        while not terminated and not truncated:
            # Use simulator's step_one_event
            step_result = self._simulator.step_one_event(self._state, self._subject)

            # Compute reward for time elapsed
            time_elapsed = self._state.time - prev_time
            reward += self.revenue_per_time * time_elapsed
            prev_time = self._state.time

            if step_result.truncated:
                # Survived to max_time - add survival bonus (training reward)
                reward += self.failure_cost  # survival_bonus = failure_cost
                truncated = True
                break

            if step_result.event_name == 'service':
                # Service happened - subtract cost, update tracking
                reward -= self.service_cost
                self._service_count += 1
                interval = step_result.event_time - self._time_of_last_service
                self._total_service_time += interval
                self._last_interval = interval  # Store BEFORE updating time
                self._time_of_last_service = step_result.event_time
                # Decision point - return to agent
                break

            elif step_result.terminated:
                # Failure - no penalty in training reward (reward survival formulation)
                # Original formulation would subtract failure_cost here
                self._failed = True
                terminated = True
                break

        # Compute original metric (unscaled, for comparison with baselines)
        reward_original = reward
        if terminated:  # Failed this step
            reward_original -= self.failure_cost  # Add the failure penalty
        if truncated:  # Survived to max_time
            reward_original -= self.failure_cost  # Remove the survival bonus
        self._cumulative_reward_original += reward_original

        # Scale reward for training (keeps Q-values in reasonable range)
        reward_scaled = reward / self.reward_scale
        self._cumulative_reward += reward_scaled

        return self._get_observation(), reward_scaled, terminated, truncated, self._get_info()

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector (normalised by dividing by max)."""
        current_time = self._state.time

        # Average service interval (0 if no services yet)
        if self._service_count > 0:
            avg_interval = self._total_service_time / self._service_count
        else:
            avg_interval = 0.0

        durability = self._subject.get_feature('durability', 1.0)

        obs = np.array([
            current_time / self.max_time,
            self._last_interval / self.max_time,
            self._service_count / 100.0,
            avg_interval / self.max_time,
            durability / 10.0,  # log-normal with mean~1, rarely exceeds 5
        ], dtype=np.float32)

        # Clip to observation space bounds (safety)
        return np.clip(obs, 0.0, 1.0)

    def _get_info(self) -> Dict[str, Any]:
        """Return info dict."""
        return {
            'time': self._state.time,
            'service_count': self._service_count,
            'cumulative_reward': self._cumulative_reward,  # Training reward
            'cumulative_reward_original': self._cumulative_reward_original,  # For comparison
            'failed': self._failed,
            'durability': self._subject.get_feature('durability', 1.0) if self._subject else None
        }
