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

    State space (8 features):
        - time_since_last_service: Time since last service (or episode start)
        - service_count: Number of services so far
        - avg_service_interval: Average time between services
        - durability: Subject durability feature
        - delta_t: Age reduction per service (scenario parameter)
        - service_cost: Cost per service (scenario parameter)
        - revenue_per_time: Revenue rate (scenario parameter)
        - failure_cost: Cost on failure (scenario parameter)

    Action space (6 discrete actions):
        - 0: Immediate service (delay=0)
        - 1: delay=10
        - 2: delay=20
        - 3: delay=40
        - 4: delay=80
        - 5: No more service (delay=inf)

    Reward:
        Step-wise: revenue × time_elapsed - service_cost (if serviced)
        On failure: -failure_cost
    """

    # Action delays (index -> delay)
    # Note: 0.1 instead of 0.0 to avoid infinite loops
    ACTION_DELAYS = [0.1, 10.0, 20.0, 40.0, 80.0, float('inf')]

    metadata = {'render_modes': []}

    def __init__(
        self,
        scenario: Scenario,
        max_time: float = 150.0,
        max_steps: int = 1000,
        seed: Optional[int] = None
    ):
        """
        Args:
            scenario: Scenario defining dynamics and costs
            max_time: Maximum episode length
            max_steps: Maximum steps per episode (safety limit)
            seed: Random seed
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

        # Define spaces
        # State: 8 continuous features with reasonable upper bounds
        # Note: SB3 normalizes observations, but bounded spaces help
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([
                max_time,    # time_since_last_service
                100,         # service_count (practical upper limit)
                max_time,    # avg_service_interval
                100,         # durability (rare to exceed with typical log-normal)
                100,         # delta_t
                1000,        # service_cost
                100,         # revenue_per_time
                10000,       # failure_cost
            ], dtype=np.float32),
            dtype=np.float32
        )

        # Actions: 6 discrete choices
        self.action_space = spaces.Discrete(len(self.ACTION_DELAYS))

        # Episode state (set in reset)
        self._subject: Optional[Subject] = None
        self._state: Optional[State] = None
        self._registry: Optional[EventRegistry] = None
        self._time_of_last_service: float = 0.0
        self._service_count: int = 0
        self._total_service_time: float = 0.0  # For computing average
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

        # Create event registry, simulator, and state
        self._registry = self.scenario.create_event_registry()
        self._simulator = Simulator(self._registry, max_time=self.max_time)
        self._state = State(self._subject)

        # Reset tracking
        self._time_of_last_service = 0.0
        self._service_count = 0
        self._total_service_time = 0.0
        self._cumulative_reward = 0.0
        self._step_count = 0

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

        delay = self.ACTION_DELAYS[action]
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
                truncated = True
                break

            if step_result.event_name == 'service':
                # Service happened - subtract cost, update tracking
                reward -= self.service_cost
                self._service_count += 1
                interval = step_result.event_time - self._time_of_last_service
                self._total_service_time += interval
                self._time_of_last_service = step_result.event_time
                # Decision point - return to agent
                break

            elif step_result.terminated:
                # Failure - subtract cost, episode ends
                reward -= self.failure_cost
                terminated = True
                break

        self._cumulative_reward += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        current_time = self._state.time
        time_since_last_service = current_time - self._time_of_last_service

        # Average service interval (0 if no services yet)
        if self._service_count > 0:
            avg_interval = self._total_service_time / self._service_count
        else:
            avg_interval = 0.0

        # Clip durability to observation space bounds (rare but possible to exceed)
        durability = min(self._subject.get_feature('durability', 1.0), 100.0)

        obs = np.array([
            time_since_last_service,
            self._service_count,
            avg_interval,
            durability,
            self.delta_t,
            self.service_cost,
            self.revenue_per_time,
            self.failure_cost
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Return info dict."""
        return {
            'time': self._state.time,
            'service_count': self._service_count,
            'cumulative_reward': self._cumulative_reward,
            'durability': self._subject.get_feature('durability', 1.0) if self._subject else None
        }
