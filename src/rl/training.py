"""
Training utilities for RL policy learning.

Provides functions for training DQN on ServiceEnv and evaluating
learned policies against baselines.
"""

from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from .environment import ServiceEnv
from ..scenarios.base import Scenario
from ..policy import Policy, Action
from ..state import State, Subject


@dataclass
class TrainingResult:
    """Result of training a DQN model."""
    model: DQN
    final_mean_reward: float
    training_rewards: List[float]
    eval_rewards: Optional[List[float]] = None


class RewardLoggerCallback(BaseCallback):
    """Callback to log episode rewards during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []

    def _on_step(self) -> bool:
        # Check if episode finished
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    if 'cumulative_reward' in info:
                        self.episode_rewards.append(info['cumulative_reward'])
        return True


def train_dqn(
    scenario: Scenario,
    total_timesteps: int = 50_000,
    learning_rate: float = 1e-4,
    buffer_size: int = 10_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    exploration_fraction: float = 0.3,
    exploration_final_eps: float = 0.05,
    target_update_interval: int = 1000,
    max_time: float = 150.0,
    seed: Optional[int] = None,
    verbose: int = 1,
    eval_freq: Optional[int] = None,
    n_eval_episodes: int = 20,
) -> TrainingResult:
    """
    Train a DQN model on a scenario.

    Args:
        scenario: Scenario defining dynamics and costs
        total_timesteps: Total training steps
        learning_rate: Learning rate for optimizer
        buffer_size: Size of replay buffer
        batch_size: Batch size for training
        gamma: Discount factor
        exploration_fraction: Fraction of training for epsilon decay
        exploration_final_eps: Final exploration rate
        target_update_interval: Steps between target network updates
        max_time: Maximum episode length
        seed: Random seed
        verbose: Verbosity level (0=none, 1=info)
        eval_freq: Evaluate every N steps (None=no eval)
        n_eval_episodes: Episodes per evaluation

    Returns:
        TrainingResult with trained model and metrics
    """
    # Create training environment
    env = ServiceEnv(scenario, max_time=max_time, seed=seed)
    env = Monitor(env)

    # Create model
    model = DQN(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update_interval,
        verbose=verbose,
        seed=seed,
    )

    # Set up callbacks
    callbacks = []
    reward_logger = RewardLoggerCallback()
    callbacks.append(reward_logger)

    eval_rewards = None
    if eval_freq is not None:
        eval_env = ServiceEnv(scenario, max_time=max_time, seed=seed)
        eval_env = Monitor(eval_env)
        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            verbose=verbose,
        )
        callbacks.append(eval_callback)

    # Train
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # Final evaluation
    final_rewards = evaluate_model(model, scenario, n_episodes=50, max_time=max_time, seed=seed)

    return TrainingResult(
        model=model,
        final_mean_reward=np.mean(final_rewards),
        training_rewards=reward_logger.episode_rewards,
        eval_rewards=eval_rewards,
    )


def evaluate_model(
    model: DQN,
    scenario: Scenario,
    n_episodes: int = 100,
    max_time: float = 150.0,
    seed: Optional[int] = None,
    deterministic: bool = True,
) -> List[float]:
    """
    Evaluate a trained model.

    Args:
        model: Trained DQN model
        scenario: Scenario to evaluate on
        n_episodes: Number of evaluation episodes
        max_time: Maximum episode length
        seed: Random seed
        deterministic: Use deterministic actions

    Returns:
        List of episode rewards
    """
    env = ServiceEnv(scenario, max_time=max_time, seed=seed)
    rewards = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i if seed else None)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        rewards.append(info['cumulative_reward'])

    return rewards


class DQNPolicy(Policy):
    """
    Policy wrapper for trained DQN model.

    Allows using a trained DQN in the standard Policy interface
    for comparison with LinearIntervalPolicy etc.
    """

    def __init__(
        self,
        model: DQN,
        scenario: Scenario,
        max_time: float = 150.0,
        deterministic: bool = True
    ):
        """
        Args:
            model: Trained DQN model
            scenario: Scenario (for extracting parameters)
            max_time: Maximum episode length (for normalisation)
            deterministic: Use deterministic actions
        """
        self.model = model
        self.scenario = scenario
        self.max_time = max_time
        self.deterministic = deterministic

        # Track state for observation construction
        self._time_of_last_service = 0.0
        self._service_count = 0
        self._total_service_time = 0.0
        self._last_interval = 0.0

    def reset(self) -> None:
        """Reset internal state tracking."""
        self._time_of_last_service = 0.0
        self._service_count = 0
        self._total_service_time = 0.0
        self._last_interval = 0.0

    def get_action(self, state: State, subject: Subject) -> Optional[Action]:
        """Get action from DQN model."""
        # Build observation (must match ServiceEnv._get_observation)
        current_time = state.time

        if self._service_count > 0:
            avg_interval = self._total_service_time / self._service_count
        else:
            avg_interval = 0.0

        durability = subject.get_feature('durability', 1.0)

        obs = np.array([
            current_time / self.max_time,
            self._last_interval / self.max_time,
            self._service_count / 100.0,
            avg_interval / self.max_time,
            durability / 10.0,
        ], dtype=np.float32)
        obs = np.clip(obs, 0.0, 1.0)

        # Get action from model
        action_idx, _ = self.model.predict(obs, deterministic=self.deterministic)
        delay = ServiceEnv.ACTION_DELAYS[int(action_idx)]

        # Update tracking (will be called after service event)
        # Note: This is called BEFORE the service happens, so we're predicting
        # what delay to use. The tracking update happens via record_service().

        if delay == float('inf'):
            return None

        return Action(event_name='service', delay=delay)

    def record_service(self, time: float) -> None:
        """
        Record that a service occurred (call from simulation loop).

        This updates internal tracking for observation construction.
        """
        interval = time - self._time_of_last_service
        self._total_service_time += interval
        self._last_interval = interval
        self._service_count += 1
        self._time_of_last_service = time
