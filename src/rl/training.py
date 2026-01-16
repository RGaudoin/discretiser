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


class EpisodeDiagnosticsCallback(BaseCallback):
    """Callback to log episode diagnostics to TensorBoard."""

    def __init__(
        self,
        max_time: float = 150.0,
        log_freq: int = 100,
        verbose: int = 0,
        eval_env: Optional['ServiceEnv'] = None,
        optimal_env: Optional['ServiceEnv'] = None,
        n_eval_episodes: int = 50,
    ):
        """
        Args:
            max_time: Episode max time (for detecting truncation)
            log_freq: Log stats every N episodes
            verbose: Verbosity level
            eval_env: Environment for evaluating current policy (optional)
            optimal_env: Environment with use_optimal_policy=True for baseline (optional)
            n_eval_episodes: Number of episodes for evaluation
        """
        super().__init__(verbose)
        self.max_time = max_time
        self.log_freq = log_freq
        self.eval_env = eval_env
        self.optimal_env = optimal_env
        self.n_eval_episodes = n_eval_episodes

        # Buffers for episode stats
        self.episode_times: List[float] = []
        self.episode_rewards_original: List[float] = []
        self.episode_service_counts: List[int] = []
        self.num_truncated: int = 0  # Reached max_time
        self.num_failed: int = 0     # Failed before max_time
        self.episode_count: int = 0

    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    self.episode_count += 1

                    # Collect episode stats
                    ep_time = info.get('time', 0.0)
                    self.episode_times.append(ep_time)
                    self.episode_rewards_original.append(
                        info.get('cumulative_reward_original', 0.0)
                    )
                    self.episode_service_counts.append(
                        info.get('service_count', 0)
                    )

                    # Count outcomes
                    if info.get('failed', False):
                        self.num_failed += 1
                    elif ep_time >= self.max_time - 0.1:  # Small tolerance
                        self.num_truncated += 1

                    # Log every log_freq episodes
                    if self.episode_count % self.log_freq == 0:
                        self._log_stats()

        return True

    def _log_stats(self):
        if len(self.episode_times) == 0:
            return

        times = np.array(self.episode_times)
        rewards_orig = np.array(self.episode_rewards_original)
        service_counts = np.array(self.episode_service_counts)

        # Episode time stats
        self.logger.record("episode/time_mean", np.mean(times))
        self.logger.record("episode/time_max", np.max(times))
        self.logger.record("episode/time_min", np.min(times))

        # Original reward (for baseline comparison)
        self.logger.record("episode/reward_original_mean", np.mean(rewards_orig))
        self.logger.record("episode/reward_original_std", np.std(rewards_orig))

        # Service counts
        self.logger.record("episode/services_mean", np.mean(service_counts))

        # Outcome rates (over buffer period)
        n = len(self.episode_times)
        self.logger.record("episode/truncated_rate", self.num_truncated / n)
        self.logger.record("episode/failed_rate", self.num_failed / n)

        # Run evaluation with current model (deterministic)
        if self.eval_env is not None and self.model is not None:
            eval_rewards, eval_times, eval_truncated = self._run_eval(
                self.eval_env, use_model=True
            )
            self.logger.record("eval/reward_mean", np.mean(eval_rewards))
            self.logger.record("eval/reward_std", np.std(eval_rewards))
            self.logger.record("eval/time_mean", np.mean(eval_times))
            self.logger.record("eval/truncated_rate", eval_truncated / self.n_eval_episodes)

        # Run optimal policy baseline
        if self.optimal_env is not None:
            opt_rewards, opt_times, opt_truncated = self._run_eval(
                self.optimal_env, use_model=False
            )
            self.logger.record("optimal/reward_mean", np.mean(opt_rewards))
            self.logger.record("optimal/time_mean", np.mean(opt_times))
            self.logger.record("optimal/truncated_rate", opt_truncated / self.n_eval_episodes)

        # Clear buffers
        self.episode_times = []
        self.episode_rewards_original = []
        self.episode_service_counts = []
        self.num_truncated = 0
        self.num_failed = 0

    def _run_eval(self, env, use_model: bool):
        """Run evaluation episodes."""
        rewards = []
        times = []
        truncated_count = 0

        for i in range(self.n_eval_episodes):
            obs, _ = env.reset(seed=1000 + i)
            done = False
            while not done:
                if use_model:
                    action, _ = self.model.predict(obs, deterministic=True)
                else:
                    action = 0  # Ignored when use_optimal_policy=True
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            rewards.append(info['cumulative_reward_original'])
            times.append(info['time'])
            if info['time'] >= self.max_time - 0.1:
                truncated_count += 1

        return rewards, times, truncated_count


class ActionStatsCallback(BaseCallback):
    """Callback to log action statistics to TensorBoard."""

    def __init__(self, action_delays: List[float], log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.action_delays = action_delays
        self.log_freq = log_freq
        self.actions_buffer: List[int] = []

    def _on_step(self) -> bool:
        # Collect actions
        if self.locals.get('actions') is not None:
            actions = self.locals['actions']
            if hasattr(actions, '__iter__'):
                self.actions_buffer.extend(actions)
            else:
                self.actions_buffer.append(actions)

        # Log stats every log_freq steps
        if self.num_timesteps % self.log_freq == 0 and len(self.actions_buffer) > 0:
            actions = np.array(self.actions_buffer)

            # Convert action indices to delay values for more interpretable stats
            delays = np.array([self.action_delays[a] for a in actions])
            # Replace inf with max_time for stats (otherwise mean is inf)
            finite_delays = delays[delays < float('inf')]

            if len(finite_delays) > 0:
                self.logger.record("actions/delay_mean", np.mean(finite_delays))
                self.logger.record("actions/delay_min", np.min(finite_delays))
                self.logger.record("actions/delay_max", np.max(finite_delays))
                self.logger.record("actions/delay_std", np.std(finite_delays))

            # Action index stats
            self.logger.record("actions/index_mean", np.mean(actions))
            self.logger.record("actions/index_std", np.std(actions))
            self.logger.record("actions/index_min", np.min(actions))
            self.logger.record("actions/index_max", np.max(actions))

            # Fraction choosing "no service" (inf)
            inf_frac = np.mean(delays == float('inf'))
            self.logger.record("actions/inf_fraction", inf_frac)

            # Clear buffer
            self.actions_buffer = []

        return True


def train_dqn(
    scenario: Scenario,
    total_timesteps: int = 50_000,
    learning_rate: float = 5e-5,
    buffer_size: int = 50_000,
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
    tensorboard_log: Optional[str] = None,
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
        tensorboard_log: Directory for TensorBoard logs (None=no logging)

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
        tensorboard_log=tensorboard_log,
    )

    # Set up callbacks
    callbacks = []
    reward_logger = RewardLoggerCallback()
    callbacks.append(reward_logger)

    # Log action stats to TensorBoard
    if tensorboard_log is not None:
        action_stats = ActionStatsCallback(log_freq=1000)
        callbacks.append(action_stats)

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
    use_original_metric: bool = True,
    action_step: float = 2.0,
    reward_scale: Optional[float] = None,
    fixed_durability: Optional[float] = None,
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
        use_original_metric: If True, return original reward metric (penalise failure)
                            for comparison with baselines. If False, return training
                            reward (reward survival).
        action_step: Action delay step size (must match training env)
        reward_scale: Reward scaling factor (must match training env)
        fixed_durability: Fixed durability value (must match training env)

    Returns:
        List of episode rewards
    """
    env = ServiceEnv(
        scenario,
        max_time=max_time,
        seed=seed,
        action_step=action_step,
        reward_scale=reward_scale,
        fixed_durability=fixed_durability,
    )
    rewards = []

    reward_key = 'cumulative_reward_original' if use_original_metric else 'cumulative_reward'

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i if seed else None)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        rewards.append(info[reward_key])

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
        action_delays: List[float],
        max_time: float = 150.0,
        deterministic: bool = True
    ):
        """
        Args:
            model: Trained DQN model
            scenario: Scenario (for extracting parameters)
            action_delays: List of delay values for each action index
            max_time: Maximum episode length (for normalisation)
            deterministic: Use deterministic actions
        """
        self.model = model
        self.scenario = scenario
        self.action_delays = action_delays
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
        delay = self.action_delays[int(action_idx)]

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
