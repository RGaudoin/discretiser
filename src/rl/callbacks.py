"""
Shared callbacks for RL training.

These callbacks work with any SB3 algorithm (DQN, SAC, etc.).
"""

from typing import Optional, List, TYPE_CHECKING
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from .environment import ServiceEnv


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
    """
    Callback to log episode diagnostics to TensorBoard.

    Works with any SB3 algorithm. Logs:
    - Episode time, reward (original metric), service counts
    - Truncated/failed rates
    - Optional: evaluation with current model vs optimal baseline
    """

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
    """
    Callback to log discrete action statistics to TensorBoard.

    For use with DQN (discrete action space).
    """

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


class ContinuousActionStatsCallback(BaseCallback):
    """
    Callback to log continuous action statistics to TensorBoard.

    For use with SAC, TD3, etc. (continuous action space).
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.actions_buffer: List[float] = []

    def _on_step(self) -> bool:
        # Collect actions (continuous = delay values directly)
        if self.locals.get('actions') is not None:
            actions = self.locals['actions']
            if hasattr(actions, 'flatten'):
                self.actions_buffer.extend(actions.flatten())
            elif hasattr(actions, '__iter__'):
                self.actions_buffer.extend(actions)
            else:
                self.actions_buffer.append(float(actions))

        # Log stats every log_freq steps
        if self.num_timesteps % self.log_freq == 0 and len(self.actions_buffer) > 0:
            delays = np.array(self.actions_buffer)

            self.logger.record("actions/delay_mean", np.mean(delays))
            self.logger.record("actions/delay_min", np.min(delays))
            self.logger.record("actions/delay_max", np.max(delays))
            self.logger.record("actions/delay_std", np.std(delays))

            # Clear buffer
            self.actions_buffer = []

        return True
