"""
Shared evaluation functions for RL experiments.

Provides model evaluation and baseline comparison utilities.
"""

from typing import Optional, List, Dict, Any
import numpy as np

from ..scenarios.base import Scenario
from ..runner import compare_policies
from .environment import ServiceEnv
from .scenarios import get_baseline_policies, DEFAULT_MAX_TIME


def evaluate_model(
    model,
    scenario: Scenario,
    n_episodes: int = 100,
    max_time: float = DEFAULT_MAX_TIME,
    seed: Optional[int] = None,
    deterministic: bool = True,
    action_step: float = 2.0,
    max_action_delay: float = 100.0,
    fixed_durability: Optional[float] = None,
    continuous_actions: bool = False,
) -> List[float]:
    """
    Evaluate a trained model (works with any SB3 algorithm).

    Args:
        model: Trained SB3 model (DQN, SAC, etc.)
        scenario: Scenario to evaluate on
        n_episodes: Number of evaluation episodes
        max_time: Maximum episode length
        seed: Random seed
        deterministic: Use deterministic actions
        action_step: Action delay step size (discrete only)
        max_action_delay: Maximum action delay
        fixed_durability: Fixed durability value (None = use generated)
        continuous_actions: Whether model uses continuous action space

    Returns:
        List of episode rewards (original metric, for baseline comparison)
    """
    env = ServiceEnv(
        scenario,
        max_time=max_time,
        seed=seed,
        action_step=action_step,
        max_action_delay=max_action_delay,
        fixed_durability=fixed_durability,
        continuous_actions=continuous_actions,
    )
    rewards = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i if seed else None)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        rewards.append(info['cumulative_reward_original'])

    return rewards


def compare_with_baselines(
    scenario: Scenario,
    n_subjects: int = 2000,
    max_time: float = DEFAULT_MAX_TIME,
    n_repeats: int = 5,
    seed: int = 42,
    print_results: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compare baseline policies and return results.

    Args:
        scenario: Scenario to evaluate on
        n_subjects: Subjects per evaluation
        max_time: Maximum episode length
        n_repeats: Number of repeats for statistics
        seed: Random seed
        print_results: Whether to print results

    Returns:
        Dictionary of policy name -> {'mean': ..., 'std': ...}
    """
    policies = get_baseline_policies()

    results = compare_policies(
        scenario,
        policies,
        n_subjects=n_subjects,
        max_time=max_time,
        n_repeats=n_repeats,
        seed=seed
    )

    if print_results:
        print("Baseline Policy Comparison")
        print("=" * 50)
        for name, stats in sorted(results.items(), key=lambda x: -x[1]['mean']):
            print(f"{name:20s}: mean={stats['mean']:8.2f} +/- {stats['std']:6.2f}")

    return results


def print_evaluation_results(
    rewards: List[float],
    model_name: str = "Model",
    n_episodes: Optional[int] = None,
) -> None:
    """Print evaluation results summary."""
    n = n_episodes or len(rewards)
    print(f"{model_name} Performance ({n} episodes, original metric):")
    print(f"  Mean reward: {np.mean(rewards):.2f}")
    print(f"  Std reward:  {np.std(rewards):.2f}")
    print(f"  Min reward:  {np.min(rewards):.2f}")
    print(f"  Max reward:  {np.max(rewards):.2f}")


def run_sanity_check(
    scenario: Scenario,
    max_time: float = DEFAULT_MAX_TIME,
    n_episodes: int = 100,
    seed: int = 42,
    fixed_durability: Optional[float] = 1.0,
    optimal_policy_params: tuple = (24.2, 24.2),
) -> Dict[str, Any]:
    """
    Run sanity check: optimal policy through RL environment.

    Verifies the reward machinery is working correctly.

    Args:
        scenario: Scenario to test
        max_time: Maximum episode length
        n_episodes: Number of test episodes
        seed: Random seed
        fixed_durability: Fixed durability value
        optimal_policy_params: (a, b) for interval = a + b * durability

    Returns:
        Dictionary with mean_reward, std_reward, mean_time, survival_rate
    """
    print("Sanity check: Optimal policy through RL environment")
    print("=" * 50)

    test_env = ServiceEnv(
        scenario,
        max_time=max_time,
        seed=seed,
        use_optimal_policy=True,
        fixed_durability=fixed_durability,
        optimal_policy_params=optimal_policy_params,
    )

    rewards = []
    times = []
    for ep in range(n_episodes):
        obs, _ = test_env.reset(seed=seed + ep)
        done = False
        while not done:
            obs, _, terminated, truncated, info = test_env.step(0)
            done = terminated or truncated
        rewards.append(info['cumulative_reward_original'])
        times.append(info['time'])

    survival_rate = sum(t >= max_time - 0.1 for t in times) / n_episodes

    a, b = optimal_policy_params
    if fixed_durability is not None:
        interval = a + b * fixed_durability
        print(f"Optimal policy (a={a}, b={b}, durability={fixed_durability} -> interval={interval:.1f}):")
    else:
        print(f"Optimal policy (a={a}, b={b}, variable durability):")

    print(f"  Mean reward (original): {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  Mean episode time: {np.mean(times):.1f}")
    print(f"  Truncated (survived): {int(survival_rate * n_episodes)}/{n_episodes}")

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_time': np.mean(times),
        'survival_rate': survival_rate,
    }
