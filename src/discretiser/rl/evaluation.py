"""
Shared evaluation functions for RL experiments.

Provides model evaluation and baseline comparison utilities.
"""

from typing import Optional, List, Dict, Any, Callable
import numpy as np

from ..scenarios.base import Scenario
from ..runner import compare_policies
from .environment import ServiceEnv
from .scenarios import get_baseline_policies, DEFAULT_MAX_TIME


# =============================================================================
# Helper functions
# =============================================================================

def _run_episodes(
    env: ServiceEnv,
    n_episodes: int,
    action_fn: Callable[[np.ndarray], Any],
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Core episode loop for evaluation.

    Args:
        env: ServiceEnv instance
        n_episodes: Number of episodes to run
        action_fn: Function that takes observation and returns action
        seed: Base random seed

    Returns:
        Dictionary with 'rewards', 'times', 'truncated_count', 'failed_count'
    """
    rewards = []
    times = []
    truncated_count = 0
    failed_count = 0

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i if seed else None)
        done = False
        while not done:
            action = action_fn(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        rewards.append(info['cumulative_reward_original'])
        times.append(info['time'])
        if truncated:
            truncated_count += 1
        if info.get('failed', False):
            failed_count += 1

    return {
        'rewards': rewards,
        'times': times,
        'truncated_count': truncated_count,
        'failed_count': failed_count,
    }


def format_stats(
    rewards: List[float],
    label: str = "",
    show_minmax: bool = False,
    times: Optional[List[float]] = None,
    truncated_count: Optional[int] = None,
    n_episodes: Optional[int] = None,
) -> str:
    """
    Format evaluation statistics consistently.

    Reports mean ± SEM with sample std and n for context.

    Args:
        rewards: List of episode rewards
        label: Optional label prefix
        show_minmax: Whether to show min/max
        times: Optional list of episode times
        truncated_count: Optional count of truncated episodes
        n_episodes: Override for n (defaults to len(rewards))

    Returns:
        Formatted string with statistics
    """
    n = n_episodes or len(rewards)
    mean = np.mean(rewards)
    std = np.std(rewards)
    sem = std / np.sqrt(n)

    lines = []
    prefix = f"{label}: " if label else ""
    lines.append(f"{prefix}mean={mean:.2f} ± {sem:.2f}  (σ={std:.1f}, n={n})")

    if show_minmax:
        lines.append(f"  min={np.min(rewards):.2f}, max={np.max(rewards):.2f}")

    if times is not None:
        lines.append(f"  mean time={np.mean(times):.1f}")

    if truncated_count is not None:
        rate = truncated_count / n
        lines.append(f"  truncated (survived): {truncated_count}/{n} ({rate:.1%})")

    return "\n".join(lines)


# =============================================================================
# Public API
# =============================================================================

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
) -> Dict[str, Any]:
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
        Dictionary with 'rewards', 'times', 'truncated_count', 'failed_count'
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

    def action_fn(obs):
        action, _ = model.predict(obs, deterministic=deterministic)
        return action

    return _run_episodes(env, n_episodes, action_fn, seed)


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

    Uses simulator-based evaluation (not RL env). Each repeat runs n_subjects,
    and statistics are computed across repeat means.

    Args:
        scenario: Scenario to evaluate on
        n_subjects: Subjects per evaluation batch
        max_time: Maximum episode length
        n_repeats: Number of repeats for statistics
        seed: Random seed
        print_results: Whether to print results

    Returns:
        Dictionary of policy name -> {'mean': ..., 'std': ..., 'values': ...}
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
        print(f"Baseline Policy Comparison ({n_repeats} repeats × {n_subjects} subjects)")
        print("=" * 50)
        for name, stats in sorted(results.items(), key=lambda x: -x[1]['mean']):
            # std here is std of batch means ≈ SEM
            print(f"{name:20s}: mean={stats['mean']:8.2f} ± {stats['std']:5.2f}")

    return results


def print_evaluation_results(
    results: Dict[str, Any],
    model_name: str = "Model",
    show_minmax: bool = True,
) -> None:
    """
    Print evaluation results summary.

    Args:
        results: Dictionary from evaluate_model with 'rewards', 'times', etc.
        model_name: Name to display
        show_minmax: Whether to show min/max values
    """
    rewards = results['rewards']
    n = len(rewards)
    print(f"{model_name} Performance ({n} episodes, original metric):")
    print(format_stats(
        rewards,
        show_minmax=show_minmax,
        times=results.get('times'),
        truncated_count=results.get('truncated_count'),
    ))


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
        Dictionary with 'rewards', 'times', 'truncated_count', 'failed_count'
    """
    print("Sanity check: Optimal policy through RL environment")
    print("=" * 50)

    env = ServiceEnv(
        scenario,
        max_time=max_time,
        seed=seed,
        use_optimal_policy=True,
        fixed_durability=fixed_durability,
        optimal_policy_params=optimal_policy_params,
    )

    # Optimal policy is built into env, action is ignored
    results = _run_episodes(env, n_episodes, action_fn=lambda obs: 0, seed=seed)

    # Print header with policy info
    a, b = optimal_policy_params
    if fixed_durability is not None:
        interval = a + b * fixed_durability
        print(f"Optimal policy (a={a}, b={b}, durability={fixed_durability} -> interval={interval:.1f}):")
    else:
        print(f"Optimal policy (a={a}, b={b}, variable durability):")

    # Print stats using consistent format
    print(format_stats(
        results['rewards'],
        times=results['times'],
        truncated_count=results['truncated_count'],
    ))

    return results
