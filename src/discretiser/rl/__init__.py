"""
Reinforcement learning module for policy optimisation.

Uses Gymnasium interface with Stable-Baselines3 algorithms (DQN, SAC, etc.).
"""

from .environment import ServiceEnv

# Callbacks (shared across algorithms)
from .callbacks import (
    RewardLoggerCallback,
    EpisodeDiagnosticsCallback,
    ActionStatsCallback,
    ContinuousActionStatsCallback,
)

# Evaluation utilities
from .evaluation import (
    evaluate_model,
    compare_with_baselines,
    print_evaluation_results,
    run_sanity_check,
)

# Scenario setup
from .scenarios import (
    get_default_scenario,
    get_baseline_policies,
    print_scenario_info,
    DEFAULT_MAX_TIME,
    DEFAULT_SCENARIO_PARAMS,
    OPTIMAL_POLICY_PARAMS,
)

# DQN-specific (for backwards compatibility)
from .training import train_dqn, DQNPolicy, TrainingResult

__all__ = [
    # Environment
    'ServiceEnv',
    # Callbacks
    'RewardLoggerCallback',
    'EpisodeDiagnosticsCallback',
    'ActionStatsCallback',
    'ContinuousActionStatsCallback',
    # Evaluation
    'evaluate_model',
    'compare_with_baselines',
    'print_evaluation_results',
    'run_sanity_check',
    # Scenarios
    'get_default_scenario',
    'get_baseline_policies',
    'print_scenario_info',
    'DEFAULT_MAX_TIME',
    'DEFAULT_SCENARIO_PARAMS',
    'OPTIMAL_POLICY_PARAMS',
    # DQN-specific
    'train_dqn',
    'DQNPolicy',
    'TrainingResult',
]
