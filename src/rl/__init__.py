"""
Reinforcement learning module for policy optimisation.

Uses Gymnasium interface with Stable-Baselines3 DQN.
"""

from .environment import ServiceEnv
from .training import (
    train_dqn, evaluate_model, DQNPolicy, TrainingResult,
    RewardLoggerCallback, ActionStatsCallback, EpisodeDiagnosticsCallback,
)

__all__ = [
    'ServiceEnv', 'train_dqn', 'evaluate_model', 'DQNPolicy', 'TrainingResult',
    'RewardLoggerCallback', 'ActionStatsCallback', 'EpisodeDiagnosticsCallback',
]
