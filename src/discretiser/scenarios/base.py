"""
Base classes and utilities for synthetic scenarios.

Each scenario defines ground truth dynamics for model validation.
RL/policy learning is a separate layer that plugs into scenarios.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ..simulator import SimulationResult
    from ..state import Subject
    from ..events import EventRegistry


class Scenario(ABC):
    """
    Abstract base for synthetic scenarios.

    A scenario defines:
    - Subject generation (with features)
    - Event registry (dynamics)
    - Cost/reward structure
    - Baseline policy (for training data generation)

    RL integration is handled separately via the scenario's dynamics and costs.
    """

    @abstractmethod
    def create_event_registry(self) -> 'EventRegistry':
        """Create the event registry defining scenario dynamics."""
        pass

    @abstractmethod
    def generate_subjects(self, n: int, seed: Optional[int] = None) -> List['Subject']:
        """Generate n subjects with appropriate features."""
        pass

    @abstractmethod
    def calculate_costs(self, result: 'SimulationResult') -> Dict[str, float]:
        """Calculate costs/rewards for a simulation result."""
        pass


@dataclass
class CostStructure:
    """
    Generic cost structure for scenarios with actions and terminal events.

    Costs are computed as:
        net_value = revenue - action_costs - terminal_cost

    Subclasses can override or extend for scenario-specific logic.
    """
    action_costs: Dict[str, float] = field(default_factory=dict)  # event_name -> cost
    terminal_costs: Dict[str, float] = field(default_factory=dict)  # event_name -> cost
    revenue_per_time: float = 0.0  # Revenue rate while operational

    def calculate(self, result: 'SimulationResult') -> Dict[str, float]:
        """
        Calculate costs for a simulation result.

        Returns dict with:
        - lifetime: Total simulation time
        - {event}_count: Count of each event type
        - {event}_cost: Total cost for each event type
        - revenue: Total revenue
        - net_value: revenue - costs
        """
        from ..simulator import SimulationResult

        output: Dict[str, float] = {}

        # Lifetime
        output['lifetime'] = result.final_time

        # Count events and calculate costs
        total_action_cost = 0.0
        total_terminal_cost = 0.0

        event_counts: Dict[str, int] = {}
        for record in result.history:
            event_counts[record.event_name] = event_counts.get(record.event_name, 0) + 1

        for event_name, count in event_counts.items():
            output[f'{event_name}_count'] = count

            if event_name in self.action_costs:
                cost = count * self.action_costs[event_name]
                output[f'{event_name}_cost'] = cost
                total_action_cost += cost

            if event_name in self.terminal_costs:
                # Terminal events should only happen once, but use count for safety
                cost = count * self.terminal_costs[event_name]
                output[f'{event_name}_cost'] = cost
                total_terminal_cost += cost

        # Revenue
        output['revenue'] = result.final_time * self.revenue_per_time

        # Summary
        output['total_action_cost'] = total_action_cost
        output['total_terminal_cost'] = total_terminal_cost
        output['net_value'] = (
            output['revenue']
            - total_action_cost
            - total_terminal_cost
        )

        return output


def summarise_cohort_costs(
    costs: List[Dict[str, float]],
    keys: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Summarise costs across a cohort of simulations.

    Args:
        costs: List of cost dicts from CostStructure.calculate()
        keys: Optional list of keys to summarise (default: all numeric keys)

    Returns:
        Dict mapping key -> {mean, std, min, max, sum}
    """
    import numpy as np

    if not costs:
        return {}

    # Determine keys to summarise
    if keys is None:
        keys = [k for k in costs[0].keys() if isinstance(costs[0][k], (int, float))]

    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = [c.get(key, 0.0) for c in costs]
        arr = np.array(values)
        summary[key] = {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'sum': float(np.sum(arr)),
        }

    return summary
