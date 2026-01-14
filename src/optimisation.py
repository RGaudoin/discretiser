"""
Parameterised policy optimisation.

Scenario-agnostic: works with any Scenario + Policy combination.
Not RL - just numerical optimisation over policy parameters.
"""

import numpy as np
from scipy import optimize
from typing import Callable, Dict, List, Tuple, Optional, Any, Type
from dataclasses import dataclass
import time

from .policy import Policy
from .runner import run_scenario, RunResult
from .scenarios.base import Scenario


@dataclass
class OptimisationResult:
    """Result of policy parameter optimisation."""
    optimal_params: np.ndarray
    optimal_value: float
    initial_params: np.ndarray
    initial_value: float
    improvement: float
    improvement_pct: float
    n_evaluations: int
    elapsed_time: float
    history: List[Tuple[np.ndarray, float]]
    converged: bool


def evaluate_policy_params(
    params: np.ndarray,
    scenario: Scenario,
    policy_factory: Callable[[np.ndarray], Policy],
    n_subjects: int = 500,
    max_time: float = 200.0,
    seed: Optional[int] = None
) -> float:
    """
    Evaluate policy parameters by simulation.

    Args:
        params: Policy parameters as array
        scenario: Scenario to simulate
        policy_factory: Function that creates Policy from params
        n_subjects: Number of subjects to simulate
        max_time: Maximum simulation time
        seed: Random seed

    Returns:
        Mean net value across cohort
    """
    policy = policy_factory(params)
    result = run_scenario(scenario, policy, n_subjects, max_time, seed)
    return result.mean_net_value


def optimise_policy(
    scenario: Scenario,
    policy_factory: Callable[[np.ndarray], Policy],
    initial_params: np.ndarray,
    bounds: List[Tuple[float, float]],
    n_subjects: int = 500,
    max_time: float = 200.0,
    method: str = 'L-BFGS-B',
    seed: Optional[int] = None,
    verbose: bool = True
) -> OptimisationResult:
    """
    Optimise policy parameters to maximise expected net value.

    Args:
        scenario: Scenario defining dynamics and costs
        policy_factory: Creates Policy from parameter array
        initial_params: Starting point
        bounds: Parameter bounds [(low, high), ...]
        n_subjects: Subjects per evaluation
        max_time: Simulation max time
        method: scipy.optimize method ('L-BFGS-B', 'Nelder-Mead', etc.)
        seed: Base seed (incremented each evaluation)
        verbose: Print progress

    Returns:
        OptimisationResult with optimal parameters
    """
    history = []
    eval_count = [0]
    start_time = time.time()

    def objective(params):
        eval_seed = None if seed is None else seed + eval_count[0]
        value = evaluate_policy_params(
            params, scenario, policy_factory, n_subjects, max_time, eval_seed
        )
        eval_count[0] += 1
        history.append((params.copy(), value))

        if verbose and eval_count[0] % 5 == 0:
            print(f"  Eval {eval_count[0]:3d}: params={np.round(params, 2)}, value={value:.2f}")

        return -value  # Minimise negative

    if verbose:
        print(f"Optimising with method={method}")
        print(f"  Initial: {initial_params}")
        print(f"  Bounds:  {bounds}")
        print()

    result = optimize.minimize(
        objective,
        initial_params,
        method=method,
        bounds=bounds,
        options={'maxiter': 100, 'disp': False}
    )

    elapsed = time.time() - start_time
    optimal_params = result.x
    optimal_value = -result.fun

    # Evaluate initial params for comparison
    initial_value = evaluate_policy_params(
        initial_params, scenario, policy_factory, n_subjects, max_time, seed
    )

    improvement = optimal_value - initial_value
    improvement_pct = (improvement / abs(initial_value) * 100) if initial_value != 0 else 0

    if verbose:
        print(f"\nDone in {elapsed:.1f}s ({eval_count[0]} evaluations)")
        print(f"  Initial: {initial_params} -> {initial_value:.2f}")
        print(f"  Optimal: {np.round(optimal_params, 2)} -> {optimal_value:.2f}")
        print(f"  Improvement: {improvement:.2f} ({improvement_pct:.1f}%)")

    return OptimisationResult(
        optimal_params=optimal_params,
        optimal_value=optimal_value,
        initial_params=initial_params,
        initial_value=initial_value,
        improvement=improvement,
        improvement_pct=improvement_pct,
        n_evaluations=eval_count[0],
        elapsed_time=elapsed,
        history=history,
        converged=result.success
    )


def grid_search(
    scenario: Scenario,
    policy_factory: Callable[[np.ndarray], Policy],
    param_grids: List[np.ndarray],
    n_subjects: int = 500,
    max_time: float = 200.0,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Grid search over policy parameters.

    Args:
        scenario: Scenario to simulate
        policy_factory: Creates Policy from params
        param_grids: List of 1D arrays, one per parameter dimension
        n_subjects: Subjects per evaluation
        max_time: Simulation max time
        seed: Random seed
        verbose: Print progress

    Returns:
        (best_params, best_value, results_grid)
    """
    grids = np.meshgrid(*param_grids, indexing='ij')
    shape = grids[0].shape
    n_combinations = int(np.prod(shape))
    results = np.zeros(shape)

    if verbose:
        print(f"Grid search: {n_combinations} combinations")

    best_value = -np.inf
    best_params = None
    count = 0

    it = np.nditer(grids[0], flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        params = np.array([grids[i][idx] for i in range(len(param_grids))])

        value = evaluate_policy_params(
            params, scenario, policy_factory, n_subjects, max_time, seed
        )
        results[idx] = value
        count += 1

        if value > best_value:
            best_value = value
            best_params = params.copy()

        if verbose and count % 25 == 0:
            print(f"  {count}/{n_combinations}: best={best_value:.2f} at {best_params}")

        it.iternext()

    if verbose:
        print(f"\nBest: {best_params} -> {best_value:.2f}")

    return best_params, best_value, results


def sensitivity_analysis(
    scenario: Scenario,
    policy_factory: Callable[[np.ndarray], Policy],
    base_params: np.ndarray,
    param_names: List[str],
    variations: np.ndarray = np.linspace(0.5, 1.5, 11),
    n_subjects: int = 500,
    max_time: float = 200.0,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Analyse sensitivity of net value to each parameter.

    Varies each parameter independently while holding others at base values.

    Args:
        scenario: Scenario to simulate
        policy_factory: Creates Policy from params
        base_params: Baseline parameter values
        param_names: Names for each parameter
        variations: Multipliers to apply (e.g., 0.5 to 1.5)
        n_subjects: Subjects per evaluation
        max_time: Simulation max time
        seed: Random seed

    Returns:
        Dict with 'multipliers' and one array per parameter name
    """
    results = {'multipliers': variations}

    for i, name in enumerate(param_names):
        values = []
        for mult in variations:
            params = base_params.copy()
            params[i] = base_params[i] * mult

            value = evaluate_policy_params(
                params, scenario, policy_factory, n_subjects, max_time, seed
            )
            values.append(value)

        results[name] = np.array(values)

    return results
