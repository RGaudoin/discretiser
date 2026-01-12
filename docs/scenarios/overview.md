# Synthetic Scenarios Overview

Design notes for synthetic data generators with known ground-truth dynamics.

## Purpose

Build synthetic scenarios where:
1. Ground truth dynamics are fully specified and known
2. Data can be generated for training generative models
3. Learning algorithms can optimise policies on both ground truth and learned models
4. Policies can be validated by comparing learned-model performance to ground-truth performance

**Important:** The primary goal is not the optimisation itself, but to evaluate whether models trained on synthetic data capture the underlying dynamics sufficiently well. Policy optimisation (RL or otherwise) is a tool for this evaluation - if a model doesn't capture the dynamics, policies learned on it will perform poorly against ground truth. The choice of optimisation method (RL, parameter search, etc.) is secondary to validating the learned model's fidelity.

## Validation Loop

```
Ground Truth Simulator
        │
        ▼
   Generate Data ──────────────────────────┐
        │                                  │
        ▼                                  │
   Train Generative Model                  │
        │                                  │
        ▼                                  │
   Policy Learning on Learned Model ─► Policy A
        │                                  │
        │                     Compare      │
        │                       │          │
        ▼                       ▼          │
   Policy Learning on Ground Truth ─► Policy B
        │                       │          │
        ▼                       ▼          │
   Evaluate Both Policies on Ground Truth ◄┘
```

**Key metrics:**
- Does learned model recover true dynamics? (distribution matching)
- Does Policy A (from learned) perform similarly to Policy B (from truth)?
- How much does the learned model's imperfections degrade policy quality?

## Scenario Organisation

Scenarios are organised by complexity, each building on the previous:

| Scenario | Description | Key Features |
|----------|-------------|--------------|
| [basic_bathtub](basic_bathtub.md) | Single service action, bathtub failure | Effective age model, subject heterogeneity |
| [widget_maintenance_full](widget_maintenance_full.md) | Two maintenance types + monitoring | Degradation events, hierarchical policies |

## Common Elements

### Policy Hierarchy

Three levels of policy sophistication (applicable to all scenarios):

**1. Global parameter optimisation:**
- "Maintain every X days" → search for best X
- Same rule for all subjects
- Produces homogeneous training data (problem for learning)

**2. Linear feature-based baseline:**
- `interval = a + b·features` - simple mapping from features to timing
- Different subjects get different intervals
- Adds data diversity tied to features (needed for learning)

**3. Full policy learning (RL or other):**
- π(state, features) → action
- Adapts to accumulated history, not just initial features
- Captures non-linear feature interactions, phase-dependent behaviour

### Embedding Validation

A trained generative model may produce embeddings at different levels:

**Subject-level embeddings:**
- Represent inherent characteristics (from subject features)
- Should correlate with ground-truth durability factors
- Validate via: clustering, regression to predict failure rates

**Journey-level embeddings:**
- Represent accumulated state (maintenance history, etc.)
- Should encode "risk state" at any point in time
- Validate via: classification ("needs maintenance soon"), time-to-failure prediction

### Data Diversity for Learning

A fixed-interval baseline produces homogeneous data - learner can't discover what happens at other intervals.

**Solution**: Use linear feature-based baseline for training data generation:
- `interval = base + weight·durability_feature`
- Creates variation tied to features
- Learner sees different intervals for different subjects
- Can discover feature→outcome relationships

## Implementation Structure

```
src/scenarios/
├── __init__.py
├── base.py              # Common interfaces, cost functions
├── basic_bathtub.py     # Simplest scenario
└── widget_maintenance.py # Full scenario with options
```

Each scenario module defines:
- Events and their survival models
- Baseline policy (for data generation)
- Cost structure
- Validation criteria

Notebooks import scenarios for exploration and visualisation.

## Scenario vs RL Layer Separation

Scenarios and RL/policy learning are **separate concerns**:

| Scenarios Define | RL Layer Defines |
|-----------------|------------------|
| Ground truth dynamics (survival models) | Action space representation |
| Subject features and generation | Decision timing (when to act) |
| Event types and transitions | Policy architecture (NN, table, etc.) |
| Cost/reward structure | Learning algorithm (PPO, DQN, etc.) |
| Baseline policy (for data generation) | Exploration strategy |

**Why separate?**
- Same scenario can be used with different RL approaches
- Same RL code can be reused across scenarios
- Scenarios focus on "what happens", RL focuses on "what to do"
- Cleaner testing: scenario dynamics tested independently

**Interface:**
- Scenarios provide `state`, `events`, and `costs`
- RL wraps scenario in an environment (Gym-style or custom)
- Actions are applied via `state.add_pending_event()` or similar

The "Open Issues" in individual scenario docs (action space, observation space, etc.) belong to the **RL layer**, not the scenario definition.

## Simulation Ending Terminology

Following RL conventions (Gymnasium), simulations can end in different ways:

| Flag | Meaning | Example |
|------|---------|---------|
| `terminated` | Natural end via terminal event | failure, healed, cured |
| `truncated` | Artificial end due to time limit | reached `max_time` |
| `censored` | Subtype of terminated - outcome unknown | lost to follow-up |

**Key points:**
- `terminated` and `truncated` are mutually exclusive
- `censored` is always a subset of `terminated` (it's a type of terminal event)
- Terminal events are scenario-specific (failure, healed, censored, etc.)
- Cost calculations should check which terminal event occurred
