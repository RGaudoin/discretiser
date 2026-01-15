# RL Design for Policy Optimisation

Design decisions for reinforcement learning approach to finding optimal service policies.

## Goals

1. Learn dynamic policies that outperform static LinearIntervalPolicy
2. Work with both ground-truth simulator and learnt models
3. Exploit bathtub hazard structure (drift through infant mortality, stay in trough)

## Design Decisions

### Decision Timing: Event-Driven

The RL agent makes decisions after each event (service or failure attempt).

**Rationale**: Fits naturally with existing policy architecture where `Policy.get_action(state, subject)` is called after each event.

### State Space

Observable features only (no ground-truth internal state like effective_age):

| Feature | Description |
|---------|-------------|
| time_since_last_service | Time elapsed since most recent service (or episode start) |
| service_count | Total services performed so far |
| avg_service_interval | Mean time between services (service_count > 0) |
| durability | Subject feature (static) |
| delta_t | Scenario parameter: age reduction per service |
| service_cost | Scenario parameter: cost per service |
| revenue_per_time | Scenario parameter: revenue rate while alive |
| failure_cost | Scenario parameter: cost on failure |

**Rationale**:
- Using observables (not effective_age) allows transfer to learnt models
- Including scenario parameters (delta_t, costs) enables generalisation across scenarios
- Neural network can ignore irrelevant features

### Action Space: Discrete Geometric

Actions represent delay until next scheduled service:

```
[0, 10, 20, 40, 80, ∞]
```

| Action | Meaning |
|--------|---------|
| 0 | Immediate service (next timestep) |
| 10, 20, 40, 80 | Schedule service after this delay |
| ∞ | No more service (let it run to failure/truncation) |

**Rationale**:
- Geometric spacing covers wide dynamic range with few actions
- 0 allows rapid consecutive services (e.g., after reaching trough)
- ∞ allows "give up" when economics don't justify service
- Discrete actions suit Q-learning / DQN well

### Reward Structure: Step-Wise

Rewards given at each transition:

```
reward = revenue_per_time × time_elapsed
         - service_cost × (1 if serviced else 0)
         - failure_cost × (1 if failed else 0)
```

**Rationale**:
- Dense signal (not just terminal) enables faster learning
- Components sum to net_value, so optimising this optimises true objective
- No reward shaping that requires ground-truth knowledge

### Episode Termination

Two terminal conditions:

1. **Failure**: Episode ends, final step includes -failure_cost
2. **Truncation** (max_time reached): Episode ends, no additional reward

**Rationale**:
- No bootstrapping at truncation (simpler, more stable)
- Consistent with LinearIntervalPolicy evaluation (fixed horizon)
- Treats problem as finite-horizon MDP

## Algorithm: DQN via Stable-Baselines3

**Choice**: DQN from Stable-Baselines3 (SB3)

**Rationale**:
- Continuous 8D state space rules out tabular Q-learning
- Discrete action space (6 actions) suits DQN well
- SB3 is battle-tested, no need to reinvent
- This repo is not about RL - use proven tools

**Architecture**:
```
┌─────────────────────────────────────────┐
│  Gymnasium Environment                  │
│  ┌─────────────────────────────────┐    │
│  │  Simulator / Learnt Model       │    │
│  │  (BasicBathtubScenario or       │    │
│  │   TrainedModelSurvival)         │    │
│  └─────────────────────────────────┘    │
│  • reset() → state                      │
│  • step(action) → state, reward, done   │
└─────────────────────────────────────────┘
           ↕
┌─────────────────────────────────────────┐
│  SB3 DQN (used directly, no wrapper)    │
└─────────────────────────────────────────┘
```

**No SB3 wrapper**: Use SB3 API directly. Abstraction is at environment level (Gymnasium interface), allowing swap between ground-truth simulator and learnt models.

## Open: Claude Agent Usage

Options for implementation:
- Use analytics-expert agent for algorithm verification
- Use general-purpose agent for full implementation
- Implement directly with agent for review
