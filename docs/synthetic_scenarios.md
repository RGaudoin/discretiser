# Synthetic Scenarios

This documentation has been reorganised into separate files per scenario.

## Documentation

- [Overview](scenarios/overview.md) - Purpose, validation loop, common elements
- [Basic Bathtub](scenarios/basic_bathtub.md) - Simplest scenario: single service action, effective age model
- [Widget Maintenance Full](scenarios/widget_maintenance_full.md) - Two maintenance types, degradation, monitoring

## Quick Reference

| Scenario | Complexity | Key Features |
|----------|------------|--------------|
| basic_bathtub | Low | Single action, bathtub failure, effective age |
| widget_maintenance_full | High | Two actions, degradation, optional monitoring |

## Implementation Status

- [ ] Implement `time_since()` and `events_since()` methods in State
- [ ] Create failure model (StateDependentWeibull or subject-dependent bathtub)
- [ ] Build baseline heuristic policy
- [ ] Generate sample data and visualise
- [ ] Define RL interface (action space, observation space, reward)

## Code Structure

```
src/scenarios/
├── __init__.py
├── base.py              # Common interfaces, cost functions
├── basic_bathtub.py     # Simplest scenario
└── widget_maintenance.py # Full scenario with options
```
