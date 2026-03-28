from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .core import Action, ClickAction
from .dsl import EnvironmentCase, is_solved, simulate_action


@dataclass(frozen=True, slots=True)
class OracleResult:
    solved: bool
    steps: int
    expansions: int


def true_actions(case: EnvironmentCase, level_index: int) -> tuple[Action, ...]:
    state = case.levels[level_index]
    config = case.config_for_level(level_index)
    actions: list[Action] = list(config.available_buttons)
    if config.click_mode == "switch":
        actions.extend(ClickAction(row, col) for row, col in sorted(state.switches))
    elif config.click_mode == "teleport":
        actions.extend(ClickAction(row, col) for row, col in state.portals)
    return tuple(actions)


def solve_with_oracle(
    case: EnvironmentCase,
    level_index: int,
    *,
    max_depth: int = 80,
    max_expansions: int = 50000,
) -> OracleResult:
    start = case.levels[level_index]
    config = case.config_for_level(level_index)
    if is_solved(start, config.family):
        return OracleResult(solved=True, steps=0, expansions=0)

    actions = true_actions(case, level_index)
    queue = deque([(start, 0)])
    best_cost = {start.signature(): 0}
    expansions = 0

    while queue and expansions < max_expansions:
        state, depth = queue.popleft()
        expansions += 1
        if depth >= max_depth:
            continue
        for action in actions:
            next_state = simulate_action(state, action, config)
            next_key = next_state.signature()
            next_depth = depth + 1
            if next_depth >= best_cost.get(next_key, 10**9):
                continue
            if is_solved(next_state, config.family):
                return OracleResult(solved=True, steps=next_depth, expansions=expansions)
            best_cost[next_key] = next_depth
            queue.append((next_state, next_depth))

    return OracleResult(solved=False, steps=max_depth, expansions=expansions)
