from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass

from .core import Action, ClickAction, GameState, StepObservation
from .dsl import EnvironmentCase, HiddenMechanicEnvironment
from .hypotheses import MechanicPosterior
from .parser import ObjectTracker, build_state, derive_static_scene, parse_frame
from .planner import plan_with_hypothesis
from .prior import MechanicProposalPrior


@dataclass(frozen=True, slots=True)
class LevelOutcome:
    solved: bool
    action_count: int
    top_confidence: float = 0.0
    hypothesis_count: int = 0
    explored_states: int = 0


class BaseAgent:
    name = "base"

    def solve_case(self, case: EnvironmentCase, *, step_limit: int = 64) -> tuple[LevelOutcome, ...]:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._random = random.Random(seed)

    def solve_case(self, case: EnvironmentCase, *, step_limit: int = 64) -> tuple[LevelOutcome, ...]:
        env = HiddenMechanicEnvironment(case)
        outcomes: list[LevelOutcome] = []
        for level_index in range(len(case.levels)):
            obs = env.reset(level_index)
            while not obs.solved and obs.action_count < step_limit:
                actions: list[Action] = list(obs.available_buttons)
                if obs.allows_click:
                    parsed = parse_frame(obs.frame)
                    actions.extend(ClickAction(*coord) for coord in parsed.clickable_targets())
                if obs.allows_undo:
                    actions.append("undo")
                obs = env.step(self._random.choice(actions))
            outcomes.append(
                LevelOutcome(
                    solved=obs.solved,
                    action_count=obs.action_count,
                    explored_states=obs.action_count,
                )
            )
        return tuple(outcomes)


class FrontierGraphPolicy:
    def __init__(self) -> None:
        self.transitions: dict[tuple[object, ...], dict[Action, tuple[object, ...]]] = defaultdict(dict)

    def observe(self, state: GameState, action: Action, next_state: GameState) -> None:
        self.transitions[state.signature()][action] = next_state.signature()

    def choose(self, state: GameState, actions: tuple[Action, ...]) -> Action:
        signature = state.signature()
        tried = self.transitions.get(signature, {})
        for action in actions:
            if action not in tried:
                return action
        queue = deque([(signature, ())])
        seen = {signature}
        while queue:
            current, path = queue.popleft()
            current_tried = self.transitions.get(current, {})
            if len(current_tried) < len(actions):
                next_action = next(action for action in actions if action not in current_tried)
                return path[0] if path else next_action
            for action, neighbor in current_tried.items():
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, path + (action,)))
        return actions[0]


class FrontierGraphAgent(BaseAgent):
    name = "frontier_graph"

    def solve_case(self, case: EnvironmentCase, *, step_limit: int = 64) -> tuple[LevelOutcome, ...]:
        env = HiddenMechanicEnvironment(case)
        outcomes: list[LevelOutcome] = []
        for level_index in range(len(case.levels)):
            obs = env.reset(level_index)
            parsed = parse_frame(obs.frame)
            static_scene = derive_static_scene(parsed)
            prior_state = build_state(parsed, static_scene)
            policy = FrontierGraphPolicy()
            while not obs.solved and obs.action_count < step_limit:
                actions: tuple[Action, ...] = tuple(obs.available_buttons)
                action = policy.choose(prior_state, actions)
                next_obs = env.step(action)
                next_state = build_state(parse_frame(next_obs.frame), static_scene, prior_state=prior_state)
                policy.observe(prior_state, action, next_state)
                prior_state = next_state
                obs = next_obs
            outcomes.append(
                LevelOutcome(
                    solved=obs.solved,
                    action_count=obs.action_count,
                    explored_states=len(policy.transitions),
                )
            )
        return tuple(outcomes)


class TACTICAgent(BaseAgent):
    name = "tactic"

    def __init__(
        self,
        *,
        transfer: bool = True,
        planner_enabled: bool = True,
        proposal_prior: MechanicProposalPrior | None = None,
        strict_observation: bool = False,
    ) -> None:
        self.transfer = transfer
        self.planner_enabled = planner_enabled
        self.proposal_prior = proposal_prior
        self.strict_observation = strict_observation
        self.memory: dict[str, dict] = {}

    def solve_case(self, case: EnvironmentCase, *, step_limit: int = 64) -> tuple[LevelOutcome, ...]:
        env = HiddenMechanicEnvironment(case, reveal_affordances=not self.strict_observation)
        outcomes: list[LevelOutcome] = []
        prior = self.memory.get(case.env_id)

        for level_index in range(len(case.levels)):
            obs = env.reset(level_index)
            tracker = ObjectTracker()
            parsed = parse_frame(obs.frame)
            tracker.update(parsed)
            static_scene = derive_static_scene(parsed)
            state = build_state(parsed, static_scene)
            posterior = MechanicPosterior(
                obs.available_buttons,
                state,
                allows_click=True if self.strict_observation else obs.allows_click,
                allows_undo=True if self.strict_observation else obs.allows_undo,
                prior=prior if self.transfer else None,
                proposal_prior=self.proposal_prior,
            )
            frontier = FrontierGraphPolicy()
            planned_actions: tuple[Action, ...] = ()

            while not obs.solved and obs.action_count < step_limit:
                parsed = parse_frame(obs.frame)
                tracker.update(parsed)
                state = build_state(parsed, static_scene, prior_state=state)
                actions = self._candidate_actions(obs, parsed, state)

                top_hypothesis, confidence = posterior.top
                if self.planner_enabled and confidence >= 0.55 and not planned_actions:
                    plan = plan_with_hypothesis(state, top_hypothesis, max_depth=step_limit - obs.action_count)
                    if plan and plan.actions:
                        planned_actions = plan.actions

                if planned_actions:
                    action = planned_actions[0]
                    planned_actions = planned_actions[1:]
                else:
                    action = self._choose_experiment(
                        state,
                        actions,
                        posterior,
                        frontier,
                        remaining_level_weight=float(len(case.levels) - level_index),
                    )

                next_obs = env.step(action)
                next_state = build_state(parse_frame(next_obs.frame), static_scene, prior_state=state)
                posterior.update(state, action, next_state)
                frontier.observe(state, action, next_state)
                state = next_state
                obs = next_obs
                planned_actions = ()

            prior = posterior.transfer_prior(state)
            top_hypothesis, confidence = posterior.top
            outcomes.append(
                LevelOutcome(
                    solved=obs.solved,
                    action_count=obs.action_count,
                    top_confidence=confidence,
                    hypothesis_count=posterior.hypothesis_count,
                    explored_states=len(frontier.transitions),
                )
            )

        if self.transfer:
            self.memory[case.env_id] = prior
        return tuple(outcomes)

    def _candidate_actions(self, obs: StepObservation, parsed_frame, state: GameState) -> tuple[Action, ...]:
        actions: list[Action] = list(obs.available_buttons)
        if self.strict_observation:
            click_targets = parsed_frame.non_wall_anchors()
            actions.extend(ClickAction(row, col) for row, col in click_targets)
            actions.append("undo")
        else:
            if obs.allows_click:
                click_targets = sorted(state.switches or set(state.portals) or state.goals)
                actions.extend(ClickAction(row, col) for row, col in click_targets)
            if obs.allows_undo:
                actions.append("undo")
        return tuple(dict.fromkeys(actions))

    def _choose_experiment(
        self,
        state: GameState,
        actions: tuple[Action, ...],
        posterior: MechanicPosterior,
        frontier: FrontierGraphPolicy,
        *,
        remaining_level_weight: float,
    ) -> Action:
        tried = frontier.transitions.get(state.signature(), {})
        best_score = float("-inf")
        best_action = actions[0]
        for action in actions:
            score = posterior.score_action(
                state,
                action,
                remaining_level_weight=remaining_level_weight,
                novel=action not in tried,
            )
            if score > best_score:
                best_score = score
                best_action = action
        if best_score < 0.1:
            return frontier.choose(state, actions)
        return best_action


class TACTICNoTransferAgent(TACTICAgent):
    name = "tactic_no_transfer"

    def __init__(self) -> None:
        super().__init__(transfer=False, planner_enabled=True)


class TACTICNoPlannerAgent(TACTICAgent):
    name = "tactic_no_planner"

    def __init__(self) -> None:
        super().__init__(transfer=True, planner_enabled=False)


class TACTICStrictAgent(TACTICAgent):
    name = "tactic_strict"

    def __init__(self, *, transfer: bool = True, proposal_prior: MechanicProposalPrior | None = None) -> None:
        super().__init__(
            transfer=transfer,
            planner_enabled=True,
            proposal_prior=proposal_prior,
            strict_observation=True,
        )
