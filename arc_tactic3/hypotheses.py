from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, permutations

from .core import Action, CARDINALS, Family, GameState, MechanicConfig, manhattan
from .dsl import is_solved, render_state, simulate_action
from .prior import MechanicProposalPrior, PriorPrediction


@dataclass(frozen=True, slots=True)
class MechanicHypothesis:
    family: Family
    movement_map: tuple[tuple[str, tuple[int, int] | None], ...]
    click_mode: str | None
    allows_undo: bool

    def config(self, available_buttons: tuple[str, ...]) -> MechanicConfig:
        return MechanicConfig(
            family=self.family,
            available_buttons=available_buttons,
            movement_map=self.movement_map,
            click_mode=self.click_mode,
            allows_undo=self.allows_undo,
        )


def entropy(weights: dict[MechanicHypothesis, float]) -> float:
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    score = 0.0
    for weight in weights.values():
        probability = weight / total
        if probability > 0:
            score -= probability * math.log2(probability)
    return score


def compatible_families(state: GameState, allows_click: bool) -> tuple[tuple[Family, str | None], ...]:
    families: list[tuple[Family, str | None]] = []
    if state.boxes or state.targets:
        families.append(("push_box", None))
    if state.switches:
        families.append(("switch_goal", None))
        if allows_click:
            families.append(("switch_goal", "switch"))
    if state.keys or state.has_key or (state.doors and not state.switches):
        families.append(("key_goal", None))
    if state.portals:
        families.append(("portal_goal", None))
        if allows_click:
            families.append(("portal_goal", "teleport"))
    if not families:
        families.append(("reach_goal", None))
    return tuple(dict.fromkeys(families))


def enumerate_movement_maps(
    available_buttons: tuple[str, ...],
) -> tuple[tuple[tuple[str, tuple[int, int] | None], ...], ...]:
    if len(available_buttons) < 4:
        raise ValueError("at least four buttons are required")
    maps: list[tuple[tuple[str, tuple[int, int] | None], ...]] = []
    for active in combinations(available_buttons, 4):
        inactive = [button for button in available_buttons if button not in active]
        for assignment in permutations(CARDINALS, 4):
            pairs = [(button, delta) for button, delta in zip(active, assignment, strict=True)]
            pairs.extend((button, None) for button in inactive)
            maps.append(tuple(sorted(pairs)))
    return tuple(dict.fromkeys(maps))


class MechanicPosterior:
    def __init__(
        self,
        available_buttons: tuple[str, ...],
        state: GameState,
        *,
        allows_click: bool,
        allows_undo: bool,
        prior: dict[MechanicHypothesis, float] | None = None,
        proposal_prior: MechanicProposalPrior | None = None,
    ) -> None:
        self.available_buttons = available_buttons
        self.allows_click = allows_click
        self.allows_undo = allows_undo
        if prior:
            self.weights = dict(prior)
        else:
            families = compatible_families(state, allows_click)
            movement_maps = enumerate_movement_maps(available_buttons)
            base_weight = 1.0 / max(len(families) * len(movement_maps), 1)
            self.weights = {
                MechanicHypothesis(
                    family=family,
                    movement_map=movement_map,
                    click_mode=click_mode,
                    allows_undo=allows_undo,
                ): base_weight
                for family, click_mode in families
                for movement_map in movement_maps
            }
        if proposal_prior is not None and not prior:
            prediction = proposal_prior.predict(
                state,
                available_buttons=available_buttons,
                allows_click=allows_click,
                allows_undo=allows_undo,
            )
            self._apply_prediction(prediction)
        self.normalize()

    def normalize(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            uniform = 1.0 / max(len(self.weights), 1)
            for hypothesis in list(self.weights):
                self.weights[hypothesis] = uniform
            return
        for hypothesis in list(self.weights):
            self.weights[hypothesis] /= total

    @property
    def uncertainty(self) -> float:
        return entropy(self.weights)

    @property
    def top(self) -> tuple[MechanicHypothesis, float]:
        hypothesis = max(self.weights, key=self.weights.get)
        return hypothesis, self.weights[hypothesis]

    @property
    def hypothesis_count(self) -> int:
        return len(self.weights)

    def update(self, state: GameState, action: Action, observed_state: GameState) -> None:
        observed_frame = render_state(observed_state)
        next_weights: dict[MechanicHypothesis, float] = {}
        for hypothesis, weight in self.weights.items():
            predicted = simulate_action(state, action, hypothesis.config(self.available_buttons))
            if render_state(predicted) == observed_frame:
                next_weights[hypothesis] = weight
        if next_weights:
            self.weights = next_weights
            self.normalize()

    def transfer_prior(self, state: GameState, *, mode: str = "full") -> dict[MechanicHypothesis, float]:
        family_set = set(compatible_families(state, self.allows_click))
        filtered = {
            hypothesis: weight
            for hypothesis, weight in self.weights.items()
            if (hypothesis.family, hypothesis.click_mode) in family_set
        }
        total = sum(filtered.values())
        if total <= 0:
            return {}
        if mode == "full":
            return {hypothesis: weight / total for hypothesis, weight in filtered.items()}

        grouped: dict[tuple, float] = defaultdict(float)
        counts: dict[tuple, int] = defaultdict(int)
        if mode == "family_click":
            for hypothesis, weight in filtered.items():
                key = (hypothesis.family, hypothesis.click_mode, hypothesis.allows_undo)
                grouped[key] += weight
                counts[key] += 1
            redistributed = {
                hypothesis: grouped[(hypothesis.family, hypothesis.click_mode, hypothesis.allows_undo)]
                / counts[(hypothesis.family, hypothesis.click_mode, hypothesis.allows_undo)]
                for hypothesis in filtered
            }
        elif mode == "family_only":
            for hypothesis, weight in filtered.items():
                key = (hypothesis.family, hypothesis.allows_undo)
                grouped[key] += weight
                counts[key] += 1
            redistributed = {
                hypothesis: grouped[(hypothesis.family, hypothesis.allows_undo)]
                / counts[(hypothesis.family, hypothesis.allows_undo)]
                for hypothesis in filtered
            }
        else:
            raise ValueError(f"unsupported transfer mode: {mode}")

        normalized_total = sum(redistributed.values())
        return {
            hypothesis: weight / normalized_total
            for hypothesis, weight in redistributed.items()
        }

    def transfer_summary(self, state: GameState, *, mode: str = "family_click") -> PriorPrediction:
        family_set = set(compatible_families(state, self.allows_click))
        filtered = {
            hypothesis: weight
            for hypothesis, weight in self.weights.items()
            if (hypothesis.family, hypothesis.click_mode) in family_set
        }
        total = sum(filtered.values())
        if total <= 0:
            return PriorPrediction()

        family_scores: dict[Family, float] = defaultdict(float)
        click_mode_scores: dict[str | None, float] = defaultdict(float)
        for hypothesis, weight in filtered.items():
            family_scores[hypothesis.family] += weight
            if mode == "family_click":
                click_mode_scores[hypothesis.click_mode] += weight
            elif mode != "family_only":
                raise ValueError(f"unsupported summary transfer mode: {mode}")

        family_total = sum(family_scores.values())
        click_total = sum(click_mode_scores.values())
        normalized_family = {
            family: weight / family_total
            for family, weight in family_scores.items()
        }
        normalized_click = (
            {
                click_mode: weight / click_total
                for click_mode, weight in click_mode_scores.items()
            }
            if click_total > 0
            else {}
        )
        return PriorPrediction(
            family_scores=normalized_family,
            click_mode_scores=normalized_click,
            button_direction_scores={},
        )

    def _apply_prediction(self, prediction: PriorPrediction) -> None:
        self.weights = {
            hypothesis: weight * prediction_multiplier(prediction, hypothesis)
            for hypothesis, weight in self.weights.items()
        }

    def apply_prediction(self, prediction: PriorPrediction) -> None:
        self._apply_prediction(prediction)
        self.normalize()

    def score_action(
        self,
        state: GameState,
        action: Action,
        *,
        remaining_level_weight: float,
        novel: bool,
    ) -> float:
        partitions: dict[tuple[tuple[int, ...], ...], dict[MechanicHypothesis, float]] = defaultdict(dict)
        progress = 0.0
        for hypothesis, weight in self.weights.items():
            predicted = simulate_action(state, action, hypothesis.config(self.available_buttons))
            partitions[render_state(predicted)][hypothesis] = weight
            progress += weight * progress_score(predicted, hypothesis.family)
        total = sum(sum(group.values()) for group in partitions.values())
        expected_entropy = 0.0
        for group in partitions.values():
            probability = sum(group.values()) / total
            expected_entropy += probability * entropy(group)
        novelty_bonus = 0.25 if novel else 0.0
        transfer_bonus = 0.20 * remaining_level_weight * self.uncertainty
        info_gain = self.uncertainty - expected_entropy
        return 1.8 * info_gain + 0.8 * progress + transfer_bonus + novelty_bonus


def progress_score(state: GameState, family: Family) -> float:
    if family == "push_box":
        if state.boxes & state.targets:
            return 5.0
        if not state.boxes or not state.targets:
            return 0.0
        best = min(manhattan(box, target) for box in state.boxes for target in state.targets)
        player_to_box = min(manhattan(state.player, box) for box in state.boxes)
        return 1.0 / (1 + best + player_to_box)

    if not state.goals:
        goal_dist = 0
    else:
        goal_dist = min(manhattan(state.player, goal) for goal in state.goals)
    if family == "key_goal" and not state.has_key and state.keys:
        goal_dist += min(manhattan(state.player, key) for key in state.keys)
    if family == "switch_goal" and not state.switch_active and state.switches:
        goal_dist += min(manhattan(state.player, switch) for switch in state.switches)
    if family == "portal_goal" and state.portals:
        goal_dist = min(goal_dist, min(manhattan(state.player, portal) for portal in state.portals) + 1)
    return (4.0 if is_solved(state, family) else 0.0) + 1.0 / (1 + goal_dist)


def prediction_multiplier(prediction: PriorPrediction, hypothesis: MechanicHypothesis) -> float:
    multiplier = 1.0
    if prediction.family_scores:
        multiplier *= 0.1 + prediction.family_scores.get(hypothesis.family, 0.1)
    if prediction.click_mode_scores:
        multiplier *= 0.1 + prediction.click_mode_scores.get(hypothesis.click_mode, 0.1)
    if prediction.button_direction_scores:
        for button, direction in hypothesis.movement_map:
            per_button = prediction.button_direction_scores.get(button)
            if not per_button:
                continue
            multiplier *= 0.2 + per_button.get(direction, 0.0)
    return multiplier
