from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import arc_agi
from arcengine import GameAction

from .arc_agi3_policy import (
    ArcAgi3TACTICPublicAgent,
    ActionCandidate,
    EnvironmentMemory,
    LevelMemory,
    component_by_feature,
    frame_signature,
    infer_component_motion,
    per_level_budget,
    primary_frame,
    refinement_clicks,
    reward_for_transition,
    transition_mask,
)

HARNESS_SRC = Path(__file__).resolve().parents[1] / "arc-agi-3-benchmarking" / "src"
if str(HARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(HARNESS_SRC))

from arcagi3.agent import MultimodalAgent  # noqa: E402
from arcagi3.schemas import GameStep  # noqa: E402
from arcagi3.utils.context import SessionContext  # noqa: E402


_BASELINE_ACTIONS: dict[str, tuple[int, ...]] | None = None


def _baseline_actions_lookup() -> dict[str, tuple[int, ...]]:
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.ONLINE)
    lookup: dict[str, tuple[int, ...]] = {}
    for info in arcade.get_environments():
        baseline_actions = tuple(int(value) for value in getattr(info, "baseline_actions", ()) or ())
        lookup[info.game_id] = baseline_actions
        lookup[info.game_id.split("-", 1)[0]] = baseline_actions
    return lookup


def _get_baseline_actions(game_id: str) -> tuple[int, ...]:
    global _BASELINE_ACTIONS
    if _BASELINE_ACTIONS is None:
        _BASELINE_ACTIONS = _baseline_actions_lookup()
    return _BASELINE_ACTIONS.get(game_id, ())


def _available_engine_actions(values: Iterable[str]) -> tuple[GameAction, ...]:
    actions: list[GameAction] = []
    for value in values:
        action_name = f"ACTION{int(str(value))}"
        try:
            actions.append(GameAction[action_name])
        except (KeyError, ValueError):
            continue
    return tuple(actions)


@dataclass(slots=True)
class PendingTransition:
    candidate: ActionCandidate
    state_signature: bytes
    level_index: int


class TACTICHarnessAgent(MultimodalAgent):
    """
    Harness-facing wrapper for the current public TACTIC controller.

    The official ARC harness expects one `step(context)` at a time. The existing
    TACTIC controller was written around a toolkit environment loop, so this
    adapter carries forward the same transition bookkeeping one step later.
    """

    def __init__(
        self,
        *args,
        use_vision: bool = False,
        show_images: bool = False,
        memory_word_limit: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_vision = use_vision
        self.show_images = show_images
        self.memory_word_limit = memory_word_limit
        self.policy = ArcAgi3TACTICPublicAgent()
        self.environment_memory = EnvironmentMemory()
        self.level_memory: LevelMemory | None = None
        self.pending: PendingTransition | None = None
        self.last_guid: str | None = None
        self.current_level_budget = 64
        self.baseline_actions = ()

    def step(self, context: SessionContext) -> GameStep:
        frame = primary_frame(context.frames.frame_grids)
        guid = context.game.guid or "noguid"
        level_index = int(context.game.current_score)
        if not self.baseline_actions:
            self.baseline_actions = _get_baseline_actions(context.game.game_id)
            self.current_level_budget = per_level_budget(self.baseline_actions, level_index)

        if self.last_guid != guid or self.level_memory is None:
            self._reset_level(frame, context, level_index)
            self.last_guid = guid
        else:
            self._ingest_pending_transition(context, frame, level_index)

        signature = frame_signature(frame)
        self.level_memory.mark_seen(signature)
        available_actions = _available_engine_actions(context.game.available_actions)
        if not available_actions:
            available_actions = (
                GameAction.ACTION1,
                GameAction.ACTION2,
                GameAction.ACTION3,
                GameAction.ACTION4,
                GameAction.ACTION5,
                GameAction.ACTION6,
                GameAction.ACTION7,
            )

        if self.level_memory.step_index >= self.current_level_budget:
            self.level_memory.probe_queue.clear()

        candidate = self.policy._choose_action(
            frame,
            signature,
            available_actions,
            self.environment_memory,
            self.level_memory,
        )
        self.pending = PendingTransition(
            candidate=candidate,
            state_signature=signature,
            level_index=level_index,
        )

        action_payload: dict[str, object] = {"action": candidate.action.name}
        if candidate.data is not None:
            action_payload["data"] = dict(candidate.data)
        return GameStep(
            action=action_payload,
            reasoning={
                "agent": "tactic-public",
                "label": candidate.label,
                "level_index": level_index,
                "budget": self.current_level_budget,
                "step_index": self.level_memory.step_index,
            },
        )

    def _reset_level(self, frame, context: SessionContext, level_index: int) -> None:
        self.level_memory = LevelMemory()
        self.pending = None
        self.current_level_budget = per_level_budget(self.baseline_actions, level_index)
        self.level_memory.mark_seen(frame_signature(frame))
        self.policy._prime_probe_queue(
            _ContextObs(frame=(frame,)),
            list(_available_engine_actions(context.game.available_actions)),
            self.environment_memory,
            self.level_memory,
        )

    def _ingest_pending_transition(
        self,
        context: SessionContext,
        frame,
        level_index: int,
    ) -> None:
        pending = self.pending
        if pending is None or not context.frames.previous_grids:
            return
        previous_frame = primary_frame(context.frames.previous_grids)
        current_signature = frame_signature(frame)
        level_gain = max(0, level_index - pending.level_index)
        raw_delta_mask = previous_frame != frame
        self.environment_memory.record_transition_mask(raw_delta_mask)
        meaningful_mask = transition_mask(
            previous_frame,
            frame,
            noise_mask=self.environment_memory.noise_mask(),
        )
        changed = bool(meaningful_mask.any())
        reward = reward_for_transition(
            meaningful_mask,
            level_gain=level_gain,
            is_win=context.game.current_state == "WIN",
        )

        if pending.candidate.action not in {GameAction.ACTION6, GameAction.ACTION7} and changed:
            inferred_motion = infer_component_motion(previous_frame, frame)
            if inferred_motion is not None:
                feature, delta = inferred_motion
                self.environment_memory.record_motion(
                    pending.candidate.action.name,
                    delta,
                    feature,
                )

        self.environment_memory.stat_for(pending.candidate.key).update(
            reward,
            changed=changed,
            level_gain=level_gain > 0,
        )
        self.environment_memory.record_effect(pending.candidate.key, meaningful_mask)

        if pending.candidate.action == GameAction.ACTION6 and pending.candidate.key[1] is not None:
            self.environment_memory.click_stat_for(pending.candidate.key[1]).update(
                reward,
                changed=changed,
                level_gain=level_gain > 0,
            )

        if pending.candidate.action == GameAction.ACTION6 and changed and pending.candidate.data is not None:
            clicked_coord = (
                int(pending.candidate.data["y"]),
                int(pending.candidate.data["x"]),
            )
            for coord, feature in refinement_clicks(clicked_coord, meaningful_mask):
                refined = ActionCandidate(
                    action=GameAction.ACTION6,
                    data={"x": coord[1], "y": coord[0]},
                    key=(GameAction.ACTION6.name, feature),
                    label=f"{GameAction.ACTION6.name}@{coord[0]},{coord[1]}",
                )
                if not self.level_memory.has_tried(pending.state_signature, refined.key):
                    self.level_memory.probe_queue.appendleft(refined)

        self.level_memory.mark_tried(pending.state_signature, pending.candidate.key)
        self.level_memory.probed_keys.add(pending.candidate.key)
        self.level_memory.observe_transition(
            pending.state_signature,
            pending.candidate.key,
            current_signature,
        )
        self.level_memory.mark_seen(current_signature)
        self.level_memory.step_index += 1
        if changed:
            self.level_memory.last_changed_step = self.level_memory.step_index

        actor_component = component_by_feature(frame, self.environment_memory.actor_features)
        if actor_component is not None:
            self.level_memory.actor_positions.add(actor_component.anchor)

        if (
            pending.candidate.action not in {GameAction.ACTION6, GameAction.ACTION7}
            and changed
            and level_gain == 0
            and "7" in {str(value) for value in context.game.available_actions}
            and not self.level_memory.probe_queue
            and self.level_memory.step_index <= 12
        ):
            self.level_memory.probe_queue.appendleft(
                ActionCandidate(
                    action=GameAction.ACTION7,
                    data=None,
                    key=(GameAction.ACTION7.name, None),
                    label=GameAction.ACTION7.name,
                )
            )

        self.pending = None
        if level_index != pending.level_index:
            self._reset_level(frame, context, level_index)


@dataclass(frozen=True, slots=True)
class _ContextObs:
    frame: tuple


definition = {
    "name": "tactic-public",
    "description": "Current non-LLM TACTIC ARC public controller wired into the official harness",
    "agent_class": TACTICHarnessAgent,
}
