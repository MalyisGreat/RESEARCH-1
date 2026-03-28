from dataclasses import dataclass

import numpy as np
from arcengine import GameAction, GameState as ArcGameState

from arc_tactic3.arc_agi3_policy import (
    ArcAgi3TACTICPublicAgent,
    ActionCandidate,
    ActionStats,
    action_mask,
    coarse_cell_for_coord,
    coarse_cells_for_bounds,
    coarse_path_distance,
    EnvironmentMemory,
    LevelMemory,
    candidate_clicks,
    coarse_cells_from_mask,
    coarse_blocked_cells_for_components,
    delta_signature,
    MechanicHint,
    extract_components,
    FrameComponent,
    frame_signature,
    infer_component_motion,
    infer_actor_features,
    infer_interaction_targets,
    is_small_border_only_change,
    key_stability_score,
    PolicyTuning,
    per_level_budget,
    primary_frame,
    refinement_clicks,
    sweep_clicks,
    transition_mask,
)


def test_extract_components_ignores_background_and_finds_anchors() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[1:3, 1:3] = 5
    frame[5, 6] = 7
    components = extract_components(frame)
    assert len(components) == 2
    assert components[0].color == 5
    assert components[0].anchor in {(1, 2), (2, 1)}


def test_primary_frame_uses_latest_frame_in_stack() -> None:
    earlier = np.zeros((4, 4), dtype=np.int8)
    later = np.ones((4, 4), dtype=np.int8) * 7
    frame = primary_frame((earlier, later))
    assert int(frame[0, 0]) == 7


def test_candidate_clicks_prioritizes_known_feature() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[1:3, 1:3] = 5
    frame[4:7, 4:7] = 8
    memory = EnvironmentMemory()
    target_feature = extract_components(frame)[0].feature
    memory.click_stat_for(target_feature).update(5.0, changed=True, level_gain=True)
    clicks = candidate_clicks(frame, memory, limit=3)
    assert clicks[0][1] == target_feature


def test_action6_candidates_emit_coordinate_payloads() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[2:4, 2:4] = 3
    agent = ArcAgi3TACTICPublicAgent()
    candidates = agent._candidates(frame, (GameAction.ACTION6,), EnvironmentMemory())
    assert candidates
    assert all(candidate.data is not None for candidate in candidates)
    assert all("x" in candidate.data and "y" in candidate.data for candidate in candidates)


def test_infer_component_motion_detects_simple_translation() -> None:
    previous = np.zeros((8, 8), dtype=np.int8)
    current = np.zeros((8, 8), dtype=np.int8)
    previous[2:4, 2:4] = 6
    current[2:4, 3:5] = 6
    inferred = infer_component_motion(previous, current)
    assert inferred is not None
    _, delta = inferred
    assert delta == (0, 1)


def test_infer_component_motion_prefers_near_match_over_far_same_color() -> None:
    previous = np.zeros((16, 16), dtype=np.int8)
    current = np.zeros((16, 16), dtype=np.int8)
    previous[2:4, 2:4] = 6
    previous[10:12, 10:12] = 6
    current[2:4, 3:5] = 6
    current[10:12, 10:12] = 6
    inferred = infer_component_motion(previous, current)
    assert inferred is not None
    _, delta = inferred
    assert delta == (0, 1)


def test_infer_component_motion_requires_unique_match() -> None:
    previous = np.zeros((16, 16), dtype=np.int8)
    current = np.zeros((16, 16), dtype=np.int8)
    previous[2:4, 2:4] = 6
    previous[2:4, 8:10] = 6
    current[2:4, 3:5] = 6
    current[2:4, 9:11] = 6
    inferred = infer_component_motion(previous, current)
    assert inferred is None


def test_infer_actor_features_falls_back_to_changed_local_components() -> None:
    previous = np.zeros((16, 16), dtype=np.int8)
    current = np.zeros((16, 16), dtype=np.int8)
    previous[2:4, 2:4] = 5
    previous[10:12, 10:12] = 7
    current[2:4, 3:5] = 6
    current[10:12, 10:12] = 7
    features = infer_actor_features(previous, current, previous != current)
    assert (7, 4, 0, 0) not in features
    assert any(feature[0] in {5, 6} for feature in features)


def test_choose_action_prefers_key_score_on_tie() -> None:
    agent = ArcAgi3TACTICPublicAgent()
    frame = np.zeros((8, 8), dtype=np.int8)
    level_memory = LevelMemory()
    signature = frame_signature(frame)
    actions = (GameAction.ACTION1, GameAction.ACTION2)
    candidate = agent._choose_action(frame, signature, actions, EnvironmentMemory(), level_memory)
    key_scores = {
        action.name: key_stability_score((action.name, None)) for action in actions
    }
    expected = max(key_scores.items(), key=lambda item: item[1])[0]
    assert candidate.key[0] == expected


def test_transition_mask_ignores_ubiquitous_noise_pixels() -> None:
    memory = EnvironmentMemory()
    for _ in range(4):
        delta = np.zeros((8, 8), dtype=bool)
        delta[0, 7] = True
        memory.record_transition_mask(delta)
    previous = np.zeros((8, 8), dtype=np.int8)
    current = previous.copy()
    current[0, 7] = 1
    masked = transition_mask(previous, current, noise_mask=memory.noise_mask()[:8, :8])
    assert int(masked.sum()) == 0


def test_transition_mask_suppresses_small_border_only_changes() -> None:
    previous = np.zeros((8, 8), dtype=np.int8)
    current = previous.copy()
    current[0, 6] = 1
    current[0, 7] = 1
    masked = transition_mask(previous, current)
    assert int(masked.sum()) == 0
    assert is_small_border_only_change(previous != current)


def test_refinement_clicks_prioritize_changed_region_and_neighbors() -> None:
    delta = np.zeros((16, 16), dtype=bool)
    delta[4:6, 10:12] = True
    refined = refinement_clicks((5, 9), delta, limit=6)
    coords = [coord for coord, _ in refined]
    assert (5, 9) in coords
    assert any(row >= 4 and col >= 10 for row, col in coords)


def test_per_level_budget_tracks_current_level_not_environment_sum() -> None:
    baselines = (6, 13, 31)
    assert per_level_budget(baselines, 0) == 24
    assert per_level_budget(baselines, 1) == 39


def test_effect_diversity_drops_for_repeated_same_mask() -> None:
    memory = EnvironmentMemory()
    key = ("ACTION6", (-3, 1, 2, 7))
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 6:8] = True
    memory.action_stats[key] = ActionStats(attempts=3)
    memory.record_effect(key, mask)
    memory.record_effect(key, mask)
    assert delta_signature(mask)
    assert memory.effect_diversity(key) == 1.0 / 3.0


def test_record_motion_increments_actor_features_once() -> None:
    memory = EnvironmentMemory()
    feature = (5, 2, 0, 0)
    memory.record_motion("ACTION1", (1, 0), feature)
    assert memory.actor_features[feature] == 1


def test_coarse_cells_from_mask_orders_by_density() -> None:
    delta = np.zeros((32, 32), dtype=bool)
    delta[2:5, 2:5] = True
    delta[18:20, 18:20] = True
    cells = coarse_cells_from_mask(delta)
    assert cells[0] == (0, 0)
    assert (2, 2) in cells


@dataclass
class FakeObs:
    frame: tuple[np.ndarray, ...]
    levels_completed: int = 0
    state: ArcGameState = ArcGameState.NOT_FINISHED
    win_levels: int = 1


class RecordingAgent(ArcAgi3TACTICPublicAgent):
    def __init__(self) -> None:
        super().__init__()
        self.chosen: list = []

    def _choose_action(self, frame, signature, available_actions, memory, level_memory):
        candidate = super()._choose_action(frame, signature, available_actions, memory, level_memory)
        self.chosen.append(candidate)
        return candidate


class FakeAdvisor:
    def __init__(self, choice: str) -> None:
        self.choice = choice
        self.prompts: list[str] = []

    def choose_option(self, prompt: str, allowed_options: tuple[str, ...]) -> str | None:
        self.prompts.append(prompt)
        if self.choice in allowed_options:
            return self.choice
        return None

    def choose_action(self, prompt: str, allowed_actions: tuple[str, ...]) -> str | None:
        return self.choose_option(prompt, allowed_actions)


class FakeMechanicAdvisor:
    def __init__(self, summary: dict[str, object] | None) -> None:
        self.summary = summary
        self.prompts: list[str] = []

    def summarize_mechanic(
        self,
        prompt: str,
        *,
        allowed_modes: tuple[str, ...],
        allowed_goals: tuple[str, ...],
        allowed_focuses: tuple[str, ...],
    ) -> dict[str, object] | None:
        self.prompts.append(prompt)
        if self.summary is None:
            return None
        merged = dict(self.summary)
        merged.setdefault("mode", "UNKNOWN")
        merged.setdefault("goal", "UNKNOWN")
        merged.setdefault("focus", "UNKNOWN")
        merged.setdefault("confidence", 0.5)
        merged.setdefault("raw_text", "fake summary")
        assert merged["mode"] in allowed_modes
        assert merged["goal"] in allowed_goals
        assert merged["focus"] in allowed_focuses
        return merged


def test_prime_probe_queue_seeds_sweep_clicks_for_action6_only_levels() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[24:28, 24:28] = 7
    agent = ArcAgi3TACTICPublicAgent()
    level_memory = LevelMemory()
    agent._prime_probe_queue(FakeObs(frame=(frame,)), [GameAction.ACTION6], EnvironmentMemory(), level_memory)
    queue_coords = [
        (int(candidate.data["y"]), int(candidate.data["x"]))
        for candidate in level_memory.probe_queue
    ]
    sweep_coords = [coord for coord, _ in sweep_clicks()]
    assert queue_coords[: len(sweep_coords)] == sweep_coords
    assert agent.diagnostics.sweep_probe_levels == 1
    assert agent.diagnostics.sweep_probe_points == len(sweep_coords)


def test_prime_probe_queue_prioritizes_keyboard_before_click_on_mixed_levels() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[24:28, 24:28] = 7
    agent = ArcAgi3TACTICPublicAgent()
    level_memory = LevelMemory()
    agent._prime_probe_queue(
        FakeObs(frame=(frame,)),
        [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION6],
        EnvironmentMemory(),
        level_memory,
    )
    queued = list(level_memory.probe_queue)
    assert queued[0].action == GameAction.ACTION1
    assert queued[1].action == GameAction.ACTION2
    assert any(candidate.action == GameAction.ACTION6 for candidate in queued[2:])


def test_prime_probe_queue_can_bootstrap_click_first_on_small_click_like_mixed_level() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[24, 24] = 7
    agent = ArcAgi3TACTICPublicAgent()
    level_memory = LevelMemory()
    agent._prime_probe_queue(
        FakeObs(frame=(frame,)),
        [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION6],
        EnvironmentMemory(),
        level_memory,
    )
    queued = list(level_memory.probe_queue)
    assert queued[0].action == GameAction.ACTION6
    assert agent.diagnostics.bootstrap_probe_reorders == 1


def test_play_environment_requeues_local_refinement_after_changed_click() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.action_space = [GameAction.ACTION6]
            self._step_index = 0
            self._frame = np.zeros((64, 64), dtype=np.int8)

        def reset(self):
            self._step_index = 0
            self._frame = np.zeros((64, 64), dtype=np.int8)
            return FakeObs(frame=(self._frame.copy(),))

        def step(self, action, data=None):
            self._step_index += 1
            next_frame = self._frame.copy()
            if self._step_index == 1 and data is not None:
                row = int(data["y"])
                col = int(data["x"])
                next_frame[row : min(row + 2, 64), col : min(col + 2, 64)] = 9
                self._frame = next_frame
                return FakeObs(frame=(next_frame,), levels_completed=0, state=ArcGameState.NOT_FINISHED)
            return FakeObs(frame=(next_frame,), levels_completed=1, state=ArcGameState.WIN)

    agent = RecordingAgent()
    env = FakeEnv()
    agent.play_environment(env, env_id="fake-123", baseline_actions=(6,), show_progress=False)
    assert len(agent.chosen) >= 2
    first = agent.chosen[0]
    second = agent.chosen[1]
    clicked_coord = (int(first.data["y"]), int(first.data["x"]))
    delta = np.zeros((64, 64), dtype=bool)
    delta[clicked_coord[0] : min(clicked_coord[0] + 2, 64), clicked_coord[1] : min(clicked_coord[1] + 2, 64)] = True
    refined_coords = {coord for coord, _ in refinement_clicks(clicked_coord, delta)}
    assert (int(second.data["y"]), int(second.data["x"])) in refined_coords
    assert agent.diagnostics.refinement_clicks_enqueued > 0


def test_play_environment_enqueues_undo_after_changed_click_when_available() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.action_space = [GameAction.ACTION6, GameAction.ACTION7]
            self._step_index = 0
            self._frame = np.zeros((64, 64), dtype=np.int8)

        def reset(self):
            self._step_index = 0
            self._frame = np.zeros((64, 64), dtype=np.int8)
            return FakeObs(frame=(self._frame.copy(),))

        def step(self, action, data=None):
            self._step_index += 1
            next_frame = self._frame.copy()
            if self._step_index == 1 and action == GameAction.ACTION6 and data is not None:
                row = int(data["y"])
                col = int(data["x"])
                next_frame[row : min(row + 2, 64), col : min(col + 2, 64)] = 9
                self._frame = next_frame
                return FakeObs(frame=(next_frame,), levels_completed=0, state=ArcGameState.NOT_FINISHED)
            return FakeObs(frame=(next_frame,), levels_completed=1, state=ArcGameState.WIN)

    agent = ArcAgi3TACTICPublicAgent()
    env = FakeEnv()
    agent.play_environment(env, env_id="fake-undo", baseline_actions=(6,), show_progress=False)
    assert agent.diagnostics.click_undo_enqueued == 1


def test_enqueue_keyboard_followups_requeues_same_action_on_new_anchor() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[8:10, 8:10] = 5
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory()
    feature = extract_components(frame)[0].feature
    memory.record_motion("ACTION1", (0, 1), feature)
    executed = ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name)

    agent._enqueue_keyboard_followups(
        executed=executed,
        next_signature=frame_signature(frame),
        next_frame=frame,
        available_actions=(GameAction.ACTION1, GameAction.ACTION2),
        memory=memory,
        level_memory=level_memory,
        changed_pixels=12,
    )

    assert level_memory.probe_queue
    assert level_memory.probe_queue[0].action == GameAction.ACTION1
    assert agent.diagnostics.keyboard_followups_enqueued == 1


def test_enqueue_keyboard_followups_adds_turn_probe_on_plateau() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[8:10, 8:10] = 5
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory()
    feature = extract_components(frame)[0].feature
    memory.record_motion("ACTION1", (0, 1), feature)
    level_memory.actor_positions.add(extract_components(frame)[0].anchor)
    level_memory.keyboard_repeat_key = (GameAction.ACTION1.name, None)
    level_memory.keyboard_repeat_steps = 3
    executed = ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name)

    agent._enqueue_keyboard_followups(
        executed=executed,
        next_signature=frame_signature(frame),
        next_frame=frame,
        available_actions=(GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3),
        memory=memory,
        level_memory=level_memory,
        changed_pixels=1,
    )

    queued_actions = [candidate.action for candidate in level_memory.probe_queue]
    assert GameAction.ACTION1 not in queued_actions
    assert GameAction.ACTION2 in queued_actions
    assert GameAction.ACTION3 in queued_actions
    assert agent.diagnostics.keyboard_turn_probes_enqueued == 2


def test_enqueue_keyboard_followups_prioritizes_control_map_toward_target() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[8:10, 8:10] = 5
    frame[8:10, 14:16] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.record_motion("ACTION1", (0, -1), actor_feature)
    for _ in range(3):
        memory.record_motion("ACTION2", (0, 1), actor_feature)
    memory.record_motion("ACTION3", (1, 0), actor_feature)
    for _ in range(2):
        memory.interaction_target_stat_for(target_feature).update(4.0, changed=True, level_gain=True)
    level_memory.actor_positions.add(next(component.anchor for component in components if component.color == 5))
    level_memory.keyboard_repeat_key = (GameAction.ACTION1.name, None)
    level_memory.keyboard_repeat_steps = 3
    executed = ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name)

    agent._enqueue_keyboard_followups(
        executed=executed,
        next_signature=frame_signature(frame),
        next_frame=frame,
        available_actions=(GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3),
        memory=memory,
        level_memory=level_memory,
        changed_pixels=1,
    )

    assert level_memory.probe_queue
    assert level_memory.probe_queue[0].action == GameAction.ACTION2


def test_choose_action_penalizes_repeated_action6_without_progress() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[8:12, 8:12] = 5
    frame[40:44, 40:44] = 8
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory()
    signature = frame_signature(frame)
    candidates = agent._candidates(frame, (GameAction.ACTION6,), memory)
    penalized = candidates[0]
    alternative = candidates[1]
    penalized_stats = memory.stat_for(penalized.key)
    penalized_stats.attempts = 4
    penalized_stats.total_reward = 2.4
    penalized_stats.changed_count = 4
    mask = np.zeros((64, 64), dtype=bool)
    mask[8:10, 8:10] = True
    memory.record_effect(penalized.key, mask)
    chosen = agent._choose_action(frame, signature, (GameAction.ACTION6,), memory, level_memory)
    assert chosen.key == alternative.key
    assert agent.diagnostics.repeated_action6_penalty_events == 0


def test_choose_action_defers_clicks_while_keyboard_control_is_unknown() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[8:12, 8:12] = 5
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory()
    signature = frame_signature(frame)
    click_candidate = next(
        candidate
        for candidate in agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION6), memory)
        if candidate.action == GameAction.ACTION6
    )
    click_stats = memory.stat_for(click_candidate.key)
    click_stats.attempts = 1
    click_stats.total_reward = 1.5
    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION6), memory, level_memory)
    assert chosen.action == GameAction.ACTION1
    assert agent.diagnostics.mixed_action_click_deferrals == 0


def test_choose_action_penalizes_repeated_keyboard_no_change_streak() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory()
    signature = frame_signature(frame)
    action5_stats = memory.stat_for((GameAction.ACTION5.name, None))
    action4_stats = memory.stat_for((GameAction.ACTION4.name, None))
    action5_stats.attempts = 3
    action5_stats.total_reward = 0.9
    action4_stats.attempts = 1
    action4_stats.total_reward = 0.4
    level_memory.keyboard_no_change_streaks[GameAction.ACTION5.name] = 3
    chosen = agent._choose_action(frame, signature, (GameAction.ACTION4, GameAction.ACTION5), memory, level_memory)
    assert chosen.action == GameAction.ACTION4


def test_choose_action_uses_keyboard_context_stats_when_actor_is_known() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[8:12, 8:12] = 5
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    memory.actor_features[(5, 4, 0, 0)] = 2
    level_memory = LevelMemory()
    signature = frame_signature(frame)
    context_stat = memory.keyboard_context_stat_for(GameAction.ACTION2.name, (1, 1))
    context_stat.attempts = 2
    context_stat.total_reward = 1.6
    context_stat.changed_count = 2
    context_stat.level_gain_count = 1

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION2), memory, level_memory)
    assert chosen.action == GameAction.ACTION2


def test_planned_frontier_action_routes_to_reachable_unexplored_keyboard_branch() -> None:
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    frame = np.zeros((64, 64), dtype=np.int8)
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION2), memory)
    level_memory = LevelMemory()
    start = b"start"
    branch = b"branch"
    level_memory.mark_available(start, tuple(candidate.key for candidate in candidates))
    level_memory.mark_tried(start, (GameAction.ACTION1.name, None))
    level_memory.mark_tried(start, (GameAction.ACTION2.name, None))
    level_memory.mark_available(branch, tuple(candidate.key for candidate in candidates))
    level_memory.mark_tried(branch, (GameAction.ACTION1.name, None))
    level_memory.observe_transition(start, (GameAction.ACTION1.name, None), branch)

    planned = agent._planned_frontier_action(frame, start, candidates, memory, level_memory)
    assert planned is not None
    assert planned.action == GameAction.ACTION1


def test_choose_action_uses_frontier_plan_when_current_keyboard_state_is_exhausted() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[8:12, 8:12] = 5
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory()
    signature = frame_signature(frame)
    branch = b"branch"
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION2), memory)
    level_memory.mark_available(branch, tuple(candidate.key for candidate in candidates))
    level_memory.mark_tried(signature, (GameAction.ACTION1.name, None))
    level_memory.mark_tried(signature, (GameAction.ACTION2.name, None))
    level_memory.mark_tried(branch, (GameAction.ACTION1.name, None))
    level_memory.observe_transition(signature, (GameAction.ACTION1.name, None), branch)

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION2), memory, level_memory)
    assert chosen.action == GameAction.ACTION1
    assert agent.diagnostics.frontier_plan_routes == 1


def test_planned_rollout_action_prefers_sequence_toward_target_component() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[8:10, 8:10] = 5
    frame[8:10, 20:22] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=3)
    actor_feature = next(component.feature for component in extract_components(frame) if component.color == 5)
    memory.record_motion("ACTION1", (0, 1), actor_feature)
    memory.record_motion("ACTION2", (1, 0), actor_feature)

    planned = agent._planned_rollout_action(frame, (GameAction.ACTION1, GameAction.ACTION2), memory, level_memory)
    assert planned is not None
    assert planned.action == GameAction.ACTION1


def test_infer_interaction_targets_uses_contact_and_changed_region() -> None:
    previous = np.zeros((64, 64), dtype=np.int8)
    next_frame = np.zeros((64, 64), dtype=np.int8)
    previous[8:10, 8:10] = 5
    previous[8:10, 12:14] = 7
    previous[28:30, 28:30] = 9
    next_frame[8:10, 10:12] = 5
    next_frame[8:10, 12:14] = 7
    next_frame[28:30, 28:30] = 9

    previous_components = extract_components(previous)
    next_components = extract_components(next_frame)
    actor_before = next(component for component in previous_components if component.color == 5)
    actor_after = next(component for component in next_components if component.color == 5)
    target_feature = next(component.feature for component in previous_components if component.color == 7)

    targets = infer_interaction_targets(previous, next_frame, previous != next_frame, actor_before, actor_after)
    assert target_feature in targets
    assert actor_before.feature not in targets


def test_level_memory_exposes_recent_target_contacts_with_ttl() -> None:
    level_memory = LevelMemory(step_index=5)
    feature = (7, 4, 1, 1)
    level_memory.note_target_contacts((feature,))
    assert feature in level_memory.active_target_contacts()
    assert set(level_memory.active_target_cells()) == {(2, 2), (2, 3), (3, 2), (3, 3)}
    level_memory.step_index = 9
    assert feature not in level_memory.active_target_contacts()
    assert set(level_memory.active_target_cells()) == {(2, 2), (2, 3), (3, 2), (3, 3)}
    level_memory.step_index = 11
    assert not level_memory.active_target_cells()


def test_interaction_targets_prioritize_recent_contact_cell() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[8:10, 8:10] = 5
    frame[8:10, 16:18] = 7
    frame[24:26, 8:10] = 9
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=6)
    components = extract_components(frame)
    actor_component = next(component for component in components if component.color == 5)
    down_target = next(component for component in components if component.color == 9)
    level_memory.note_target_contacts((down_target.feature,))

    targets = agent._interaction_targets(frame, memory, level_memory, actor_component)

    assert targets
    assert coarse_cell_for_coord(targets[0][0]) == coarse_cell_for_coord(down_target.anchor)


def test_planned_rollout_action_prefers_rewarded_interaction_target_over_nearer_distractor() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[20:22, 20:22] = 5
    frame[20:22, 34:36] = 7
    frame[26:28, 20:22] = 9
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=3)
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    rewarded_target = next(component.feature for component in components if component.color == 7)
    memory.record_motion("ACTION1", (0, 1), actor_feature)
    memory.record_motion("ACTION2", (1, 0), actor_feature)
    memory.interaction_target_stat_for(rewarded_target).update(3.0, changed=True, level_gain=True)

    planned = agent._planned_rollout_action(frame, (GameAction.ACTION1, GameAction.ACTION2), memory, level_memory)
    assert planned is not None
    assert planned.action == GameAction.ACTION1


def test_planned_rollout_action_can_follow_recent_contact_cell() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[20:22, 20:22] = 5
    frame[20:22, 34:36] = 7
    frame[34:36, 20:22] = 9
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=6)
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    down_target = next(component for component in components if component.color == 9)
    memory.record_motion("ACTION1", (0, 1), actor_feature)
    memory.record_motion("ACTION2", (1, 0), actor_feature)
    level_memory.note_target_contacts((down_target.feature,))

    planned = agent._planned_rollout_action(frame, (GameAction.ACTION1, GameAction.ACTION2), memory, level_memory)

    assert planned is not None
    assert planned.action == GameAction.ACTION2


def test_target_affordance_score_prefers_rewarded_followup_branch() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[20:22, 20:22] = 5
    frame[20:24, 23:27] = 7
    frame[23:27, 20:24] = 9
    memory = EnvironmentMemory()
    components = extract_components(frame)
    right_target = next(component.feature for component in components if component.color == 7)
    down_target = next(component.feature for component in components if component.color == 9)
    memory.interaction_target_stat_for(down_target).update(1.0, changed=True, level_gain=False)
    memory.target_affordance_stat_for(right_target, "ACTION2").update(20.0, changed=True, level_gain=True)
    memory.target_affordance_stat_for(down_target, "ACTION2").update(1.0, changed=False, level_gain=False)

    assert memory.target_affordance_score(right_target, "ACTION2") > memory.target_affordance_score(
        down_target,
        "ACTION2",
    )


def test_choose_action_uses_rollout_plan_before_frontier_plan() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[8:10, 8:10] = 5
    frame[8:10, 20:22] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=3)
    signature = frame_signature(frame)
    actor_feature = next(component.feature for component in extract_components(frame) if component.color == 5)
    memory.record_motion("ACTION1", (0, 1), actor_feature)
    memory.record_motion("ACTION2", (1, 0), actor_feature)
    level_memory.mark_tried(signature, (GameAction.ACTION1.name, None))
    level_memory.mark_tried(signature, (GameAction.ACTION2.name, None))

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION2), memory, level_memory)
    assert chosen.action == GameAction.ACTION1
    assert agent.diagnostics.rollout_plan_choices == 1


def test_choose_action_can_use_contact_guided_rollout_before_probe() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[20:22, 20:22] = 5
    frame[34:36, 20:22] = 9
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=6)
    signature = frame_signature(frame)
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 9)
    memory.record_motion("ACTION1", (0, 1), actor_feature)
    memory.record_motion("ACTION2", (1, 0), actor_feature)
    memory.actor_features[actor_feature] = 3
    memory.interaction_target_stat_for(target_feature).update(3.0, changed=True, level_gain=True)
    level_memory.note_target_contacts((target_feature,))
    level_memory.family_no_progress_counts["MOVE"] = 1
    probe = ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name)
    level_memory.probe_queue.append(probe)

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION2), memory, level_memory)

    assert chosen.action == GameAction.ACTION2
    assert agent.diagnostics.rollout_plan_choices == 1


def test_choose_action_prefers_active_subgoal_cell_after_probe_phase() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[8:12, 8:12] = 5
    frame[40:44, 40:44] = 8
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory()
    signature = frame_signature(frame)
    candidates = agent._candidates(frame, (GameAction.ACTION6,), memory)
    target = next(
        candidate
        for candidate in candidates
        if coarse_cell_for_coord((int(candidate.data["y"]), int(candidate.data["x"]))) == (5, 5)
    )
    level_memory.record_subgoal_cells(((5, 5),), 3.0)
    chosen = agent._choose_action(frame, signature, (GameAction.ACTION6,), memory, level_memory)
    chosen_cell = coarse_cell_for_coord((int(chosen.data["y"]), int(chosen.data["x"])))
    target_cell = coarse_cell_for_coord((int(target.data["y"]), int(target.data["x"])))
    assert chosen_cell == target_cell


def test_choose_action_prefers_action5_when_actor_is_adjacent_to_rewarded_target() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[20:22, 20:22] = 5
    frame[20:22, 22:24] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=6)
    signature = frame_signature(frame)
    components = extract_components(frame)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.actor_features[next(component.feature for component in components if component.color == 5)] = 2
    memory.interaction_target_stat_for(target_feature).update(4.0, changed=True, level_gain=True)
    level_memory.note_target_contacts((target_feature,))

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION5), memory, level_memory)
    assert chosen.action == GameAction.ACTION5


def test_pop_probe_candidate_uses_control_map_bonus_to_choose_targeted_keyboard_probe() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[2:4, 2:4] = 5
    frame[2:4, 5:7] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    for _ in range(3):
        memory.record_motion("ACTION2", (0, 1), actor_feature)
    memory.record_motion("ACTION3", (1, 0), actor_feature)
    for _ in range(2):
        memory.interaction_target_stat_for(target_feature).update(4.0, changed=True, level_gain=True)
    level_memory = LevelMemory(
        step_index=6,
        mechanic_hint=MechanicHint(
            mode="MOVE",
            goal="CONTACT",
            focus="MOVING_OBJECT",
            confidence=0.7,
            source="SYMBOLIC",
        ),
    )
    level_memory.probe_queue.extend(
        (
            ActionCandidate(GameAction.ACTION3, None, (GameAction.ACTION3.name, None), GameAction.ACTION3.name),
            ActionCandidate(GameAction.ACTION2, None, (GameAction.ACTION2.name, None), GameAction.ACTION2.name),
        )
    )

    chosen = agent._pop_probe_candidate(frame, frame_signature(frame), memory, level_memory)

    assert chosen is not None
    assert chosen.action == GameAction.ACTION2


def test_choose_action_can_bypass_probe_queue_with_control_map_plan() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[2:4, 2:4] = 5
    frame[2:4, 5:7] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    for _ in range(3):
        memory.record_motion("ACTION1", (0, 1), actor_feature)
    memory.record_motion("ACTION2", (1, 0), actor_feature)
    for _ in range(3):
        memory.interaction_target_stat_for(target_feature).update(4.0, changed=True, level_gain=True)
    level_memory = LevelMemory(step_index=6, last_changed_step=3)
    signature = frame_signature(frame)
    level_memory.mark_seen(signature)
    level_memory.probe_queue.extend(
        (
            ActionCandidate(GameAction.ACTION2, None, (GameAction.ACTION2.name, None), GameAction.ACTION2.name),
            ActionCandidate(GameAction.ACTION6, {"x": 0, "y": 0}, (GameAction.ACTION6.name, target_feature), "ACTION6@0,0"),
        )
    )

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION6), memory, level_memory)

    assert chosen.action == GameAction.ACTION1
    assert agent.diagnostics.control_map_plan_choices == 1


def test_choose_action_can_use_control_map_from_actor_cell_prior() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10:12, 10:12] = 5
    frame[10:12, 18:20] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.actor_cell_scores[(1, 1)] = 3.0
    memory.action_motion["ACTION1"] = {(0, 1): 4}
    memory.action_motion["ACTION2"] = {(1, 0): 1}
    for _ in range(3):
        memory.interaction_target_stat_for(target_feature).update(4.0, changed=True, level_gain=True)
    level_memory = LevelMemory(step_index=6, last_changed_step=3)
    signature = frame_signature(frame)
    level_memory.mark_seen(signature)
    level_memory.probe_queue.extend(
        (
            ActionCandidate(GameAction.ACTION2, None, (GameAction.ACTION2.name, None), GameAction.ACTION2.name),
            ActionCandidate(GameAction.ACTION6, {"x": 0, "y": 0}, (GameAction.ACTION6.name, target_feature), "ACTION6@0,0"),
        )
    )

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION6), memory, level_memory)

    assert chosen.action == GameAction.ACTION1
    assert agent.diagnostics.control_map_plan_choices == 1


def test_choose_action_can_seed_control_commit_macro_after_strong_control_map_choice() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10:12, 10:12] = 5
    frame[10:12, 26:28] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.record_actor_hint(actor_feature, weight=4)
    for _ in range(4):
        memory.record_motion("ACTION1", (0, 1), actor_feature)
    for _ in range(3):
        memory.interaction_target_stat_for(target_feature).update(4.0, changed=True, level_gain=True)
    for _ in range(2):
        memory.stat_for((GameAction.ACTION1.name, None)).update(2.0, changed=True, level_gain=False)
    level_memory = LevelMemory(step_index=4, last_changed_step=4)
    level_memory.note_target_values((target_feature,), reward=4.0, changed=True, level_gain=True)
    level_memory.note_target_values((target_feature,), reward=4.0, changed=True, level_gain=True)
    signature = frame_signature(frame)
    level_memory.mark_seen(signature)
    level_memory.probe_queue.append(
        ActionCandidate(GameAction.ACTION2, None, (GameAction.ACTION2.name, None), GameAction.ACTION2.name)
    )

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION2), memory, level_memory)

    assert chosen.action == GameAction.ACTION1
    assert agent.diagnostics.control_map_plan_choices + agent.diagnostics.target_macro_plan_choices == 1
    assert level_memory.macro_queue
    assert all(candidate.action == GameAction.ACTION1 for candidate in level_memory.macro_queue)
    assert (
        agent.diagnostics.control_commit_injections == 1
        or agent.diagnostics.target_option_bundle_injections == 1
    )


def test_interaction_targets_prefer_feature_with_positive_target_value_memory() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10:12, 10:12] = 5
    frame[10:12, 18:20] = 7
    frame[18:20, 10:12] = 8
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=6)
    components = extract_components(frame)
    actor_component = next(component for component in components if component.color == 5)
    valued_target = next(component for component in components if component.color == 8)
    level_memory.note_target_values((valued_target.feature,), reward=4.0, changed=True, level_gain=True)
    level_memory.note_target_values((valued_target.feature,), reward=4.0, changed=True, level_gain=True)

    targets = agent._interaction_targets(frame, memory, level_memory, actor_component)

    assert targets
    assert targets[0][0] == valued_target.anchor


def test_interaction_targets_downrank_blocked_target_cell() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10:12, 10:12] = 5
    frame[10:12, 18:20] = 7
    frame[18:20, 10:12] = 8
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=6)
    components = extract_components(frame)
    actor_component = next(component for component in components if component.color == 5)
    blocked_target = next(component for component in components if component.color == 7)
    safe_target = next(component for component in components if component.color == 8)
    level_memory.note_blocked_target(blocked_target.anchor, weight=2.0)

    targets = agent._interaction_targets(frame, memory, level_memory, actor_component)

    assert targets
    assert targets[0][0] == safe_target.anchor


def test_target_approach_cells_prefer_less_blocked_contact_cell() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10:12, 18:20] = 5
    frame[14:16, 10:26] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=6)
    components = extract_components(frame)
    actor_component = next(component for component in components if component.color == 5)
    target_component = next(component for component in components if component.color == 7)
    right_bridge_cell = coarse_cell_for_coord((15, 26))
    level_memory.note_target_values((target_component.feature,), reward=4.0, changed=True, level_gain=True)
    level_memory.record_subgoal_cells((right_bridge_cell,), weight=2.0)
    level_memory.note_blocked_target((15, 9), weight=2.0)
    level_memory.actor_positions.update({(15, 9), (15, 10)})

    approach_cells = agent._target_approach_cells(
        frame,
        memory,
        level_memory,
        actor_component,
        top_target=target_component.anchor,
    )

    assert approach_cells
    assert coarse_cell_for_coord(approach_cells[0][0]) == right_bridge_cell


def test_seed_control_commit_bundle_requires_positive_target_value_support() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10:12, 10:12] = 5
    frame[10:12, 26:28] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_component = next(component for component in components if component.color == 5)
    target_component = next(component for component in components if component.color == 7)
    for _ in range(4):
        memory.record_motion("ACTION1", (0, 1), actor_component.feature)
    chosen = ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name)

    blocked_level_memory = LevelMemory(step_index=4, last_changed_step=4)
    agent._seed_control_commit_bundle(
        frame,
        (GameAction.ACTION1, GameAction.ACTION2),
        (chosen,),
        memory,
        blocked_level_memory,
        chosen,
        control_score=1.3,
        control_margin=0.2,
        actor_component=actor_component,
    )

    assert not blocked_level_memory.macro_queue
    assert agent.diagnostics.target_value_commit_blocks == 1

    gated_level_memory = LevelMemory(step_index=4, last_changed_step=4)
    gated_level_memory.note_target_values((target_component.feature,), reward=4.0, changed=True, level_gain=True)
    gated_level_memory.note_target_values((target_component.feature,), reward=4.0, changed=True, level_gain=True)

    agent._seed_control_commit_bundle(
        frame,
        (GameAction.ACTION1, GameAction.ACTION2),
        (chosen,),
        memory,
        gated_level_memory,
        chosen,
        control_score=1.3,
        control_margin=0.2,
        actor_component=actor_component,
    )

    assert gated_level_memory.macro_queue
    assert agent.diagnostics.target_value_commit_gates == 1
    assert agent.diagnostics.control_commit_injections == 1


def test_target_conditioned_macro_planner_prefers_move_interact_when_target_is_high_value() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10:12, 10:12] = 5
    frame[10:12, 13:15] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_component = next(component for component in components if component.color == 5)
    target_component = next(component for component in components if component.color == 7)
    for _ in range(4):
        memory.record_motion("ACTION1", (0, 1), actor_component.feature)
    for _ in range(3):
        memory.interaction_target_stat_for(target_component.feature).update(4.0, changed=True, level_gain=True)
    memory.target_affordance_stat_for(target_component.feature, GameAction.ACTION5.name).update(
        3.0,
        changed=True,
        level_gain=True,
    )
    level_memory = LevelMemory(step_index=6, last_changed_step=4)
    level_memory.note_target_values((target_component.feature,), reward=4.0, changed=True, level_gain=True)
    level_memory.note_target_values((target_component.feature,), reward=4.0, changed=True, level_gain=True)
    level_memory.mechanic_hint = MechanicHint(
        mode="INTERACT",
        goal="CONTACT",
        focus="MOVING_OBJECT",
        confidence=0.85,
        source="SYMBOLIC",
    )
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION5, GameAction.ACTION6), memory)

    chosen, score, margin, kind = agent._target_conditioned_macro_decision(
        frame,
        candidates,
        (GameAction.ACTION1, GameAction.ACTION5, GameAction.ACTION6),
        memory,
        level_memory,
        actor_component=actor_component,
    )

    assert chosen is not None
    assert chosen.action == GameAction.ACTION1
    assert kind in {"MOVE_INTERACT", "OPTION_PATH"}
    assert score >= agent.tuning.target_macro_accept_score
    assert margin >= agent.tuning.target_macro_margin_floor
    if kind == "OPTION_PATH":
        assert level_memory.pending_option_plan is not None
        assert level_memory.pending_option_plan.follow_ups[-1].action == GameAction.ACTION5
        assert level_memory.pending_option_plan.finisher_kind == "INTERACT"


def test_target_conditioned_macro_planner_can_choose_click_probe_when_move_is_weak() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[2:4, 2:4] = 5
    frame[2:4, 5:7] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_component = next(component for component in components if component.color == 5)
    target_component = next(component for component in components if component.color == 7)
    memory.click_stat_for(target_component.feature).update(5.0, changed=True, level_gain=True)
    level_memory = LevelMemory(
        step_index=6,
        last_changed_step=4,
        mechanic_hint=MechanicHint(
            mode="CLICK",
            goal="CLEAR",
            focus="RARE_COLOR",
            confidence=0.9,
            source="SYMBOLIC",
        ),
    )
    level_memory.note_target_values((target_component.feature,), reward=3.0, changed=True, level_gain=False)
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION6), memory)

    chosen, score, margin, kind = agent._target_conditioned_macro_decision(
        frame,
        candidates,
        (GameAction.ACTION1, GameAction.ACTION6),
        memory,
        level_memory,
        actor_component=actor_component,
    )

    assert chosen is not None
    assert chosen.action == GameAction.ACTION6
    assert kind == "CLICK_PROBE"
    assert score >= agent.tuning.target_macro_accept_score
    assert margin >= agent.tuning.target_macro_margin_floor


def test_target_option_bundle_plan_can_finish_with_interact() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10:12, 10:12] = 5
    frame[10:12, 13:15] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_component = next(component for component in components if component.color == 5)
    target_component = next(component for component in components if component.color == 7)
    for _ in range(4):
        memory.record_motion("ACTION1", (0, 1), actor_component.feature)
    memory.target_affordance_stat_for(target_component.feature, GameAction.ACTION5.name).update(
        3.0,
        changed=True,
        level_gain=True,
    )
    level_memory = LevelMemory(step_index=6, last_changed_step=4)
    level_memory.note_target_values((target_component.feature,), reward=4.0, changed=True, level_gain=True)
    level_memory.note_target_values((target_component.feature,), reward=4.0, changed=True, level_gain=True)
    level_memory.mechanic_hint = MechanicHint(
        mode="INTERACT",
        goal="CONTACT",
        focus="MOVING_OBJECT",
        confidence=0.85,
        source="SYMBOLIC",
    )
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION5, GameAction.ACTION6), memory)

    plan = agent._target_option_bundle_plan(
        frame,
        candidates,
        (GameAction.ACTION1, GameAction.ACTION5, GameAction.ACTION6),
        memory,
        level_memory,
        actor_component=actor_component,
        top_target=target_component.anchor,
        top_target_score=1.8,
        planning_target=target_component.anchor,
        planning_target_score=1.8,
        planning_target_value=level_memory.target_cell_value_bonus(coarse_cell_for_coord(target_component.anchor)),
        planning_blocked_penalty=0.0,
        bridge_bonus=0.0,
    )

    assert plan is not None
    assert plan.first_candidate.action == GameAction.ACTION1
    assert plan.follow_ups
    assert plan.follow_ups[-1].action == GameAction.ACTION5
    assert plan.finisher_kind == "INTERACT"


def test_target_option_bundle_plan_can_finish_with_click_when_interact_is_unavailable() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10:12, 10:12] = 5
    frame[10:12, 26:28] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_component = next(component for component in components if component.color == 5)
    target_component = next(component for component in components if component.color == 7)
    for _ in range(4):
        memory.record_motion("ACTION1", (0, 8), actor_component.feature)
    memory.click_stat_for(target_component.feature).update(5.0, changed=True, level_gain=True)
    level_memory = LevelMemory(step_index=6, last_changed_step=4)
    level_memory.note_target_values((target_component.feature,), reward=3.0, changed=True, level_gain=False)
    level_memory.mechanic_hint = MechanicHint(
        mode="CLICK",
        goal="TOGGLE",
        focus="RARE_COLOR",
        confidence=0.8,
        source="SYMBOLIC",
    )
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION6), memory)

    plan = agent._target_option_bundle_plan(
        frame,
        candidates,
        (GameAction.ACTION1, GameAction.ACTION6),
        memory,
        level_memory,
        actor_component=actor_component,
        top_target=target_component.anchor,
        top_target_score=1.2,
        planning_target=target_component.anchor,
        planning_target_score=1.2,
        planning_target_value=level_memory.target_cell_value_bonus(coarse_cell_for_coord(target_component.anchor)),
        planning_blocked_penalty=0.0,
        bridge_bonus=0.0,
    )

    assert plan is not None
    assert plan.first_candidate.action == GameAction.ACTION1
    assert plan.follow_ups
    assert plan.follow_ups[-1].action == GameAction.ACTION6
    assert plan.finisher_kind == "CLICK"


def test_choose_action_uses_target_conditioned_macro_planner_before_control_map() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10:12, 10:12] = 5
    frame[10:12, 13:15] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    for _ in range(4):
        memory.record_motion("ACTION1", (0, 1), actor_feature)
    for _ in range(3):
        memory.interaction_target_stat_for(target_feature).update(4.0, changed=True, level_gain=True)
    memory.target_affordance_stat_for(target_feature, GameAction.ACTION5.name).update(
        3.0,
        changed=True,
        level_gain=True,
    )
    level_memory = LevelMemory(step_index=6, last_changed_step=4)
    level_memory.note_target_values((target_feature,), reward=4.0, changed=True, level_gain=True)
    level_memory.note_target_values((target_feature,), reward=4.0, changed=True, level_gain=True)
    signature = frame_signature(frame)
    level_memory.mark_seen(signature)
    level_memory.mechanic_hint = MechanicHint(
        mode="INTERACT",
        goal="CONTACT",
        focus="MOVING_OBJECT",
        confidence=0.85,
        source="SYMBOLIC",
    )

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION5, GameAction.ACTION6), memory, level_memory)

    assert chosen.action == GameAction.ACTION1
    assert agent.diagnostics.target_macro_plan_choices == 1
    assert (
        agent.diagnostics.target_macro_move_interact_choices == 1
        or agent.diagnostics.target_option_plan_choices == 1
    )
    assert level_memory.macro_queue
    assert level_memory.macro_queue[-1].action == GameAction.ACTION5


def test_choose_action_can_use_target_adjacency_bridge_cell() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10:12, 18:20] = 5
    frame[14:16, 10:26] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.record_actor_hint(actor_feature, weight=4)
    for _ in range(4):
        memory.record_motion("ACTION1", (0, 1), actor_feature)
    for _ in range(2):
        memory.record_motion("ACTION2", (0, -1), actor_feature)
    for _ in range(3):
        memory.record_motion("ACTION3", (1, 0), actor_feature)
    for _ in range(3):
        memory.interaction_target_stat_for(target_feature).update(4.0, changed=True, level_gain=True)
    memory.target_affordance_stat_for(target_feature, GameAction.ACTION5.name).update(
        3.0,
        changed=True,
        level_gain=True,
    )
    level_memory = LevelMemory(step_index=6, last_changed_step=4)
    level_memory.note_target_values((target_feature,), reward=4.0, changed=True, level_gain=True)
    level_memory.record_subgoal_cells((coarse_cell_for_coord((15, 26)),), weight=2.0)
    level_memory.note_blocked_target((15, 9), weight=2.0)
    level_memory.actor_positions.update({(15, 9), (15, 10)})
    signature = frame_signature(frame)
    level_memory.mark_seen(signature)

    chosen = agent._choose_action(
        frame,
        signature,
        (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION5),
        memory,
        level_memory,
    )

    assert chosen.action in {GameAction.ACTION1, GameAction.ACTION3}
    assert agent.diagnostics.target_adjacency_cells_considered > 0
    assert agent.diagnostics.target_adjacency_bridge_activations > 0


def test_control_rollout_synthesis_can_override_posterior_when_both_planners_agree() -> None:
    agent = ArcAgi3TACTICPublicAgent()
    posterior = ActionCandidate(GameAction.ACTION2, None, (GameAction.ACTION2.name, None), GameAction.ACTION2.name)
    agreed = ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name)

    chosen, score, margin = agent._control_rollout_synthesis_decision(
        posterior,
        "KEYBOARD",
        agreed,
        1.02,
        0.18,
        agreed,
        1.08,
        0.12,
        pressure=3,
    )

    assert chosen is not None
    assert chosen.action == GameAction.ACTION1
    assert score > 1.0
    assert margin >= 0.12


def test_control_repeat_penalty_grows_for_repeated_keyboard_loop_toward_same_target() -> None:
    frame = np.zeros((32, 32), dtype=np.int8)
    frame[10:12, 10:12] = 5
    frame[10:12, 18:20] = 7
    agent = ArcAgi3TACTICPublicAgent()
    level_memory = LevelMemory(step_index=8, last_changed_step=2)
    level_memory.keyboard_repeat_key = (GameAction.ACTION1.name, None)
    level_memory.keyboard_repeat_steps = 4
    level_memory.keyboard_plateau_steps = 3
    level_memory.actor_positions.add((10, 11))

    penalty = agent._control_repeat_penalty(
        action_key=(GameAction.ACTION1.name, None),
        current_anchor=(10, 10),
        next_anchor=(10, 11),
        level_memory=level_memory,
        interaction_targets=(((10, 18), 1.0),),
        pressure=3,
    )

    other_penalty = agent._control_repeat_penalty(
        action_key=(GameAction.ACTION2.name, None),
        current_anchor=(10, 10),
        next_anchor=(11, 10),
        level_memory=level_memory,
        interaction_targets=(((10, 18), 1.0),),
        pressure=3,
    )

    assert penalty > 0.5
    assert other_penalty == 0.0


def test_control_map_candidate_score_penalizes_repeated_keyboard_loop() -> None:
    frame = np.zeros((32, 32), dtype=np.int8)
    frame[10:12, 10:12] = 5
    frame[10:12, 18:20] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_component = next(component for component in components if component.color == 5)
    target_component = next(component for component in components if component.color == 7)
    memory.record_motion("ACTION1", (0, 1), actor_component.feature)
    for _ in range(3):
        memory.stat_for((GameAction.ACTION1.name, None)).update(1.0, changed=True, level_gain=False)

    base_level_memory = LevelMemory(step_index=8, last_changed_step=2)
    penalized_level_memory = LevelMemory(step_index=8, last_changed_step=2)
    penalized_level_memory.keyboard_repeat_key = (GameAction.ACTION1.name, None)
    penalized_level_memory.keyboard_repeat_steps = 4
    penalized_level_memory.keyboard_plateau_steps = 3
    penalized_level_memory.actor_positions.add((10, 11))
    for _ in range(3):
        penalized_level_memory.mark_seen(frame_signature(frame))

    candidate = ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name)
    interaction_targets = ((target_component.anchor, 1.0),)
    base_score = agent._control_map_candidate_score(
        candidate,
        frame,
        memory,
        base_level_memory,
        actor_component=actor_component,
        interaction_targets=interaction_targets,
    )
    penalized_score = agent._control_map_candidate_score(
        candidate,
        frame,
        memory,
        penalized_level_memory,
        actor_component=actor_component,
        interaction_targets=interaction_targets,
    )

    assert penalized_score < base_score


def test_coarse_path_distance_routes_around_blocked_cells() -> None:
    blocked = frozenset({(1, 2), (1, 3), (1, 4)})

    direct = coarse_path_distance((1, 1), (1, 5))
    detoured = coarse_path_distance((1, 1), (1, 5), blocked=blocked)

    assert direct == 4
    assert detoured > direct
    assert detoured == 6


def test_coarse_blocked_cells_for_components_marks_large_obstacles_only() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[8:16, 8:16] = 5
    frame[8:16, 24:40] = 7
    frame[24:40, 8:24] = 9
    components = extract_components(frame)
    actor = next(component for component in components if component.color == 5)
    target_cell = coarse_cell_for_coord((12, 30))

    blocked = coarse_blocked_cells_for_components(
        frame,
        actor_feature=actor.feature,
        target_cells=frozenset({target_cell}),
    )

    assert coarse_cells_for_bounds((8, 24, 15, 39))[0] not in blocked
    assert coarse_cell_for_coord(actor.anchor) not in blocked
    assert (3, 1) in blocked


def test_control_map_candidate_score_uses_path_detour_signal() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[8:16, 8:16] = 5
    frame[0:32, 24:32] = 9
    frame[8:16, 40:48] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_component = next(component for component in components if component.color == 5)
    target_component = next(component for component in components if component.color == 7)
    memory.record_motion("ACTION1", (0, 8), actor_component.feature)
    memory.record_motion("ACTION2", (24, 0), actor_component.feature)
    for _ in range(3):
        memory.interaction_target_stat_for(target_component.feature).update(4.0, changed=True, level_gain=True)
    level_memory = LevelMemory(step_index=6, last_changed_step=3)
    candidate_right = ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name)
    candidate_down = ActionCandidate(GameAction.ACTION2, None, (GameAction.ACTION2.name, None), GameAction.ACTION2.name)
    interaction_targets = ((target_component.anchor, 1.0),)

    score_right = agent._control_map_candidate_score(
        candidate_right,
        frame,
        memory,
        level_memory,
        actor_component=actor_component,
        interaction_targets=interaction_targets,
    )
    score_down = agent._control_map_candidate_score(
        candidate_down,
        frame,
        memory,
        level_memory,
        actor_component=actor_component,
        interaction_targets=interaction_targets,
    )

    assert score_down > score_right


def test_control_commit_validation_aborts_when_move_does_not_reduce_distance() -> None:
    agent = ArcAgi3TACTICPublicAgent()
    level_memory = LevelMemory()
    level_memory.start_control_commit(
        primary_action=GameAction.ACTION1.name,
        target=(10, 14),
        last_distance=4,
        steps_remaining=2,
        allow_interact=False,
    )
    level_memory.macro_queue.append(
        ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name)
    )
    current_actor = FrameComponent(5, 4, (10, 10), (5, 4, 0, 0), (10, 10, 11, 11))
    next_actor = FrameComponent(5, 4, (10, 10), (5, 4, 0, 0), (10, 10, 11, 11))

    agent._update_control_commit_after_transition(
        level_memory,
        ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name),
        current_actor_component=current_actor,
        next_actor_component=next_actor,
        changed=False,
        level_gain=0,
    )

    assert not level_memory.macro_queue
    assert level_memory.control_commit_steps_remaining == 0
    assert agent.diagnostics.control_commit_aborts == 1


def test_control_commit_validation_keeps_sequence_when_progress_happens() -> None:
    agent = ArcAgi3TACTICPublicAgent()
    level_memory = LevelMemory()
    level_memory.start_control_commit(
        primary_action=GameAction.ACTION1.name,
        target=(10, 14),
        last_distance=4,
        steps_remaining=3,
        allow_interact=True,
    )
    level_memory.macro_queue.extend(
        (
            ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name),
            ActionCandidate(GameAction.ACTION5, None, (GameAction.ACTION5.name, None), GameAction.ACTION5.name),
        )
    )
    current_actor = FrameComponent(5, 4, (10, 10), (5, 4, 0, 0), (10, 10, 11, 11))
    next_actor = FrameComponent(5, 4, (10, 11), (5, 4, 0, 0), (10, 11, 11, 12))

    agent._update_control_commit_after_transition(
        level_memory,
        ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name),
        current_actor_component=current_actor,
        next_actor_component=next_actor,
        changed=True,
        level_gain=0,
    )

    assert level_memory.control_commit_steps_remaining == 2
    assert len(level_memory.macro_queue) == 2
    assert agent.diagnostics.control_commit_validations == 1


def test_interaction_graph_decision_prefers_action_with_better_recorded_transition() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10:12, 10:12] = 5
    frame[10:12, 18:20] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.record_actor_hint(actor_feature, weight=4)
    for _ in range(3):
        memory.interaction_target_stat_for(target_feature).update(4.0, changed=True, level_gain=True)
    level_memory = LevelMemory(step_index=6, last_changed_step=3)
    available = (GameAction.ACTION1, GameAction.ACTION2)
    current_key = agent._interaction_state_key(frame, available, memory, level_memory)
    level_memory.note_interaction_state(frame_signature(frame), current_key)
    level_memory.note_interaction_state(b"repeat-current", current_key)
    level_memory.note_interaction_state(b"repeat-current-2", current_key)
    better_next = (1, 2, 1, 2, 1, 1, action_mask(available))
    worse_next = (1, 1, 1, 2, 0, 0, action_mask(available))
    level_memory.record_interaction_transition(
        current_key,
        GameAction.ACTION1.name,
        "MOVE",
        better_next,
        reward=4.0,
        changed=True,
        level_gain=True,
    )
    level_memory.record_interaction_transition(
        current_key,
        GameAction.ACTION1.name,
        "MOVE",
        better_next,
        reward=4.0,
        changed=True,
        level_gain=True,
    )
    level_memory.record_interaction_transition(
        current_key,
        GameAction.ACTION2.name,
        "MOVE",
        worse_next,
        reward=0.0,
        changed=False,
        level_gain=False,
    )
    candidates = agent._candidates(frame, available, memory)

    chosen, score, margin = agent._interaction_graph_decision(frame, candidates, available, memory, level_memory)

    assert chosen is not None
    assert chosen.action == GameAction.ACTION1
    assert score > 1.0
    assert margin > 0.1


def test_planned_frontier_action_routes_to_nearest_state_with_untried_action() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION2), memory)
    level_memory = LevelMemory()
    start = b"start"
    frontier = b"frontier"

    level_memory.mark_available(start, tuple(candidate.key for candidate in candidates))
    level_memory.mark_tried(start, (GameAction.ACTION1.name, None))
    level_memory.mark_tried(start, (GameAction.ACTION2.name, None))
    level_memory.mark_available(frontier, tuple(candidate.key for candidate in candidates))
    level_memory.mark_tried(frontier, (GameAction.ACTION1.name, None))
    level_memory.observe_transition(start, (GameAction.ACTION1.name, None), frontier)

    first = agent._planned_frontier_action(frame, start, candidates, memory, level_memory)
    assert first is not None
    assert first.key == (GameAction.ACTION1.name, None)

    second = agent._planned_frontier_action(frame, frontier, candidates, memory, level_memory)
    assert second is not None
    assert second.key == (GameAction.ACTION2.name, None)


def test_planned_frontier_action_does_not_override_local_untried_actions_when_not_stalled() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION2), memory)
    level_memory = LevelMemory()
    start = b"start"
    frontier = b"frontier"

    level_memory.mark_available(start, tuple(candidate.key for candidate in candidates))
    level_memory.mark_tried(start, (GameAction.ACTION1.name, None))
    level_memory.mark_available(frontier, tuple(candidate.key for candidate in candidates))
    level_memory.mark_tried(frontier, (GameAction.ACTION1.name, None))
    level_memory.observe_transition(start, (GameAction.ACTION1.name, None), frontier)
    level_memory.last_changed_step = 0
    level_memory.step_index = 2

    planned = agent._planned_frontier_action(frame, start, candidates, memory, level_memory)
    assert planned is None


def test_planned_frontier_action_can_use_abstract_state_when_no_actor_anchor_exists() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION2), memory)
    level_memory = LevelMemory(step_index=8, last_changed_step=0)
    start = b"start"
    frontier = b"frontier"

    level_memory.mark_available(start, tuple(candidate.key for candidate in candidates))
    level_memory.mark_tried(start, (GameAction.ACTION1.name, None))
    level_memory.mark_tried(start, (GameAction.ACTION2.name, None))
    level_memory.mark_available(frontier, tuple(candidate.key for candidate in candidates))
    level_memory.mark_tried(frontier, (GameAction.ACTION1.name, None))
    level_memory.observe_transition(start, (GameAction.ACTION1.name, None), frontier)
    level_memory.note_abstract_state(start, ("UNKNOWN", "UNKNOWN", -1, -1, -1, -1, 2))
    level_memory.note_abstract_state(frontier, ("MOVE", "CONTACT", -1, -1, 2, 3, 2))

    planned = agent._planned_frontier_action(frame, start, candidates, memory, level_memory)

    assert planned is not None
    assert planned.key == (GameAction.ACTION1.name, None)
    assert agent.diagnostics.abstract_frontier_plan_routes == 1


def test_choose_action_uses_abstract_frontier_when_current_state_is_exhausted_and_actorless() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=8, last_changed_step=0)
    signature = frame_signature(frame)
    frontier = b"frontier"
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION2), memory)

    level_memory.mark_available(signature, tuple(candidate.key for candidate in candidates))
    level_memory.mark_tried(signature, (GameAction.ACTION1.name, None))
    level_memory.mark_tried(signature, (GameAction.ACTION2.name, None))
    level_memory.mark_available(frontier, tuple(candidate.key for candidate in candidates))
    level_memory.mark_tried(frontier, (GameAction.ACTION1.name, None))
    level_memory.observe_transition(signature, (GameAction.ACTION1.name, None), frontier)
    level_memory.note_abstract_state(frontier, ("MOVE", "CONTACT", -1, -1, 2, 3, 2))

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION2), memory, level_memory)

    assert chosen.key == (GameAction.ACTION1.name, None)
    assert agent.diagnostics.frontier_plan_routes == 1
    assert agent.diagnostics.abstract_frontier_plan_routes == 1


def test_choose_action_prioritizes_abstract_frontier_before_probe_in_actorless_loop() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=8, last_changed_step=1)
    signature = frame_signature(frame)
    frontier = b"frontier"
    loop = b"loop"
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION2), memory)
    abstract_key = agent._abstract_state_key(frame, (GameAction.ACTION1, GameAction.ACTION2), memory, level_memory)

    level_memory.mark_seen(signature)
    level_memory.mark_seen(signature)
    level_memory.family_no_progress_counts["MOVE"] = 2
    level_memory.mark_available(signature, tuple(candidate.key for candidate in candidates))
    level_memory.mark_available(frontier, tuple(candidate.key for candidate in candidates))
    level_memory.mark_tried(frontier, (GameAction.ACTION1.name, None))
    level_memory.observe_transition(signature, (GameAction.ACTION1.name, None), frontier)
    level_memory.note_abstract_state(signature, abstract_key)
    level_memory.note_abstract_state(loop, abstract_key)
    level_memory.note_abstract_state(frontier, ("MOVE", "CONTACT", -1, -1, 2, 3, 2))
    level_memory.probe_queue.append(
        ActionCandidate(GameAction.ACTION2, None, (GameAction.ACTION2.name, None), GameAction.ACTION2.name)
    )

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION2), memory, level_memory)

    assert chosen.key == (GameAction.ACTION1.name, None)
    assert agent.diagnostics.frontier_plan_routes == 1
    assert agent.diagnostics.abstract_frontier_plan_routes == 1


def test_choose_action_can_use_abstract_frontier_in_actorful_loop() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[8:10, 8:10] = 5
    frame[8:10, 14:16] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=8, last_changed_step=2)
    signature = frame_signature(frame)
    frontier = b"frontier"
    loop = b"loop"
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION2), memory)
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    memory.actor_features[actor_feature] = 3
    abstract_key = agent._abstract_state_key(frame, (GameAction.ACTION1, GameAction.ACTION2), memory, level_memory)

    level_memory.mark_seen(signature)
    level_memory.mark_seen(signature)
    level_memory.family_no_progress_counts["MOVE"] = 2
    level_memory.mark_available(signature, tuple(candidate.key for candidate in candidates))
    level_memory.mark_available(frontier, tuple(candidate.key for candidate in candidates))
    level_memory.mark_tried(frontier, (GameAction.ACTION2.name, None))
    level_memory.observe_transition(signature, (GameAction.ACTION1.name, None), frontier)
    level_memory.note_abstract_state(signature, abstract_key)
    level_memory.note_abstract_state(loop, abstract_key)
    level_memory.note_abstract_state(frontier, ("MOVE", "CONTACT", 0, 0, 0, 1, 2))
    level_memory.note_actor_anchor(signature, (8, 8))
    level_memory.note_actor_anchor(frontier, (8, 10))
    level_memory.probe_queue.append(
        ActionCandidate(GameAction.ACTION2, None, (GameAction.ACTION2.name, None), GameAction.ACTION2.name)
    )

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION2), memory, level_memory)

    assert chosen.key == (GameAction.ACTION1.name, None)
    assert agent.diagnostics.frontier_plan_routes == 1
    assert agent.diagnostics.abstract_frontier_plan_routes == 1


def test_agent_does_not_enable_advisor_by_default() -> None:
    agent = ArcAgi3TACTICPublicAgent()
    assert agent.action_advisor is None
    assert agent.action_advisor_budget_per_level == 0
    assert agent.mechanic_advisor is None
    assert agent.mechanic_advisor_budget_per_level == 0


def test_mechanic_hint_refresh_records_prompt_and_hint() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[1:3, 1:3] = 5
    advisor = FakeMechanicAdvisor(
        {"mode": "CLICK", "goal": "CLEAR", "focus": "RARE_COLOR", "confidence": 0.8}
    )
    agent = ArcAgi3TACTICPublicAgent(mechanic_advisor=advisor, mechanic_advisor_budget_per_level=1)
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=6, last_changed_step=2)
    level_memory.recent_events.append("ACTION1 changed=0 level_gain=0 reward=0.00")
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION6), memory)

    agent._maybe_refresh_mechanic_hint(frame, candidates, memory, level_memory)

    assert level_memory.mechanic_hint == MechanicHint(
        mode="CLICK",
        goal="CLEAR",
        focus="RARE_COLOR",
        confidence=0.8,
        source_step=6,
        source="QWEN",
        raw_text="fake summary",
    )
    assert agent.diagnostics.qwen_calls == 1
    assert agent.diagnostics.qwen_hint_calls == 1
    assert agent.diagnostics.qwen_hint_refreshes == 1
    assert advisor.prompts
    assert "recent_events:" in advisor.prompts[0]
    assert "component_summary:" in advisor.prompts[0]


def test_mechanic_hint_hybrid_prefers_symbolic_when_qwen_is_weaker() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[1:3, 1:3] = 5
    advisor = FakeMechanicAdvisor(
        {"mode": "MOVE", "goal": "CONTACT", "focus": "MOVING_OBJECT", "confidence": 0.35}
    )
    agent = ArcAgi3TACTICPublicAgent(mechanic_advisor=advisor, mechanic_advisor_budget_per_level=1)
    memory = EnvironmentMemory()
    feature = extract_components(frame)[0].feature
    for _ in range(4):
        memory.click_stat_for(feature).update(4.0, changed=True, level_gain=True)
    level_memory = LevelMemory(step_index=8, last_changed_step=0)
    level_memory.recent_events.append("ACTION6 changed=1 level_gain=1 reward=4.00")
    level_memory.note_target_contacts((feature,))
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION6), memory)

    agent._maybe_refresh_mechanic_hint(frame, candidates, memory, level_memory)

    assert level_memory.mechanic_hint is not None
    assert level_memory.mechanic_hint.source == "SYMBOLIC"
    assert agent.diagnostics.qwen_calls == 1
    assert agent.diagnostics.qwen_hint_calls == 1
    assert agent.diagnostics.qwen_hint_refreshes == 0
    assert agent.diagnostics.symbolic_hint_refreshes == 1
    assert advisor.prompts
    assert "symbolic_baseline=mode:" in advisor.prompts[0]
    assert "active_target_cells=" in advisor.prompts[0]


def test_mechanic_hint_hybrid_can_override_symbolic_when_qwen_is_stronger() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[1:3, 1:3] = 5
    advisor = FakeMechanicAdvisor(
        {"mode": "MOVE", "goal": "CONTACT", "focus": "MOVING_OBJECT", "confidence": 0.95}
    )
    agent = ArcAgi3TACTICPublicAgent(mechanic_advisor=advisor, mechanic_advisor_budget_per_level=1)
    memory = EnvironmentMemory()
    feature = extract_components(frame)[0].feature
    memory.click_stat_for(feature).update(1.0, changed=True, level_gain=False)
    level_memory = LevelMemory(step_index=8, last_changed_step=0)
    level_memory.recent_events.append("ACTION6 changed=1 level_gain=0 reward=1.00")
    signature = frame_signature(frame)
    for _ in range(4):
        level_memory.mark_seen(signature)
    level_memory.family_no_progress_counts["CLICK"] = 2
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION6), memory)

    agent._maybe_refresh_mechanic_hint(frame, candidates, memory, level_memory)

    assert level_memory.mechanic_hint == MechanicHint(
        mode="MOVE",
        goal="CONTACT",
        focus="MOVING_OBJECT",
        confidence=0.95,
        source_step=8,
        source="QWEN",
        raw_text="fake summary",
    )
    assert agent.diagnostics.qwen_calls == 1
    assert agent.diagnostics.qwen_hint_refreshes == 1


def test_symbolic_mechanic_hint_refreshes_without_qwen() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[1:3, 1:3] = 5
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    feature = extract_components(frame)[0].feature
    for _ in range(3):
        memory.click_stat_for(feature).update(4.0, changed=True, level_gain=True)
    level_memory = LevelMemory(step_index=6, last_changed_step=2)
    level_memory.recent_events.append("ACTION6 changed=1 level_gain=0 reward=1.00")
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION6), memory)

    agent._maybe_refresh_mechanic_hint(frame, candidates, memory, level_memory)

    assert level_memory.mechanic_hint is not None
    assert level_memory.mechanic_hint.mode in {"CLICK", "MIXED"}
    assert level_memory.mechanic_hint.goal in {"CLEAR", "TOGGLE", "COLLECT"}
    assert level_memory.mechanic_hint.confidence >= 0.32
    assert level_memory.mechanic_hint.source == "SYMBOLIC"
    assert level_memory.mechanic_hint.raw_text.startswith("symbolic")
    assert agent.diagnostics.symbolic_hint_refreshes == 1


def test_mechanic_hint_can_bias_click_family_without_direct_action_choice() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[1:3, 1:3] = 5
    advisor = FakeMechanicAdvisor(
        {"mode": "CLICK", "goal": "CLEAR", "focus": "SMALL_OBJECT", "confidence": 0.9}
    )
    agent = ArcAgi3TACTICPublicAgent(mechanic_advisor=advisor, mechanic_advisor_budget_per_level=1)
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=9, last_changed_step=1)
    level_memory.recent_events.append("ACTION1 changed=0 level_gain=0 reward=0.00")
    signature = frame_signature(frame)

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION6), memory, level_memory)

    assert chosen.action == GameAction.ACTION6
    assert agent.diagnostics.qwen_hint_calls == 1
    assert level_memory.mechanic_hint is not None


def test_symbolic_mechanic_hint_can_reorder_probe_queue() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[1:3, 1:3] = 5
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    feature = extract_components(frame)[0].feature
    for _ in range(3):
        memory.click_stat_for(feature).update(4.0, changed=True, level_gain=True)
    level_memory = LevelMemory(step_index=7, last_changed_step=2)
    level_memory.probe_queue.extend(
        [
            ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name),
            ActionCandidate(
                GameAction.ACTION6,
                {"x": 1, "y": 1},
                (GameAction.ACTION6.name, feature),
                "ACTION6@1,1",
            ),
        ]
    )
    signature = frame_signature(frame)

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION6), memory, level_memory)

    assert chosen.action == GameAction.ACTION6
    assert level_memory.mechanic_hint is not None
    assert agent.diagnostics.symbolic_hint_refreshes == 1


def test_symbolic_mechanic_hint_refreshes_on_loop_pressure_even_without_long_stall() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[1:3, 1:3] = 5
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    feature = extract_components(frame)[0].feature
    for _ in range(3):
        memory.click_stat_for(feature).update(3.0, changed=True, level_gain=False)
    signature = frame_signature(frame)
    level_memory = LevelMemory(
        step_index=8,
        last_changed_step=7,
        mechanic_hint=MechanicHint(mode="CLICK", goal="CLEAR", focus="SMALL_OBJECT", confidence=0.5),
        mechanic_hint_step=6,
        mechanic_hint_pressure=0,
    )
    level_memory.family_no_progress_counts["CLICK"] = 4
    level_memory.mark_seen(signature)
    level_memory.mark_seen(signature)
    candidates = agent._candidates(frame, (GameAction.ACTION6, GameAction.ACTION7), memory)

    agent._maybe_refresh_mechanic_hint(frame, candidates, memory, level_memory)

    assert level_memory.mechanic_hint is not None
    assert level_memory.mechanic_hint.source_step == 8
    assert level_memory.mechanic_hint_pressure >= 5
    assert agent.diagnostics.symbolic_hint_refreshes == 1


def test_symbolic_mechanic_hint_can_shift_to_interact_when_contact_loops_build() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[2:4, 2:4] = 5
    frame[2:4, 4:6] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    memory.actor_features[actor_feature] = 3
    level_memory = LevelMemory(step_index=9, last_changed_step=4)
    level_memory.family_no_progress_counts["CLICK"] = 3
    level_memory.family_no_progress_counts["MOVE"] = 2
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION5, GameAction.ACTION6), memory)

    hint = agent._symbolic_mechanic_hint(frame, candidates, memory, level_memory)

    assert hint is not None
    assert hint.mode == "INTERACT"


def test_symbolic_mechanic_hint_can_shift_probe_choice_to_undo_under_loops() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[1:3, 1:3] = 5
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    feature = extract_components(frame)[0].feature
    for _ in range(2):
        memory.click_stat_for(feature).update(1.0, changed=True, level_gain=False)
    signature = frame_signature(frame)
    level_memory = LevelMemory(step_index=9, last_changed_step=7)
    level_memory.family_no_progress_counts["CLICK"] = 4
    level_memory.family_no_progress_counts["MOVE"] = 2
    level_memory.mark_seen(signature)
    level_memory.mark_seen(signature)
    level_memory.probe_queue.extend(
        [
            ActionCandidate(
                GameAction.ACTION6,
                {"x": 1, "y": 1},
                (GameAction.ACTION6.name, feature),
                "ACTION6@1,1",
            ),
            ActionCandidate(GameAction.ACTION7, None, (GameAction.ACTION7.name, None), GameAction.ACTION7.name),
        ]
    )

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION6, GameAction.ACTION7), memory, level_memory)

    assert level_memory.mechanic_hint is not None
    assert level_memory.mechanic_hint.mode == "UNDO"
    assert chosen.action == GameAction.ACTION7


def test_symbolic_mechanic_hint_can_shift_to_undo_from_repeated_changed_click_dead_ends() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[1:3, 1:3] = 5
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    feature = extract_components(frame)[0].feature
    memory.click_stat_for(feature).update(1.0, changed=True, level_gain=False)
    level_memory = LevelMemory(step_index=9, last_changed_step=7)
    level_memory.family_no_progress_counts["CLICK"] = 2
    level_memory.recent_events.extend(
        [
            "ACTION6 changed=1 level_gain=0 reward=1.00",
            "ACTION6 changed=1 level_gain=0 reward=1.00",
        ]
    )
    candidates = agent._candidates(frame, (GameAction.ACTION6, GameAction.ACTION7), memory)

    hint = agent._symbolic_mechanic_hint(frame, candidates, memory, level_memory)

    assert hint is not None
    assert hint.mode == "UNDO"


def test_posterior_click_planner_can_choose_undo_from_repeated_changed_click_dead_ends() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[1:3, 1:3] = 5
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    feature = extract_components(frame)[0].feature
    memory.click_stat_for(feature).update(1.0, changed=True, level_gain=False)
    level_memory = LevelMemory(
        step_index=9,
        last_changed_step=7,
        mechanic_hint=MechanicHint(
            mode="UNDO",
            goal="CLEAR",
            focus="SMALL_OBJECT",
            confidence=0.84,
            source="SYMBOLIC",
        ),
    )
    level_memory.family_no_progress_counts["CLICK"] = 2
    level_memory.recent_events.extend(
        [
            "ACTION6 changed=1 level_gain=0 reward=1.00",
            "ACTION6 changed=1 level_gain=0 reward=1.00",
        ]
    )
    candidates = agent._candidates(frame, (GameAction.ACTION6, GameAction.ACTION7), memory)

    chosen, score, _margin = agent._planned_click_decision(frame, candidates, memory, level_memory)

    assert chosen is not None
    assert chosen.action == GameAction.ACTION7
    assert score > 0.78


def test_posterior_click_planner_can_override_probe_queue_under_click_hint() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[1:3, 1:3] = 7
    frame[5:7, 5:7] = 3
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    for component in extract_components(frame):
        for _ in range(3):
            memory.click_stat_for(component.feature).update(4.0, changed=True, level_gain=True)
    level_memory = LevelMemory(
        step_index=9,
        last_changed_step=4,
        mechanic_hint=MechanicHint(
            mode="CLICK",
            goal="CLEAR",
            focus="RARE_COLOR",
            confidence=0.9,
            source="SYMBOLIC",
        ),
    )
    signature = frame_signature(frame)
    level_memory.mark_seen(signature)
    level_memory.mark_seen(signature)
    level_memory.probe_queue.append(
        ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name)
    )

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION6), memory, level_memory)

    assert chosen.action == GameAction.ACTION6
    assert int(chosen.data["y"]) in {1, 2}
    assert int(chosen.data["x"]) in {1, 2}
    assert agent.diagnostics.posterior_plan_choices == 1
    assert agent.diagnostics.posterior_click_plan_choices == 1


def test_posterior_click_planner_can_faststart_without_loop_pressure_under_strong_symbolic_hint() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[1:3, 1:3] = 7
    frame[5:7, 5:7] = 3
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    for component in extract_components(frame):
        for _ in range(3):
            memory.click_stat_for(component.feature).update(4.0, changed=True, level_gain=True)
    level_memory = LevelMemory(
        step_index=6,
        last_changed_step=5,
        mechanic_hint=MechanicHint(
            mode="CLICK",
            goal="CLEAR",
            focus="RARE_COLOR",
            confidence=0.92,
            source="SYMBOLIC",
        ),
    )
    signature = frame_signature(frame)
    level_memory.mark_seen(signature)
    level_memory.probe_queue.append(
        ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name)
    )

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION6), memory, level_memory)

    assert chosen.action == GameAction.ACTION6
    assert agent.diagnostics.posterior_plan_choices == 1
    assert agent.diagnostics.posterior_faststart_choices == 1
    assert agent.diagnostics.posterior_click_plan_choices == 1


def test_posterior_click_planner_can_detour_to_keyboard_when_control_is_partially_known() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[2:4, 2:4] = 5
    frame[2:4, 5:7] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    memory.record_actor_hint(actor_feature, weight=4)
    level_memory = LevelMemory(
        step_index=6,
        last_changed_step=5,
        keyboard_control_confidence=0.6,
        mechanic_hint=MechanicHint(
            mode="CLICK",
            goal="CLEAR",
            focus="RARE_COLOR",
            confidence=0.88,
            source="SYMBOLIC",
        ),
    )
    signature = frame_signature(frame)
    level_memory.mark_seen(signature)
    click_candidate = ActionCandidate(GameAction.ACTION6, {"x": 5, "y": 2}, (GameAction.ACTION6.name, (2, 5, 2, 5)), "ACTION6@2,5")
    keyboard_candidate = ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name)
    agent._planned_click_decision = lambda *_args, **_kwargs: (click_candidate, 1.0, 0.2)  # type: ignore[method-assign]
    agent._planned_rollout_decision = lambda *_args, **_kwargs: (keyboard_candidate, 0.92, 0.18)  # type: ignore[method-assign]

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION6), memory, level_memory)

    assert chosen.action == GameAction.ACTION1
    assert agent.diagnostics.late_click_detours == 1


def test_posterior_keyboard_planner_prefers_interact_when_hint_and_affordance_align() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[2:4, 2:4] = 5
    frame[2:4, 4:6] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.actor_features[actor_feature] = 4
    memory.record_motion("ACTION1", (0, 1), actor_feature)
    for _ in range(2):
        memory.stat_for((GameAction.ACTION5.name, None)).update(4.0, changed=True, level_gain=True)
    for _ in range(2):
        memory.target_affordance_stat_for(target_feature, GameAction.ACTION5.name).update(
            4.0,
            changed=True,
            level_gain=True,
        )
    level_memory = LevelMemory(
        step_index=8,
        last_changed_step=2,
        mechanic_hint=MechanicHint(
            mode="INTERACT",
            goal="TOGGLE",
            focus="MOVING_OBJECT",
            confidence=0.9,
            source="SYMBOLIC",
        ),
    )
    signature = frame_signature(frame)
    level_memory.mark_seen(signature)
    level_memory.mark_seen(signature)
    level_memory.family_no_progress_counts["MOVE"] = 3

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION5), memory, level_memory)

    assert chosen.action == GameAction.ACTION5
    assert agent.diagnostics.posterior_plan_choices == 1
    assert agent.diagnostics.posterior_keyboard_plan_choices == 1


def test_posterior_keyboard_planner_can_faststart_without_loop_pressure_under_strong_symbolic_hint() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[2:4, 2:4] = 5
    frame[2:4, 5:7] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.actor_features[actor_feature] = 4
    memory.record_motion("ACTION1", (0, 1), actor_feature)
    for _ in range(3):
        memory.interaction_target_stat_for(target_feature).update(4.0, changed=True, level_gain=True)
    level_memory = LevelMemory(
        step_index=6,
        last_changed_step=5,
        mechanic_hint=MechanicHint(
            mode="MOVE",
            goal="CONTACT",
            focus="MOVING_OBJECT",
            confidence=0.9,
            source="SYMBOLIC",
        ),
    )
    signature = frame_signature(frame)
    level_memory.mark_seen(signature)
    level_memory.probe_queue.append(
        ActionCandidate(GameAction.ACTION6, {"x": 0, "y": 0}, (GameAction.ACTION6.name, target_feature), "ACTION6@0,0")
    )

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION6), memory, level_memory)

    assert chosen.action == GameAction.ACTION1
    assert agent.diagnostics.posterior_plan_choices == 1
    assert agent.diagnostics.posterior_faststart_choices == 1
    assert agent.diagnostics.posterior_keyboard_plan_choices == 1


def test_stalled_keyboard_posterior_can_yield_to_control_map() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[2:4, 2:4] = 5
    frame[2:4, 5:7] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    memory.actor_features[actor_feature] = 4
    level_memory = LevelMemory(
        step_index=8,
        last_changed_step=3,
        mechanic_hint=MechanicHint(
            mode="MOVE",
            goal="CONTACT",
            focus="MOVING_OBJECT",
            confidence=0.88,
            source="SYMBOLIC",
        ),
    )
    level_memory.family_no_progress_counts["MOVE"] = 4
    level_memory.recent_events.extend(
        [
            "ACTION1 changed=0 level_gain=0 reward=0.00",
            "ACTION2 changed=0 level_gain=0 reward=0.00",
        ]
    )
    signature = frame_signature(frame)
    level_memory.mark_seen(signature)
    level_memory.mark_seen(signature)
    posterior_candidate = ActionCandidate(
        GameAction.ACTION2,
        None,
        (GameAction.ACTION2.name, None),
        GameAction.ACTION2.name,
    )
    control_candidate = ActionCandidate(
        GameAction.ACTION1,
        None,
        (GameAction.ACTION1.name, None),
        GameAction.ACTION1.name,
    )
    agent._posterior_guided_decision = lambda *_args, **_kwargs: (posterior_candidate, 1.2, 0.2, "KEYBOARD")  # type: ignore[method-assign]
    agent._control_map_decision = lambda *_args, **_kwargs: (control_candidate, 1.15, 0.25)  # type: ignore[method-assign]
    agent._planned_rollout_decision = lambda *_args, **_kwargs: (None, 0.0, 0.0)  # type: ignore[method-assign]

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION2), memory, level_memory)

    assert chosen.action == GameAction.ACTION1
    assert agent.diagnostics.posterior_stall_suppressions == 1
    assert agent.diagnostics.control_map_plan_choices == 1
    assert agent.diagnostics.posterior_plan_choices == 0


def test_stalled_keyboard_posterior_prefers_control_map_before_rollout() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[2:4, 2:4] = 5
    frame[2:4, 5:7] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    memory.actor_features[actor_feature] = 4
    level_memory = LevelMemory(
        step_index=8,
        last_changed_step=3,
        mechanic_hint=MechanicHint(
            mode="MOVE",
            goal="CONTACT",
            focus="MOVING_OBJECT",
            confidence=0.88,
            source="SYMBOLIC",
        ),
    )
    target_cell = coarse_cell_for_coord(next(component.anchor for component in components if component.color == 7))
    level_memory.recent_target_cells[target_cell] = 3
    level_memory.family_no_progress_counts["MOVE"] = 4
    level_memory.recent_events.extend(
        [
            "ACTION1 changed=0 level_gain=0 reward=0.00",
            "ACTION2 changed=0 level_gain=0 reward=0.00",
        ]
    )
    signature = frame_signature(frame)
    level_memory.mark_seen(signature)
    level_memory.mark_seen(signature)
    posterior_candidate = ActionCandidate(GameAction.ACTION2, None, (GameAction.ACTION2.name, None), GameAction.ACTION2.name)
    control_candidate = ActionCandidate(GameAction.ACTION1, None, (GameAction.ACTION1.name, None), GameAction.ACTION1.name)
    rollout_candidate = ActionCandidate(GameAction.ACTION3, None, (GameAction.ACTION3.name, None), GameAction.ACTION3.name)
    agent._posterior_guided_decision = lambda *_args, **_kwargs: (posterior_candidate, 1.2, 0.2, "KEYBOARD")  # type: ignore[method-assign]
    agent._control_map_decision = lambda *_args, **_kwargs: (control_candidate, 1.15, 0.25)  # type: ignore[method-assign]
    agent._planned_rollout_decision = lambda *_args, **_kwargs: (rollout_candidate, 1.1, 0.2)  # type: ignore[method-assign]

    chosen = agent._choose_action(
        frame,
        signature,
        (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3),
        memory,
        level_memory,
    )

    assert chosen.action == GameAction.ACTION1
    assert agent.diagnostics.posterior_stall_suppressions == 1
    assert agent.diagnostics.control_map_plan_choices == 1
    assert agent.diagnostics.rollout_plan_choices == 0


def test_posterior_keyboard_plan_can_seed_macro_bundle() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[2:4, 2:4] = 5
    frame[2:4, 5:7] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.actor_features[actor_feature] = 4
    memory.record_motion("ACTION1", (0, 1), actor_feature)
    for _ in range(3):
        memory.interaction_target_stat_for(target_feature).update(4.0, changed=True, level_gain=True)
    level_memory = LevelMemory(
        step_index=6,
        last_changed_step=5,
        mechanic_hint=MechanicHint(
            mode="MOVE",
            goal="CONTACT",
            focus="MOVING_OBJECT",
            confidence=0.9,
            source="SYMBOLIC",
        ),
    )
    signature = frame_signature(frame)
    level_memory.mark_seen(signature)

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION6), memory, level_memory)

    assert chosen.action == GameAction.ACTION1
    assert agent.diagnostics.macro_bundle_injections == 1
    assert list(level_memory.macro_queue)[0].action == GameAction.ACTION1
    assert level_memory.macro_source_step == level_memory.step_index


def test_macro_bundle_can_drive_followup_action_on_next_state() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[2:4, 2:4] = 5
    frame[2:4, 5:7] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.actor_features[actor_feature] = 4
    memory.record_motion("ACTION1", (0, 1), actor_feature)
    for _ in range(3):
        memory.interaction_target_stat_for(target_feature).update(4.0, changed=True, level_gain=True)
    level_memory = LevelMemory(
        step_index=6,
        last_changed_step=5,
        mechanic_hint=MechanicHint(
            mode="MOVE",
            goal="CONTACT",
            focus="MOVING_OBJECT",
            confidence=0.9,
            source="SYMBOLIC",
        ),
    )
    signature = frame_signature(frame)
    level_memory.mark_seen(signature)
    first = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION6), memory, level_memory)
    assert first.action == GameAction.ACTION1
    level_memory.mark_tried(signature, first.key)

    next_frame = frame.copy()
    next_frame[0, 0] = 9
    next_signature = frame_signature(next_frame)
    level_memory.step_index = 7

    second = agent._choose_action(next_frame, next_signature, (GameAction.ACTION1, GameAction.ACTION6), memory, level_memory)

    assert second.action == GameAction.ACTION1
    assert agent.diagnostics.macro_actions_used == 1
    assert not level_memory.macro_queue


def test_posterior_mixed_planner_can_switch_to_keyboard_after_productive_click_change() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[2:4, 2:4] = 5
    frame[2:4, 5:7] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.actor_features[actor_feature] = 4
    memory.record_motion("ACTION1", (0, 1), actor_feature)
    for _ in range(2):
        memory.click_stat_for(target_feature).update(3.0, changed=True, level_gain=False)
        memory.interaction_target_stat_for(target_feature).update(3.0, changed=True, level_gain=False)
        memory.stat_for((GameAction.ACTION1.name, None)).update(3.0, changed=True, level_gain=False)
    level_memory = LevelMemory(
        step_index=8,
        last_changed_step=2,
        mechanic_hint=MechanicHint(
            mode="MIXED",
            goal="CONTACT",
            focus="MOVING_OBJECT",
            confidence=0.88,
            source="SYMBOLIC",
        ),
    )
    signature = frame_signature(frame)
    level_memory.mark_seen(signature)
    level_memory.mark_seen(signature)
    level_memory.recent_events.append("ACTION6 changed=1 level_gain=0 reward=1.00")

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION6), memory, level_memory)

    assert chosen.action == GameAction.ACTION1
    assert agent.diagnostics.posterior_plan_choices == 1
    assert agent.diagnostics.posterior_keyboard_plan_choices == 1


def test_posterior_mixed_planner_can_switch_back_to_click_after_keyboard_stall() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[2:4, 2:4] = 5
    frame[2:4, 5:7] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.actor_features[actor_feature] = 4
    memory.record_motion("ACTION1", (0, 1), actor_feature)
    for _ in range(3):
        memory.click_stat_for(target_feature).update(4.0, changed=True, level_gain=True)
    level_memory = LevelMemory(
        step_index=8,
        last_changed_step=2,
        mechanic_hint=MechanicHint(
            mode="MIXED",
            goal="CLEAR",
            focus="RARE_COLOR",
            confidence=0.88,
            source="SYMBOLIC",
        ),
    )
    signature = frame_signature(frame)
    level_memory.mark_seen(signature)
    level_memory.mark_seen(signature)
    level_memory.family_no_progress_counts["MOVE"] = 3
    level_memory.recent_events.append("ACTION1 changed=0 level_gain=0 reward=0.00")

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION6), memory, level_memory)

    assert chosen.action == GameAction.ACTION6
    assert agent.diagnostics.posterior_plan_choices == 1
    assert agent.diagnostics.posterior_click_plan_choices == 1


def test_wrong_mechanic_hint_does_not_beat_strong_empirical_keyboard_signal() -> None:
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[1:3, 1:3] = 5
    advisor = FakeMechanicAdvisor(
        {"mode": "CLICK", "goal": "CLEAR", "focus": "SMALL_OBJECT", "confidence": 0.9}
    )
    agent = ArcAgi3TACTICPublicAgent(mechanic_advisor=advisor, mechanic_advisor_budget_per_level=1)
    memory = EnvironmentMemory()
    for _ in range(3):
        memory.stat_for((GameAction.ACTION1.name, None)).update(4.0, changed=True, level_gain=True)
    level_memory = LevelMemory(step_index=9, last_changed_step=1)
    level_memory.recent_events.append("ACTION1 changed=1 level_gain=1 reward=4.00")
    signature = frame_signature(frame)

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION6), memory, level_memory)

    assert chosen.action == GameAction.ACTION1


def test_advisor_action_can_override_stalled_keyboard_choice() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[8:12, 8:12] = 5
    frame[8:12, 16:20] = 7
    advisor = FakeAdvisor(GameAction.ACTION2.name)
    agent = ArcAgi3TACTICPublicAgent(action_advisor=advisor, action_advisor_budget_per_level=1)
    memory = EnvironmentMemory()
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.record_motion("ACTION1", (0, 1), actor_feature)
    memory.record_motion("ACTION2", (1, 0), actor_feature)
    memory.interaction_target_stat_for(target_feature).update(2.0, changed=True, level_gain=False)
    level_memory = LevelMemory()
    level_memory.step_index = 6
    level_memory.last_changed_step = 1
    level_memory.recent_events.extend(
        [
            "ACTION1 changed=0 level_gain=0 reward=0.00",
            "ACTION4 changed=0 level_gain=0 reward=0.00",
        ]
    )
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION2), memory)

    chosen = agent._advisor_action(frame, candidates, memory, level_memory)
    assert chosen.action == GameAction.ACTION2
    assert agent.diagnostics.qwen_calls == 1
    assert agent.diagnostics.qwen_overrides == 0
    assert advisor.prompts


def test_mode_advisor_can_choose_click_family() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[8:12, 8:12] = 5
    frame[24:28, 24:28] = 7
    advisor = FakeAdvisor("CLICK")
    agent = ArcAgi3TACTICPublicAgent(action_advisor=advisor, action_advisor_budget_per_level=1)
    memory = EnvironmentMemory()
    memory.actor_features[(5, 4, 0, 0)] = 2
    level_memory = LevelMemory(step_index=6, last_changed_step=2)
    signature = frame_signature(frame)

    chosen = agent._choose_action(
        frame,
        signature,
        (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION6),
        memory,
        level_memory,
    )

    assert chosen.action == GameAction.ACTION6
    assert agent.diagnostics.qwen_mode_calls == 1
    assert agent.diagnostics.qwen_mode_overrides == 1
    assert advisor.prompts


def test_advisor_prompt_includes_interaction_targets_and_action_consequences() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[20:22, 20:22] = 5
    frame[20:22, 34:36] = 7
    advisor = FakeAdvisor(GameAction.ACTION2.name)
    agent = ArcAgi3TACTICPublicAgent(action_advisor=advisor, action_advisor_budget_per_level=1)
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=6, last_changed_step=1)
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.record_motion("ACTION1", (0, 1), actor_feature)
    memory.record_motion("ACTION2", (1, 0), actor_feature)
    memory.interaction_target_stat_for(target_feature).update(3.0, changed=True, level_gain=True)
    candidates = agent._candidates(frame, (GameAction.ACTION1, GameAction.ACTION2), memory)

    chosen = agent._advisor_action(frame, candidates, memory, level_memory)

    assert chosen is not None
    assert chosen.action == GameAction.ACTION2
    assert advisor.prompts
    assert "Highest-priority interaction targets:" in advisor.prompts[0]
    assert "Predicted action consequences:" in advisor.prompts[0]


def test_choose_action_uses_rollout_before_advisor() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[20:22, 20:22] = 5
    frame[20:22, 34:36] = 7
    advisor = FakeAdvisor(GameAction.ACTION2.name)
    agent = ArcAgi3TACTICPublicAgent(action_advisor=advisor, action_advisor_budget_per_level=1)
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=6, last_changed_step=1)
    signature = frame_signature(frame)
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.record_motion("ACTION1", (0, 1), actor_feature)
    memory.record_motion("ACTION2", (1, 0), actor_feature)
    memory.interaction_target_stat_for(target_feature).update(3.0, changed=True, level_gain=True)

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION2), memory, level_memory)

    assert chosen.action == GameAction.ACTION1
    assert agent.diagnostics.rollout_plan_choices == 1
    assert agent.diagnostics.qwen_calls == 0


def test_planned_rollout_action_can_select_move_then_action5_sequence() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[20:22, 20:22] = 5
    frame[20:22, 24:26] = 7
    agent = ArcAgi3TACTICPublicAgent()
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=5)
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.record_motion("ACTION1", (0, 1), actor_feature)
    memory.interaction_target_stat_for(target_feature).update(4.0, changed=True, level_gain=True)
    memory.target_affordance_stat_for(target_feature, GameAction.ACTION5.name).update(
        8.0,
        changed=True,
        level_gain=True,
    )

    planned = agent._planned_rollout_action(frame, (GameAction.ACTION1, GameAction.ACTION5), memory, level_memory)
    assert planned is not None
    assert planned.action == GameAction.ACTION1


def test_choose_action_uses_advisor_when_rollout_is_low_confidence() -> None:
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[20:22, 20:22] = 5
    frame[24:26, 24:26] = 7
    advisor = FakeAdvisor(GameAction.ACTION2.name)
    agent = ArcAgi3TACTICPublicAgent(action_advisor=advisor, action_advisor_budget_per_level=1)
    memory = EnvironmentMemory()
    level_memory = LevelMemory(step_index=6, last_changed_step=1)
    signature = frame_signature(frame)
    components = extract_components(frame)
    actor_feature = next(component.feature for component in components if component.color == 5)
    target_feature = next(component.feature for component in components if component.color == 7)
    memory.record_motion("ACTION1", (0, 1), actor_feature)
    memory.record_motion("ACTION2", (1, 0), actor_feature)
    memory.interaction_target_stat_for(target_feature).update(2.0, changed=True, level_gain=False)

    chosen = agent._choose_action(frame, signature, (GameAction.ACTION1, GameAction.ACTION2), memory, level_memory)

    assert chosen.action == GameAction.ACTION2
    assert agent.diagnostics.qwen_calls == 1
    assert agent.diagnostics.qwen_overrides == 1
    assert advisor.prompts
    assert "Current heuristic rollout suggestion:" in advisor.prompts[0]


def test_policy_tuning_from_dict_accepts_known_keys() -> None:
    tuning = PolicyTuning.from_dict(
        {
            "faststart_click_score": 1.5,
            "control_map_accept_score": 1.2,
            "interaction_target_value_weight": 0.9,
        }
    )

    assert tuning.faststart_click_score == 1.5
    assert tuning.control_map_accept_score == 1.2
    assert tuning.interaction_target_value_weight == 0.9


def test_policy_tuning_from_dict_rejects_unknown_keys() -> None:
    try:
        PolicyTuning.from_dict({"not_a_real_threshold": 1.0})
    except ValueError as exc:
        assert "Unknown policy tuning keys" in str(exc)
    else:
        raise AssertionError("Expected PolicyTuning.from_dict to reject unknown keys")


def test_agent_accepts_custom_policy_tuning() -> None:
    tuning = PolicyTuning(
        posterior_hint_min_confidence=0.9,
        control_map_accept_score=1.25,
        interaction_target_reward_weight=0.5,
    )
    agent = ArcAgi3TACTICPublicAgent(tuning=tuning)

    assert agent.tuning.posterior_hint_min_confidence == 0.9
    assert agent.tuning.control_map_accept_score == 1.25
    assert agent.tuning.interaction_target_reward_weight == 0.5
