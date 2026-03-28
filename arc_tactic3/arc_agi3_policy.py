from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import numpy as np
from arcengine import GameAction, GameState as ArcGameState

from .progress import Spinner
from .qwen_arc_advisor import OllamaArcAdvisor


Coord = tuple[int, int]
CoarseCell = tuple[int, int]
ActionKey = tuple[str, tuple[int, ...] | None]
ClickFeature = tuple[int, int, int, int]
MechanicMode = str
MechanicGoal = str
MechanicFocus = str
AbstractStateKey = tuple[str, str, int, int, int, int, int]
InteractionStateKey = tuple[int, int, int, int, int, int, int]

MECHANIC_MODES: tuple[MechanicMode, ...] = ("CLICK", "MOVE", "INTERACT", "UNDO", "MIXED", "UNKNOWN")
MECHANIC_GOALS: tuple[MechanicGoal, ...] = ("CONTACT", "ALIGN", "COLLECT", "TOGGLE", "CLEAR", "UNKNOWN")
MECHANIC_FOCUSES: tuple[MechanicFocus, ...] = (
    "RARE_COLOR",
    "LARGE_OBJECT",
    "SMALL_OBJECT",
    "HOTSPOT",
    "CENTER",
    "MOVING_OBJECT",
    "UNKNOWN",
)


@dataclass(frozen=True, slots=True)
class PolicyTuning:
    posterior_hint_min_confidence: float = 0.42
    faststart_click_score: float = 1.18
    faststart_keyboard_score: float = 1.12
    faststart_keyboard_margin: float = 0.16
    posterior_click_override_score: float = 1.0
    posterior_click_override_margin: float = 0.08
    posterior_pressure_score: float = 1.05
    posterior_relaxed_score: float = 0.85
    posterior_relaxed_margin: float = 0.05
    posterior_keyboard_stall_failure_count: int = 3
    posterior_keyboard_stall_event_count: int = 2
    contact_rollout_score: float = 0.9
    contact_rollout_margin: float = 0.05
    control_map_accept_score: float = 1.0
    control_map_soft_score: float = 0.82
    control_map_soft_margin: float = 0.12
    control_map_delta_confidence_min: float = 0.4
    control_map_candidate_confidence_min: float = 0.62
    control_map_context_confidence_min: float = 0.45
    control_map_score_floor: float = 0.9
    control_map_margin_floor: float = 0.18
    interaction_graph_accept_score: float = 1.08
    interaction_graph_soft_score: float = 0.9
    interaction_graph_soft_margin: float = 0.2
    interaction_target_base: float = 0.15
    interaction_target_reward_weight: float = 0.35
    interaction_target_change_weight: float = 0.75
    interaction_target_level_gain_weight: float = 1.1
    interaction_target_subgoal_weight: float = 0.45
    interaction_target_value_weight: float = 0.65
    interaction_target_blocked_penalty: float = 0.6
    interaction_target_active_contact_bonus: float = 1.15
    interaction_target_contact_cell_bonus: float = 0.95
    target_macro_accept_score: float = 1.02
    target_macro_margin_floor: float = 0.08
    target_adjacency_bridge_floor: float = 0.22
    target_option_min_score: float = 1.18
    target_option_margin_floor: float = 0.1
    target_option_action_confidence_min: float = 0.38
    target_option_total_actions: int = 3

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PolicyTuning":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown policy tuning keys: {', '.join(unknown)}")
        return cls(**{key: value for key, value in payload.items() if key in allowed})


def key_stability_score(key: ActionKey) -> float:
    name = key[0]
    return (sum(ord(char) for char in name) % 100) / 1000.0


def action_family(action: GameAction) -> MechanicMode:
    if action == GameAction.ACTION6:
        return "CLICK"
    if action == GameAction.ACTION5:
        return "INTERACT"
    if action == GameAction.ACTION7:
        return "UNDO"
    return "MOVE"


@dataclass(frozen=True, slots=True)
class FrameComponent:
    color: int
    size: int
    anchor: Coord
    feature: ClickFeature
    bounds: tuple[int, int, int, int]


@dataclass(slots=True)
class ActionStats:
    attempts: int = 0
    total_reward: float = 0.0
    changed_count: int = 0
    level_gain_count: int = 0

    def update(self, reward: float, *, changed: bool, level_gain: bool) -> None:
        self.attempts += 1
        self.total_reward += reward
        self.changed_count += int(changed)
        self.level_gain_count += int(level_gain)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.attempts if self.attempts else 0.0

    @property
    def change_rate(self) -> float:
        return self.changed_count / self.attempts if self.attempts else 0.0

    @property
    def level_gain_rate(self) -> float:
        return self.level_gain_count / self.attempts if self.attempts else 0.0


@dataclass(frozen=True, slots=True)
class MechanicHint:
    mode: MechanicMode = "UNKNOWN"
    goal: MechanicGoal = "UNKNOWN"
    focus: MechanicFocus = "UNKNOWN"
    confidence: float = 0.0
    source_step: int = 0
    source: str = "UNKNOWN"
    raw_text: str = ""


@dataclass(slots=True)
class EnvironmentMemory:
    action_stats: dict[ActionKey, ActionStats] = field(default_factory=dict)
    click_stats: dict[ClickFeature, ActionStats] = field(default_factory=dict)
    interaction_target_stats: dict[ClickFeature, ActionStats] = field(default_factory=dict)
    target_affordance_stats: dict[tuple[ClickFeature, str], ActionStats] = field(default_factory=dict)
    keyboard_context_stats: dict[tuple[str, CoarseCell], ActionStats] = field(default_factory=dict)
    action_motion: dict[str, dict[Coord, int]] = field(default_factory=dict)
    actor_features: dict[ClickFeature, int] = field(default_factory=dict)
    actor_cell_scores: dict[CoarseCell, float] = field(default_factory=dict)
    effect_signatures: dict[ActionKey, set[bytes]] = field(default_factory=dict)
    pixel_change_counts: np.ndarray = field(default_factory=lambda: np.zeros((64, 64), dtype=np.int16))
    transition_count: int = 0

    def stat_for(self, key: ActionKey) -> ActionStats:
        return self.action_stats.setdefault(key, ActionStats())

    def click_stat_for(self, feature: ClickFeature) -> ActionStats:
        return self.click_stats.setdefault(feature, ActionStats())

    def interaction_target_stat_for(self, feature: ClickFeature) -> ActionStats:
        return self.interaction_target_stats.setdefault(feature, ActionStats())

    def target_affordance_stat_for(self, feature: ClickFeature, action_name: str) -> ActionStats:
        return self.target_affordance_stats.setdefault((feature, action_name), ActionStats())

    def target_affordance_score(self, feature: ClickFeature, action_name: str) -> float:
        stats = self.target_affordance_stats.get((feature, action_name))
        if stats is None:
            return 0.0
        return (
            0.25 * max(stats.mean_reward, 0.0)
            + 0.8 * stats.change_rate
            + 1.2 * stats.level_gain_rate
        )

    def keyboard_context_stat_for(self, action_name: str, actor_cell: CoarseCell) -> ActionStats:
        return self.keyboard_context_stats.setdefault((action_name, actor_cell), ActionStats())

    def record_motion(self, action_name: str, delta: Coord, feature: ClickFeature) -> None:
        self.action_motion.setdefault(action_name, {})
        self.action_motion[action_name][delta] = self.action_motion[action_name].get(delta, 0) + 1
        self.actor_features[feature] = self.actor_features.get(feature, 0) + 1

    def record_actor_hint(self, feature: ClickFeature, *, weight: int = 1) -> None:
        self.actor_features[feature] = self.actor_features.get(feature, 0) + weight

    def record_actor_cell_hint(self, cell: CoarseCell, *, weight: float = 1.0) -> None:
        self.actor_cell_scores[cell] = self.actor_cell_scores.get(cell, 0.0) + weight

    def best_delta(self, action_name: str) -> tuple[Coord | None, float]:
        counts = self.action_motion.get(action_name, {})
        if not counts:
            return None, 0.0
        delta, count = max(counts.items(), key=lambda item: item[1])
        total = sum(counts.values())
        return delta, count / max(total, 1)

    def record_transition_mask(self, delta_mask: np.ndarray) -> None:
        self.transition_count += 1
        row_limit = min(self.pixel_change_counts.shape[0], delta_mask.shape[0])
        col_limit = min(self.pixel_change_counts.shape[1], delta_mask.shape[1])
        self.pixel_change_counts[:row_limit, :col_limit] += delta_mask[:row_limit, :col_limit].astype(np.int16)

    def noise_mask(self) -> np.ndarray:
        if self.transition_count < 4:
            return np.zeros_like(self.pixel_change_counts, dtype=bool)
        threshold = max(3, int(self.transition_count * 0.7))
        return self.pixel_change_counts >= threshold

    def record_effect(self, key: ActionKey, delta_mask: np.ndarray) -> None:
        self.effect_signatures.setdefault(key, set()).add(delta_signature(delta_mask))

    def effect_diversity(self, key: ActionKey) -> float:
        stats = self.action_stats.get(key)
        if stats is None or stats.attempts == 0:
            return 1.0
        return len(self.effect_signatures.get(key, set())) / stats.attempts


@dataclass(slots=True)
class LevelMemory:
    seen_states: set[bytes] = field(default_factory=set)
    state_visit_counts: dict[bytes, int] = field(default_factory=dict)
    abstract_state_keys: dict[bytes, AbstractStateKey] = field(default_factory=dict)
    interaction_state_keys: dict[bytes, InteractionStateKey] = field(default_factory=dict)
    tried_from_state: dict[bytes, set[ActionKey]] = field(default_factory=dict)
    available_from_state: dict[bytes, tuple[ActionKey, ...]] = field(default_factory=dict)
    transitions: dict[tuple[bytes, ActionKey], bytes] = field(default_factory=dict)
    state_actor_anchors: dict[bytes, Coord] = field(default_factory=dict)
    last_changed_step: int = 0
    step_index: int = 0
    probe_queue: deque[ActionCandidate] = field(default_factory=deque)
    frontier_plan: deque[ActionKey] = field(default_factory=deque)
    probed_keys: set[ActionKey] = field(default_factory=set)
    actor_positions: set[Coord] = field(default_factory=set)
    subgoal_cells: dict[CoarseCell, float] = field(default_factory=dict)
    keyboard_no_change_streaks: dict[str, int] = field(default_factory=dict)
    keyboard_repeat_key: ActionKey | None = None
    keyboard_repeat_steps: int = 0
    keyboard_plateau_steps: int = 0
    keyboard_control_confidence: float = 0.0
    qwen_calls_used: int = 0
    mechanic_qwen_calls_used: int = 0
    recent_events: deque[str] = field(default_factory=lambda: deque(maxlen=8))
    recent_target_contacts: dict[ClickFeature, int] = field(default_factory=dict)
    recent_target_cells: dict[CoarseCell, int] = field(default_factory=dict)
    pending_option_plan: OptionBundlePlan | None = None
    target_value_scores: dict[ClickFeature, float] = field(default_factory=dict)
    target_value_cells: dict[CoarseCell, float] = field(default_factory=dict)
    blocked_target_cells: dict[CoarseCell, float] = field(default_factory=dict)
    family_no_progress_counts: dict[MechanicMode, int] = field(default_factory=dict)
    macro_queue: deque[ActionCandidate] = field(default_factory=deque)
    macro_source_step: int = -1
    control_commit_primary_action: str | None = None
    control_commit_allow_interact: bool = False
    control_commit_target: Coord | None = None
    control_commit_last_distance: int | None = None
    control_commit_steps_remaining: int = 0
    mechanic_hint: MechanicHint | None = None
    mechanic_hint_step: int = -1
    mechanic_hint_pressure: int = 0
    mechanic_hint_raw: str = ""
    interaction_graph_action_stats: dict[tuple[InteractionStateKey, str], ActionStats] = field(default_factory=dict)
    interaction_graph_family_stats: dict[tuple[InteractionStateKey, MechanicMode], ActionStats] = field(default_factory=dict)
    interaction_graph_transitions: dict[tuple[InteractionStateKey, str, InteractionStateKey], int] = field(
        default_factory=dict
    )

    def mark_seen(self, signature: bytes) -> None:
        self.seen_states.add(signature)
        self.state_visit_counts[signature] = self.state_visit_counts.get(signature, 0) + 1

    def mark_tried(self, state_signature: bytes, action_key: ActionKey) -> None:
        self.tried_from_state.setdefault(state_signature, set()).add(action_key)

    def mark_available(self, state_signature: bytes, action_keys: tuple[ActionKey, ...]) -> None:
        self.available_from_state[state_signature] = action_keys

    def has_tried(self, state_signature: bytes, action_key: ActionKey) -> bool:
        return action_key in self.tried_from_state.get(state_signature, set())

    def observe_transition(self, state_signature: bytes, action_key: ActionKey, next_signature: bytes) -> None:
        self.transitions[(state_signature, action_key)] = next_signature

    def note_actor_anchor(self, state_signature: bytes, anchor: Coord) -> None:
        self.state_actor_anchors[state_signature] = anchor

    def note_abstract_state(self, state_signature: bytes, abstract_key: AbstractStateKey) -> None:
        self.abstract_state_keys[state_signature] = abstract_key

    def note_interaction_state(self, state_signature: bytes, interaction_key: InteractionStateKey) -> None:
        self.interaction_state_keys[state_signature] = interaction_key

    def abstract_revisit_count(self, abstract_key: AbstractStateKey) -> int:
        return max(0, sum(1 for key in self.abstract_state_keys.values() if key == abstract_key) - 1)

    def interaction_revisit_count(self, interaction_key: InteractionStateKey) -> int:
        return max(0, sum(1 for key in self.interaction_state_keys.values() if key == interaction_key) - 1)

    def untried_keys(self, state_signature: bytes) -> tuple[ActionKey, ...]:
        available = self.available_from_state.get(state_signature, ())
        tried = self.tried_from_state.get(state_signature, set())
        return tuple(key for key in available if key not in tried)

    def record_subgoal_cells(self, cells: tuple[CoarseCell, ...], weight: float) -> None:
        if weight <= 0.0:
            return
        touched = set(cells)
        for cell in list(self.subgoal_cells):
            if cell not in touched:
                self.subgoal_cells[cell] *= 0.9
                if self.subgoal_cells[cell] < 0.05:
                    del self.subgoal_cells[cell]
        for cell in touched:
            updated = self.subgoal_cells.get(cell, 0.0) * 0.7 + weight
            self.subgoal_cells[cell] = min(updated, 4.0)

    def subgoal_bonus(self, cell: CoarseCell) -> float:
        bonus = self.subgoal_cells.get(cell, 0.0)
        row, col = cell
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (row + delta_row, col + delta_col)
            bonus = max(bonus, 0.35 * self.subgoal_cells.get(neighbor, 0.0))
        return bonus

    def note_keyboard_outcome(self, action_name: str, *, changed: bool) -> int:
        if changed:
            self.keyboard_no_change_streaks[action_name] = 0
            return 0
        streak = self.keyboard_no_change_streaks.get(action_name, 0) + 1
        self.keyboard_no_change_streaks[action_name] = streak
        return streak

    def active_target_contacts(self, *, ttl_steps: int = 3) -> tuple[ClickFeature, ...]:
        return tuple(
            feature
            for feature, last_step in self.recent_target_contacts.items()
            if self.step_index - last_step <= ttl_steps
        )

    def active_target_cells(self, *, ttl_steps: int = 5) -> tuple[CoarseCell, ...]:
        return tuple(
            cell
            for cell, last_step in self.recent_target_cells.items()
            if self.step_index - last_step <= ttl_steps
        )

    def note_target_contacts(self, features: tuple[ClickFeature, ...]) -> None:
        for feature in features:
            self.recent_target_contacts[feature] = self.step_index
            for cell in feature_region_cells(feature):
                self.recent_target_cells[cell] = self.step_index

    def note_target_values(
        self,
        features: tuple[ClickFeature, ...],
        *,
        reward: float,
        changed: bool,
        level_gain: bool,
    ) -> None:
        if not features:
            return
        base = 0.12 * max(reward, 0.0)
        if changed:
            base += 0.28
        if level_gain:
            base += 0.95
        if base <= 0.0:
            return

        touched_features = set(features)
        touched_cells: set[CoarseCell] = set()
        for feature in touched_features:
            updated = self.target_value_scores.get(feature, 0.0) * 0.65 + base
            self.target_value_scores[feature] = min(updated, 4.0)
            for cell in feature_region_cells(feature):
                touched_cells.add(cell)
                cell_updated = self.target_value_cells.get(cell, 0.0) * 0.65 + base
                self.target_value_cells[cell] = min(cell_updated, 4.0)

        for feature in list(self.target_value_scores):
            if feature not in touched_features:
                self.target_value_scores[feature] *= 0.98
                if self.target_value_scores[feature] < 0.05:
                    del self.target_value_scores[feature]
        for cell in list(self.target_value_cells):
            if cell not in touched_cells:
                self.target_value_cells[cell] *= 0.98
                if self.target_value_cells[cell] < 0.05:
                    del self.target_value_cells[cell]

    def target_cell_value_bonus(self, cell: CoarseCell) -> float:
        bonus = self.target_value_cells.get(cell, 0.0)
        row, col = cell
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (row + delta_row, col + delta_col)
            bonus = max(bonus, 0.55 * self.target_value_cells.get(neighbor, 0.0))
        return bonus

    def target_value_bonus(self, feature: ClickFeature, cell: CoarseCell) -> float:
        return max(self.target_value_scores.get(feature, 0.0), self.target_cell_value_bonus(cell))

    def note_blocked_target(self, target: Coord, *, weight: float = 1.0) -> None:
        cell = coarse_cell_for_coord(target)
        self.blocked_target_cells[cell] = min(self.blocked_target_cells.get(cell, 0.0) + weight, 4.0)

    def blocked_target_penalty(self, cell: CoarseCell) -> float:
        penalty = self.blocked_target_cells.get(cell, 0.0)
        row, col = cell
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (row + delta_row, col + delta_col)
            penalty = max(penalty, 0.45 * self.blocked_target_cells.get(neighbor, 0.0))
        return penalty

    def record_family_outcome(
        self,
        family: MechanicMode,
        *,
        changed: bool,
        level_gain: bool,
        repeated_state: bool,
    ) -> None:
        if level_gain:
            self.family_no_progress_counts[family] = 0
            return
        if changed and not repeated_state:
            self.family_no_progress_counts[family] = max(0, self.family_no_progress_counts.get(family, 0) - 1)
            return
        penalty = 2 if repeated_state and not changed else 1
        self.family_no_progress_counts[family] = min(8, self.family_no_progress_counts.get(family, 0) + penalty)

    def family_failure_count(self, family: MechanicMode) -> int:
        return self.family_no_progress_counts.get(family, 0)

    def loop_pressure(self, signature: bytes | None = None) -> int:
        revisit = 0
        if signature is not None:
            revisit = max(0, self.state_visit_counts.get(signature, 0) - 1)
        global_revisit = max((count - 1 for count in self.state_visit_counts.values()), default=0)
        return sum(self.family_no_progress_counts.values()) + max(revisit, global_revisit)

    def clear_control_commit(self) -> None:
        self.control_commit_primary_action = None
        self.control_commit_allow_interact = False
        self.control_commit_target = None
        self.control_commit_last_distance = None
        self.control_commit_steps_remaining = 0

    def start_control_commit(
        self,
        *,
        primary_action: str,
        target: Coord,
        last_distance: int,
        steps_remaining: int,
        allow_interact: bool,
    ) -> None:
        self.control_commit_primary_action = primary_action
        self.control_commit_allow_interact = allow_interact
        self.control_commit_target = target
        self.control_commit_last_distance = last_distance
        self.control_commit_steps_remaining = max(steps_remaining, 0)

    def record_interaction_transition(
        self,
        current_key: InteractionStateKey,
        action_name: str,
        family: MechanicMode,
        next_key: InteractionStateKey,
        *,
        reward: float,
        changed: bool,
        level_gain: bool,
    ) -> None:
        self.interaction_graph_action_stats.setdefault((current_key, action_name), ActionStats()).update(
            reward,
            changed=changed,
            level_gain=level_gain,
        )
        self.interaction_graph_family_stats.setdefault((current_key, family), ActionStats()).update(
            reward,
            changed=changed,
            level_gain=level_gain,
        )
        edge = (current_key, action_name, next_key)
        self.interaction_graph_transitions[edge] = self.interaction_graph_transitions.get(edge, 0) + 1


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    action: GameAction
    data: dict[str, int] | None
    key: ActionKey
    label: str


@dataclass(frozen=True, slots=True)
class OptionBundlePlan:
    first_candidate: ActionCandidate
    follow_ups: tuple[ActionCandidate, ...]
    score: float
    margin: float
    target: Coord
    path: tuple[CoarseCell, ...]
    finisher_kind: str


@dataclass(slots=True)
class PolicyDiagnostics:
    border_only_suppressions: int = 0
    sweep_probe_levels: int = 0
    sweep_probe_points: int = 0
    bootstrap_probe_reorders: int = 0
    refinement_clicks_enqueued: int = 0
    repeated_action6_penalty_events: int = 0
    keyboard_followups_enqueued: int = 0
    keyboard_turn_probes_enqueued: int = 0
    click_undo_enqueued: int = 0
    mixed_action_click_deferrals: int = 0
    frontier_plan_routes: int = 0
    abstract_frontier_plan_routes: int = 0
    rollout_plan_choices: int = 0
    interaction_target_updates: int = 0
    target_affordance_updates: int = 0
    interaction_graph_updates: int = 0
    qwen_calls: int = 0
    qwen_overrides: int = 0
    qwen_mode_calls: int = 0
    qwen_mode_overrides: int = 0
    late_click_detours: int = 0
    mechanic_hint_applications: int = 0
    qwen_hint_calls: int = 0
    qwen_hint_refreshes: int = 0
    qwen_hint_applications: int = 0
    symbolic_hint_applications: int = 0
    symbolic_hint_refreshes: int = 0
    posterior_plan_choices: int = 0
    posterior_faststart_choices: int = 0
    posterior_click_plan_choices: int = 0
    posterior_keyboard_plan_choices: int = 0
    posterior_stall_suppressions: int = 0
    control_map_plan_choices: int = 0
    interaction_graph_plan_choices: int = 0
    control_rollout_synthesis_choices: int = 0
    control_repeat_suppressions: int = 0
    target_macro_plan_choices: int = 0
    target_macro_repeat_move_choices: int = 0
    target_macro_move_interact_choices: int = 0
    target_macro_click_choices: int = 0
    target_macro_undo_choices: int = 0
    target_adjacency_cells_considered: int = 0
    target_adjacency_bridge_activations: int = 0
    target_option_paths_considered: int = 0
    target_option_plan_choices: int = 0
    target_option_bundle_injections: int = 0
    target_option_bundle_actions: int = 0
    target_value_commit_gates: int = 0
    target_value_commit_blocks: int = 0
    control_commit_injections: int = 0
    control_commit_validations: int = 0
    control_commit_aborts: int = 0
    macro_bundle_injections: int = 0
    macro_actions_used: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "border_only_suppressions": self.border_only_suppressions,
            "sweep_probe_levels": self.sweep_probe_levels,
            "sweep_probe_points": self.sweep_probe_points,
            "bootstrap_probe_reorders": self.bootstrap_probe_reorders,
            "refinement_clicks_enqueued": self.refinement_clicks_enqueued,
            "repeated_action6_penalty_events": self.repeated_action6_penalty_events,
            "keyboard_followups_enqueued": self.keyboard_followups_enqueued,
            "keyboard_turn_probes_enqueued": self.keyboard_turn_probes_enqueued,
            "click_undo_enqueued": self.click_undo_enqueued,
            "mixed_action_click_deferrals": self.mixed_action_click_deferrals,
            "frontier_plan_routes": self.frontier_plan_routes,
            "abstract_frontier_plan_routes": self.abstract_frontier_plan_routes,
            "rollout_plan_choices": self.rollout_plan_choices,
            "interaction_target_updates": self.interaction_target_updates,
            "target_affordance_updates": self.target_affordance_updates,
            "interaction_graph_updates": self.interaction_graph_updates,
            "qwen_calls": self.qwen_calls,
            "qwen_overrides": self.qwen_overrides,
            "qwen_mode_calls": self.qwen_mode_calls,
            "qwen_mode_overrides": self.qwen_mode_overrides,
            "late_click_detours": self.late_click_detours,
            "mechanic_hint_applications": self.mechanic_hint_applications,
            "qwen_hint_calls": self.qwen_hint_calls,
            "qwen_hint_refreshes": self.qwen_hint_refreshes,
            "qwen_hint_applications": self.qwen_hint_applications,
            "symbolic_hint_applications": self.symbolic_hint_applications,
            "symbolic_hint_refreshes": self.symbolic_hint_refreshes,
            "posterior_plan_choices": self.posterior_plan_choices,
            "posterior_faststart_choices": self.posterior_faststart_choices,
            "posterior_click_plan_choices": self.posterior_click_plan_choices,
            "posterior_keyboard_plan_choices": self.posterior_keyboard_plan_choices,
            "posterior_stall_suppressions": self.posterior_stall_suppressions,
            "control_map_plan_choices": self.control_map_plan_choices,
            "interaction_graph_plan_choices": self.interaction_graph_plan_choices,
            "control_rollout_synthesis_choices": self.control_rollout_synthesis_choices,
            "control_repeat_suppressions": self.control_repeat_suppressions,
            "target_macro_plan_choices": self.target_macro_plan_choices,
            "target_macro_repeat_move_choices": self.target_macro_repeat_move_choices,
            "target_macro_move_interact_choices": self.target_macro_move_interact_choices,
            "target_macro_click_choices": self.target_macro_click_choices,
            "target_macro_undo_choices": self.target_macro_undo_choices,
            "target_adjacency_cells_considered": self.target_adjacency_cells_considered,
            "target_adjacency_bridge_activations": self.target_adjacency_bridge_activations,
            "target_option_paths_considered": self.target_option_paths_considered,
            "target_option_plan_choices": self.target_option_plan_choices,
            "target_option_bundle_injections": self.target_option_bundle_injections,
            "target_option_bundle_actions": self.target_option_bundle_actions,
            "target_value_commit_gates": self.target_value_commit_gates,
            "target_value_commit_blocks": self.target_value_commit_blocks,
            "control_commit_injections": self.control_commit_injections,
            "control_commit_validations": self.control_commit_validations,
            "control_commit_aborts": self.control_commit_aborts,
            "macro_bundle_injections": self.macro_bundle_injections,
            "macro_actions_used": self.macro_actions_used,
        }


def primary_frame(frame_stack: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
    if not frame_stack:
        return np.zeros((64, 64), dtype=np.int8)
    # ARC may return several consecutive frames after an action; use the
    # settled/latest frame rather than the first animation frame.
    frame = np.asarray(frame_stack[-1], dtype=np.int16)
    if frame.shape != (64, 64):
        return np.resize(frame, (64, 64))
    return frame


def frame_signature(frame: np.ndarray) -> bytes:
    return frame.tobytes()


def is_small_border_only_change(mask: np.ndarray) -> bool:
    changed = np.argwhere(mask)
    if len(changed) == 0 or len(changed) > 8:
        return False
    height, width = mask.shape
    return bool(
        np.all(
            (changed[:, 0] == 0)
            | (changed[:, 0] == height - 1)
            | (changed[:, 1] == 0)
            | (changed[:, 1] == width - 1)
        )
    )


def extract_components(frame: np.ndarray) -> tuple[FrameComponent, ...]:
    values, counts = np.unique(frame, return_counts=True)
    background = int(values[np.argmax(counts)]) if len(values) else 0
    visited = np.zeros(frame.shape, dtype=bool)
    components: list[FrameComponent] = []
    height, width = frame.shape

    for row in range(height):
        for col in range(width):
            if visited[row, col]:
                continue
            color = int(frame[row, col])
            if color == background:
                continue
            queue = deque([(row, col)])
            visited[row, col] = True
            cells: list[Coord] = []
            while queue:
                current_row, current_col = queue.popleft()
                cells.append((current_row, current_col))
                for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if visited[next_row, next_col] or int(frame[next_row, next_col]) != color:
                        continue
                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))
            cells.sort()
            anchor = cells[len(cells) // 2]
            size = len(cells)
            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            bounds = (min(rows), min(cols), max(rows), max(cols))
            feature = (
                color,
                min(size, 15),
                min(anchor[0] // 16, 3),
                min(anchor[1] // 16, 3),
            )
            components.append(
                FrameComponent(color=color, size=size, anchor=anchor, feature=feature, bounds=bounds)
            )

    components.sort(key=lambda item: (item.size, item.color), reverse=True)
    return tuple(components)


def candidate_clicks(
    frame: np.ndarray,
    memory: EnvironmentMemory,
    *,
    limit: int = 12,
) -> tuple[tuple[Coord, ClickFeature], ...]:
    candidates: list[tuple[float, Coord, ClickFeature]] = []
    for component in extract_components(frame):
        prior = memory.click_stats.get(component.feature, ActionStats())
        score = prior.mean_reward + 0.5 * prior.level_gain_rate + 0.25 * prior.change_rate + component.size / 32.0
        top, left, bottom, right = component.bounds
        center = ((top + bottom) // 2, (left + right) // 2)
        edge_midpoints = (
            (top, (left + right) // 2),
            (bottom, (left + right) // 2),
            ((top + bottom) // 2, left),
            ((top + bottom) // 2, right),
        )
        corners = ((top, left), (top, right), (bottom, left), (bottom, right))
        points = (component.anchor, center, *edge_midpoints, *corners)
        for index, coord in enumerate(points):
            feature = (
                component.feature[0],
                component.feature[1],
                min(coord[0] // 8, 7),
                min(coord[1] // 8, 7),
            )
            point_score = score - 0.03 * index
            candidates.append((point_score, coord, feature))

    fixed_points: tuple[Coord, ...] = (
        (32, 32),
        (16, 16),
        (16, 48),
        (48, 16),
        (48, 48),
        (8, 8),
        (8, 60),
        (24, 60),
        (32, 60),
        (40, 60),
        (56, 8),
        (56, 60),
    )
    for row, col in fixed_points:
        feature = (-1, 1, min(row // 8, 7), min(col // 8, 7))
        candidates.append((memory.click_stats.get(feature, ActionStats()).mean_reward, (row, col), feature))

    ordered: list[tuple[Coord, ClickFeature]] = []
    seen: set[Coord] = set()
    for _, anchor, feature in sorted(candidates, key=lambda item: item[0], reverse=True):
        if anchor in seen:
            continue
        seen.add(anchor)
        ordered.append((anchor, feature))
        if len(ordered) >= limit:
            break
    return tuple(ordered)


def transition_mask(
    previous_frame: np.ndarray,
    next_frame: np.ndarray,
    *,
    noise_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = previous_frame != next_frame
    if noise_mask is not None:
        mask = mask & ~noise_mask
    if not np.any(mask):
        return mask
    if is_small_border_only_change(mask):
        return np.zeros_like(mask, dtype=bool)
    return mask


def reward_for_transition(
    delta_mask: np.ndarray,
    *,
    level_gain: int,
    is_win: bool,
) -> float:
    changed_pixels = int(np.count_nonzero(delta_mask))
    reward = min(changed_pixels / 128.0, 2.0)
    reward += 4.0 * level_gain
    if is_win:
        reward += 2.0
    return reward


def delta_signature(delta_mask: np.ndarray) -> bytes:
    if not np.any(delta_mask):
        return b""
    coarse = np.zeros((8, 8), dtype=np.uint8)
    rows, cols = np.nonzero(delta_mask)
    coarse[np.minimum(rows // 8, 7), np.minimum(cols // 8, 7)] = 1
    return coarse.tobytes()


def coarse_cell_for_coord(coord: Coord) -> CoarseCell:
    return (min(coord[0] // 8, 7), min(coord[1] // 8, 7))


def coarse_cell_center(cell: CoarseCell) -> Coord:
    return (min(cell[0] * 8 + 4, 63), min(cell[1] * 8 + 4, 63))


def coarse_cells_for_bounds(bounds: tuple[int, int, int, int]) -> tuple[CoarseCell, ...]:
    top, left, bottom, right = bounds
    cells: list[CoarseCell] = []
    for row in range(max(0, top // 8), min(7, bottom // 8) + 1):
        for col in range(max(0, left // 8), min(7, right // 8) + 1):
            cells.append((row, col))
    return tuple(cells)


def coarse_path_distance(
    start: CoarseCell,
    goal: CoarseCell,
    *,
    blocked: frozenset[CoarseCell] = frozenset(),
) -> int:
    if start == goal:
        return 0
    frontier: deque[tuple[CoarseCell, int]] = deque([(start, 0)])
    seen = {start}
    while frontier:
        (row, col), dist = frontier.popleft()
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (row + delta_row, col + delta_col)
            if not (0 <= nxt[0] < 8 and 0 <= nxt[1] < 8):
                continue
            if nxt in blocked or nxt in seen:
                continue
            if nxt == goal:
                return dist + 1
            seen.add(nxt)
            frontier.append((nxt, dist + 1))
    return 99


def coarse_shortest_path(
    start: CoarseCell,
    goal: CoarseCell,
    *,
    blocked: frozenset[CoarseCell] = frozenset(),
) -> tuple[CoarseCell, ...]:
    if start == goal:
        return ()
    frontier: deque[CoarseCell] = deque([start])
    parents: dict[CoarseCell, CoarseCell | None] = {start: None}
    while frontier:
        row, col = frontier.popleft()
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (row + delta_row, col + delta_col)
            if not (0 <= nxt[0] < 8 and 0 <= nxt[1] < 8):
                continue
            if nxt in blocked or nxt in parents:
                continue
            parents[nxt] = (row, col)
            if nxt == goal:
                path: list[CoarseCell] = [goal]
                current = goal
                while parents[current] is not None:
                    current = parents[current]
                    if current != start:
                        path.append(current)
                path.reverse()
                return tuple(path)
            frontier.append(nxt)
    return ()


def coarse_blocked_cells_for_components(
    frame: np.ndarray,
    *,
    actor_feature: ClickFeature | None,
    target_cells: frozenset[CoarseCell],
) -> frozenset[CoarseCell]:
    blocked: set[CoarseCell] = set()
    for component in extract_components(frame):
        if actor_feature is not None and component.feature == actor_feature:
            continue
        component_cells = coarse_cells_for_bounds(component.bounds)
        if any(cell in target_cells for cell in component_cells):
            continue
        if component.size < 8:
            continue
        blocked.update(component_cells)
    return frozenset(blocked)


def feature_region_cells(feature: ClickFeature) -> tuple[CoarseCell, ...]:
    _, _, region_row, region_col = feature
    base_row = max(0, min(region_row * 2, 6))
    base_col = max(0, min(region_col * 2, 6))
    return (
        (base_row, base_col),
        (base_row, base_col + 1),
        (base_row + 1, base_col),
        (base_row + 1, base_col + 1),
    )


def coarse_cells_from_mask(delta_mask: np.ndarray, *, limit: int = 6) -> tuple[CoarseCell, ...]:
    if not np.any(delta_mask):
        return ()
    rows, cols = np.nonzero(delta_mask)
    counts: dict[CoarseCell, int] = {}
    for row, col in zip(rows, cols, strict=True):
        cell = (min(int(row) // 8, 7), min(int(col) // 8, 7))
        counts[cell] = counts.get(cell, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(cell for cell, _count in ranked[:limit])


def hotspot_cells(memory: EnvironmentMemory, *, limit: int = 6) -> tuple[tuple[CoarseCell, float], ...]:
    counts = memory.pixel_change_counts.astype(np.float32).copy()
    noise = memory.noise_mask()
    if noise.shape == counts.shape:
        counts[noise] = 0.0
    coarse = counts.reshape(8, 8, 8, 8).sum(axis=(1, 3))
    max_value = float(coarse.max(initial=0.0))
    if max_value <= 0.0:
        return ()
    ranked: list[tuple[CoarseCell, float]] = []
    for row in range(8):
        for col in range(8):
            value = float(coarse[row, col])
            if value <= 0.0:
                continue
            ranked.append(((row, col), value / max_value))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return tuple(ranked[:limit])


def component_by_feature(frame: np.ndarray, actor_features: dict[ClickFeature, int]) -> FrameComponent | None:
    if not actor_features:
        return None
    components = extract_components(frame)
    weighted = sorted(
        components,
        key=lambda component: actor_features.get(component.feature, 0),
        reverse=True,
    )
    if not weighted:
        return None
    if actor_features.get(weighted[0].feature, 0) <= 0:
        return None
    return weighted[0]


def component_by_actor_prior(frame: np.ndarray, memory: EnvironmentMemory) -> FrameComponent | None:
    component = component_by_feature(frame, memory.actor_features)
    if component is not None:
        return component
    if not memory.actor_cell_scores:
        return None
    components = extract_components(frame)
    if not components:
        return None

    def cell_prior(cell: CoarseCell) -> float:
        row, col = cell
        score = memory.actor_cell_scores.get(cell, 0.0)
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            score = max(score, 0.6 * memory.actor_cell_scores.get((row + delta_row, col + delta_col), 0.0))
        return score

    ranked = sorted(
        components,
        key=lambda current: (
            cell_prior(coarse_cell_for_coord(current.anchor))
            + (0.18 if current.size <= 9 else 0.0)
            + (0.08 if current.size <= 4 else 0.0)
        ),
        reverse=True,
    )
    best = ranked[0]
    if cell_prior(coarse_cell_for_coord(best.anchor)) < 0.75:
        return None
    return best


def actor_anchor_guess(frame: np.ndarray, memory: EnvironmentMemory, level_memory: LevelMemory) -> Coord | None:
    actor_component = component_by_actor_prior(frame, memory)
    if actor_component is not None:
        return actor_component.anchor
    signature = frame_signature(frame)
    state_anchor = level_memory.state_actor_anchors.get(signature)
    if state_anchor is not None:
        return state_anchor
    if not memory.actor_cell_scores:
        return None
    best_cell = max(memory.actor_cell_scores.items(), key=lambda item: item[1])[0]
    return (best_cell[0] * 8 + 4, best_cell[1] * 8 + 4)


def resolved_actor_component(
    frame: np.ndarray,
    memory: EnvironmentMemory,
    level_memory: LevelMemory,
) -> FrameComponent | None:
    actor_component = component_by_actor_prior(frame, memory)
    if actor_component is not None:
        return actor_component
    anchor = actor_anchor_guess(frame, memory, level_memory)
    if anchor is None:
        return None
    components = extract_components(frame)
    nearest: FrameComponent | None = None
    nearest_distance = 999
    for component in components:
        distance = manhattan_to_bounds(anchor, component.bounds)
        if distance < nearest_distance:
            nearest = component
            nearest_distance = distance
    if nearest is not None and nearest_distance <= 1:
        return nearest
    row, col = anchor
    return FrameComponent(
        color=-1,
        size=1,
        anchor=anchor,
        feature=(-99, 1, min(row // 8, 7), min(col // 8, 7)),
        bounds=(row, col, row, col),
    )


def infer_component_motion(previous_frame: np.ndarray, next_frame: np.ndarray) -> tuple[ClickFeature, Coord] | None:
    previous = extract_components(previous_frame)
    current = extract_components(next_frame)
    best_data: tuple[ClickFeature, Coord] | None = None
    best_score = float("-inf")
    second_best_score = float("-inf")
    best_tuple: tuple[int, int, int, ClickFeature, Coord] | None = None
    for before in previous:
        for after in current:
            if before.color != after.color:
                continue
            size_gap = abs(before.size - after.size)
            size_tolerance = max(2, min(before.size, after.size) // 3)
            if size_gap > size_tolerance:
                continue
            delta = (after.anchor[0] - before.anchor[0], after.anchor[1] - before.anchor[1])
            distance = abs(delta[0]) + abs(delta[1])
            if distance == 0 or distance > 12:
                continue
            size_value = min(before.size, after.size)
            score_value = size_value - size_gap - distance * 0.8
            score_tuple = (size_value, -size_gap, -distance, before.feature, delta)
            if best_tuple is None or score_tuple > best_tuple:
                second_best_score = best_score
                best_score = score_value
                best_tuple = score_tuple
                best_data = (before.feature, delta)
            elif score_value > second_best_score:
                second_best_score = score_value
    if best_data is None or best_score - second_best_score < 0.75:
        return None
    return best_data


def manhattan_to_bounds(coord: Coord, bounds: tuple[int, int, int, int]) -> int:
    row, col = coord
    top, left, bottom, right = bounds
    row_gap = 0 if top <= row <= bottom else min(abs(row - top), abs(row - bottom))
    col_gap = 0 if left <= col <= right else min(abs(col - left), abs(col - right))
    return row_gap + col_gap


def mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    changed = np.argwhere(mask)
    if len(changed) == 0:
        return None
    rows = changed[:, 0]
    cols = changed[:, 1]
    return (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))


def bounds_gap(a_bounds: tuple[int, int, int, int], b_bounds: tuple[int, int, int, int]) -> int:
    a_top, a_left, a_bottom, a_right = a_bounds
    b_top, b_left, b_bottom, b_right = b_bounds
    row_gap = max(0, max(a_top - b_bottom, b_top - a_bottom))
    col_gap = max(0, max(a_left - b_right, b_left - a_right))
    return row_gap + col_gap


def bounds_overlap_area(a_bounds: tuple[int, int, int, int], b_bounds: tuple[int, int, int, int]) -> int:
    top = max(a_bounds[0], b_bounds[0])
    left = max(a_bounds[1], b_bounds[1])
    bottom = min(a_bounds[2], b_bounds[2])
    right = min(a_bounds[3], b_bounds[3])
    if top > bottom or left > right:
        return 0
    return (bottom - top + 1) * (right - left + 1)


def infer_actor_features(
    previous_frame: np.ndarray,
    next_frame: np.ndarray,
    delta_mask: np.ndarray,
    *,
    limit: int = 4,
) -> tuple[ClickFeature, ...]:
    changed_bounds = mask_bounds(delta_mask)
    if changed_bounds is None:
        return ()
    feature_scores: dict[ClickFeature, float] = {}
    for frame in (previous_frame, next_frame):
        for component in extract_components(frame):
            overlap = bounds_overlap_area(component.bounds, changed_bounds)
            distance = bounds_gap(component.bounds, changed_bounds)
            if overlap == 0 and distance > 1:
                continue
            score = overlap / max(component.size, 1)
            score += 0.7 / (1.0 + distance)
            if component.size <= 9:
                score += 0.25
            if component.size <= 4:
                score += 0.15
            current = feature_scores.get(component.feature, 0.0)
            feature_scores[component.feature] = max(current, score)
    ranked = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
    return tuple(feature for feature, score in ranked[:limit] if score >= 0.2)


def infer_interaction_targets(
    previous_frame: np.ndarray,
    next_frame: np.ndarray,
    delta_mask: np.ndarray,
    actor_before: FrameComponent | None,
    actor_after: FrameComponent | None,
) -> tuple[ClickFeature, ...]:
    changed_bounds = mask_bounds(delta_mask)
    if changed_bounds is None and actor_before is None and actor_after is None:
        return ()

    actor_features = {
        component.feature
        for component in (actor_before, actor_after)
        if component is not None
    }
    features: list[ClickFeature] = []
    seen: set[ClickFeature] = set()
    contexts = (
        (extract_components(previous_frame), actor_before),
        (extract_components(next_frame), actor_after),
    )
    for components, actor_component in contexts:
        for component in components:
            if component.feature in actor_features:
                continue
            near_change = changed_bounds is not None and bounds_gap(component.bounds, changed_bounds) <= 2
            near_actor = actor_component is not None and manhattan_to_bounds(actor_component.anchor, component.bounds) <= 2
            if not near_change and not near_actor:
                continue
            if component.feature in seen:
                continue
            seen.add(component.feature)
            features.append(component.feature)
    return tuple(features)


def interaction_target_priority(component: FrameComponent, memory: EnvironmentMemory) -> float:
    stats = memory.interaction_target_stats.get(component.feature)
    if stats is None:
        return component.size / 48.0
    return (
        component.size / 48.0
        + 0.3 * max(stats.mean_reward, 0.0)
        + 1.2 * stats.change_rate
        + 1.4 * stats.level_gain_rate
    )


def per_level_budget(baseline_actions: tuple[int, ...], level_index: int) -> int:
    if not baseline_actions:
        return 64
    baseline = baseline_actions[min(level_index, len(baseline_actions) - 1)]
    return min(220, max(24, int(baseline * 3.0)))


def refinement_clicks(
    clicked_coord: Coord,
    delta_mask: np.ndarray,
    *,
    limit: int = 8,
) -> tuple[tuple[Coord, ClickFeature], ...]:
    coords: list[Coord] = [clicked_coord]
    changed = np.argwhere(delta_mask)
    if len(changed):
        centroid = tuple(int(value) for value in np.rint(changed.mean(axis=0)))
        rows = changed[:, 0]
        cols = changed[:, 1]
        coords.extend(
            [
                centroid,
                (int(rows.min()), int(cols.min())),
                (int(rows.min()), int(cols.max())),
                (int(rows.max()), int(cols.min())),
                (int(rows.max()), int(cols.max())),
            ]
        )
    for delta_row, delta_col in (
        (-8, 0),
        (8, 0),
        (0, -8),
        (0, 8),
        (-4, 0),
        (4, 0),
        (0, -4),
        (0, 4),
        (-2, -2),
        (2, 2),
    ):
        coords.append(
            (
                min(63, max(0, clicked_coord[0] + delta_row)),
                min(63, max(0, clicked_coord[1] + delta_col)),
            )
        )

    ordered: list[tuple[Coord, ClickFeature]] = []
    seen: set[Coord] = set()
    for row, col in coords:
        coord = (row, col)
        if coord in seen:
            continue
        seen.add(coord)
        ordered.append((coord, (-2, 1, min(row // 8, 7), min(col // 8, 7))))
        if len(ordered) >= limit:
            break
    return tuple(ordered)


def sweep_clicks() -> tuple[tuple[Coord, ClickFeature], ...]:
    coords: list[Coord] = []
    edge_rows = (16, 24, 32, 40, 48)
    edge_cols = (16, 24, 32, 40, 48)
    for row in edge_rows:
        coords.extend(((row, 4), (row, 60)))
    for col in edge_cols:
        coords.extend(((4, col), (60, col)))

    ordered: list[tuple[Coord, ClickFeature]] = []
    seen: set[Coord] = set()
    for row, col in coords:
        coord = (row, col)
        if coord in seen:
            continue
        seen.add(coord)
        ordered.append((coord, (-3, 1, min(row // 8, 7), min(col // 8, 7))))
    return tuple(ordered)


def keyboard_actions(available_actions: tuple[GameAction, ...] | list[GameAction]) -> tuple[GameAction, ...]:
    return tuple(action for action in available_actions if action not in {GameAction.ACTION6, GameAction.ACTION7})


def action_mask(available_actions: tuple[GameAction, ...] | list[GameAction]) -> int:
    mask = 0
    actions = tuple(available_actions)
    if GameAction.ACTION6 in actions:
        mask |= 1
    if any(action not in {GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7} for action in actions):
        mask |= 2
    if GameAction.ACTION5 in actions:
        mask |= 4
    if GameAction.ACTION7 in actions:
        mask |= 8
    return mask


class ArcAgi3TACTICPublicAgent:
    def __init__(
        self,
        *,
        action_advisor: OllamaArcAdvisor | None = None,
        action_advisor_budget_per_level: int = 0,
        mechanic_advisor: OllamaArcAdvisor | None = None,
        mechanic_advisor_budget_per_level: int = 0,
        tuning: PolicyTuning | None = None,
    ) -> None:
        self.memory: dict[str, EnvironmentMemory] = {}
        self.diagnostics = PolicyDiagnostics()
        self.action_advisor = action_advisor
        self.action_advisor_budget_per_level = action_advisor_budget_per_level
        self.mechanic_advisor = mechanic_advisor
        self.mechanic_advisor_budget_per_level = mechanic_advisor_budget_per_level
        self.tuning = tuning or PolicyTuning()

    def play_environment(
        self,
        env,
        *,
        env_id: str,
        baseline_actions: tuple[int, ...],
        show_progress: bool = False,
    ) -> None:
        obs = env.reset()
        if obs is None:
            return
        base_name = env_id.split("-", 1)[0]
        memory = self.memory.setdefault(base_name, EnvironmentMemory())
        level_memory = LevelMemory()
        level_memory.mark_seen(frame_signature(primary_frame(obs.frame)))
        self._prime_probe_queue(obs, env.action_space, memory, level_memory)
        last_level = int(obs.levels_completed)
        step_budget = per_level_budget(baseline_actions, last_level)
        spinner = Spinner(f"{base_name}-agent") if show_progress else None

        while obs.state == ArcGameState.NOT_FINISHED and level_memory.step_index < step_budget:
            frame = primary_frame(obs.frame)
            signature = frame_signature(frame)
            available = tuple(env.action_space)
            current_interaction_key = self._interaction_state_key(frame, available, memory, level_memory)
            level_memory.note_interaction_state(signature, current_interaction_key)
            current_actor_component = component_by_actor_prior(frame, memory)
            if current_actor_component is not None:
                level_memory.note_actor_anchor(signature, current_actor_component.anchor)
            candidate = self._choose_action(frame, signature, available, memory, level_memory)
            next_obs = env.step(candidate.action, data=candidate.data)
            if next_obs is None:
                break

            next_frame = primary_frame(next_obs.frame)
            next_signature = frame_signature(next_frame)
            next_level = int(next_obs.levels_completed)
            level_gain = max(0, next_level - last_level)
            raw_delta_mask = frame != next_frame
            memory.record_transition_mask(raw_delta_mask)
            noise_mask = memory.noise_mask()
            visible_delta = raw_delta_mask & ~noise_mask
            if is_small_border_only_change(visible_delta):
                self.diagnostics.border_only_suppressions += 1
            meaningful_mask = transition_mask(frame, next_frame, noise_mask=noise_mask)
            changed = bool(np.any(meaningful_mask))
            reward = reward_for_transition(
                meaningful_mask,
                level_gain=level_gain,
                is_win=next_obs.state == ArcGameState.WIN,
            )
            next_actor_component = component_by_actor_prior(next_frame, memory)
            inferred_motion = None
            if candidate.action not in {GameAction.ACTION6, GameAction.ACTION7} and changed:
                inferred_motion = infer_component_motion(frame, next_frame)
                if inferred_motion is not None:
                    feature, delta = inferred_motion
                    memory.record_motion(candidate.action.name, delta, feature)
                    for cell in feature_region_cells(feature):
                        memory.record_actor_cell_hint(cell, weight=2.0)
                else:
                    for rank, feature in enumerate(infer_actor_features(frame, next_frame, meaningful_mask)):
                        memory.record_actor_hint(feature, weight=2 if rank == 0 else 1)
                        for cell in feature_region_cells(feature):
                            memory.record_actor_cell_hint(cell, weight=2.0 if rank == 0 else 1.0)
            if candidate.action not in {GameAction.ACTION6, GameAction.ACTION7}:
                level_memory.note_keyboard_outcome(candidate.action.name, changed=changed)

            memory.stat_for(candidate.key).update(
                reward,
                changed=changed,
                level_gain=level_gain > 0,
            )
            memory.record_effect(candidate.key, meaningful_mask)
            if candidate.action == GameAction.ACTION6 and candidate.key[1] is not None:
                memory.click_stat_for(candidate.key[1]).update(
                    reward,
                    changed=changed,
                    level_gain=level_gain > 0,
                )
            if (
                candidate.action not in {GameAction.ACTION6, GameAction.ACTION7}
                and current_actor_component is not None
            ):
                memory.keyboard_context_stat_for(
                    candidate.action.name,
                    coarse_cell_for_coord(current_actor_component.anchor),
                ).update(
                    reward,
                    changed=changed,
                    level_gain=level_gain > 0,
                )
            if candidate.action not in {GameAction.ACTION6, GameAction.ACTION7}:
                for feature in level_memory.active_target_contacts():
                    memory.target_affordance_stat_for(feature, candidate.action.name).update(
                        reward,
                        changed=changed,
                        level_gain=level_gain > 0,
                    )
                    self.diagnostics.target_affordance_updates += 1
            if candidate.action not in {GameAction.ACTION6, GameAction.ACTION7} and changed:
                target_features = infer_interaction_targets(
                    frame,
                    next_frame,
                    meaningful_mask,
                    current_actor_component,
                    next_actor_component,
                )
                for feature in target_features:
                    memory.interaction_target_stat_for(feature).update(
                        reward,
                        changed=changed,
                        level_gain=level_gain > 0,
                    )
                if target_features:
                    self.diagnostics.interaction_target_updates += len(target_features)
                    level_memory.note_target_contacts(target_features)
                    level_memory.note_target_values(
                        target_features,
                        reward=reward,
                        changed=changed,
                        level_gain=level_gain > 0,
                    )
            if candidate.action == GameAction.ACTION6 and changed and candidate.data is not None:
                clicked_coord = (int(candidate.data["y"]), int(candidate.data["x"]))
                subgoal_cells = (coarse_cell_for_coord(clicked_coord), *coarse_cells_from_mask(meaningful_mask))
                level_memory.record_subgoal_cells(
                    subgoal_cells,
                    min(3.0, reward + 0.5 * level_gain),
                )
                for coord, feature in refinement_clicks(clicked_coord, meaningful_mask):
                    refined = ActionCandidate(
                        action=GameAction.ACTION6,
                        data={"x": coord[1], "y": coord[0]},
                        key=(GameAction.ACTION6.name, feature),
                        label=f"{GameAction.ACTION6.name}@{coord[0]},{coord[1]}",
                    )
                    if not level_memory.has_tried(signature, refined.key):
                        level_memory.probe_queue.appendleft(refined)
                        self.diagnostics.refinement_clicks_enqueued += 1
                if (
                    level_gain == 0
                    and GameAction.ACTION7 in tuple(env.action_space)
                    and level_memory.step_index <= 12
                ):
                    recent_click_dead_ends = self._recent_event_count(
                        level_memory,
                        "ACTION6 changed=1 level_gain=0",
                    )
                    undo_candidate = ActionCandidate(
                        action=GameAction.ACTION7,
                        data=None,
                        key=(GameAction.ACTION7.name, None),
                        label=GameAction.ACTION7.name,
                    )
                    if recent_click_dead_ends >= 1 or level_memory.family_failure_count("CLICK") >= 2:
                        level_memory.probe_queue.appendleft(undo_candidate)
                    else:
                        level_memory.probe_queue.append(undo_candidate)
                    self.diagnostics.click_undo_enqueued += 1
            if (
                candidate.action not in {GameAction.ACTION6, GameAction.ACTION7}
                and inferred_motion is not None
                and level_gain == 0
            ):
                self._enqueue_keyboard_followups(
                    executed=candidate,
                    next_signature=next_signature,
                    next_frame=next_frame,
                    available_actions=available,
                    memory=memory,
                    level_memory=level_memory,
                    changed_pixels=int(np.count_nonzero(meaningful_mask)),
                )

            level_memory.mark_tried(signature, candidate.key)
            level_memory.probed_keys.add(candidate.key)
            level_memory.observe_transition(signature, candidate.key, next_signature)
            next_available = tuple(env.action_space)
            next_interaction_key = self._interaction_state_key(next_frame, next_available, memory, level_memory)
            level_memory.note_interaction_state(next_signature, next_interaction_key)
            level_memory.record_interaction_transition(
                current_interaction_key,
                candidate.action.name,
                action_family(candidate.action),
                next_interaction_key,
                reward=reward,
                changed=changed,
                level_gain=level_gain > 0,
            )
            self.diagnostics.interaction_graph_updates += 1
            repeated_state = next_signature in level_memory.seen_states
            level_memory.record_family_outcome(
                action_family(candidate.action),
                changed=changed,
                level_gain=level_gain > 0,
                repeated_state=repeated_state,
            )
            level_memory.mark_seen(next_signature)
            level_memory.step_index += 1
            if changed:
                level_memory.last_changed_step = level_memory.step_index
            if next_actor_component is not None:
                memory.record_actor_cell_hint(coarse_cell_for_coord(next_actor_component.anchor), weight=1.5)
                level_memory.actor_positions.add(next_actor_component.anchor)
                level_memory.note_actor_anchor(next_signature, next_actor_component.anchor)

            self._update_control_commit_after_transition(
                level_memory,
                candidate,
                current_actor_component=current_actor_component,
                next_actor_component=next_actor_component,
                changed=changed,
                level_gain=level_gain,
            )

            if (
                candidate.action not in {GameAction.ACTION6, GameAction.ACTION7}
                and changed
                and level_gain == 0
                and GameAction.ACTION7 in tuple(env.action_space)
                and not level_memory.probe_queue
                and level_memory.step_index <= 12
            ):
                level_memory.probe_queue.appendleft(
                    ActionCandidate(
                        action=GameAction.ACTION7,
                        data=None,
                        key=(GameAction.ACTION7.name, None),
                        label=GameAction.ACTION7.name,
                    )
                )

            level_memory.recent_events.append(
                f"{candidate.label} changed={int(changed)} level_gain={level_gain} reward={reward:.2f}"
            )

            if next_level != last_level:
                level_memory = LevelMemory()
                level_memory.mark_seen(next_signature)
                self._prime_probe_queue(next_obs, env.action_space, memory, level_memory)
                last_level = next_level
                step_budget = per_level_budget(baseline_actions, last_level)

            obs = next_obs
            if spinner is not None and (level_memory.step_index == 1 or level_memory.step_index % 25 == 0):
                spinner.tick(
                    f"steps={level_memory.step_index} levels={obs.levels_completed}/{obs.win_levels}"
                )

        if spinner is not None:
            spinner.finish(f"levels={obs.levels_completed}/{obs.win_levels}")

    def _choose_action(
        self,
        frame: np.ndarray,
        signature: bytes,
        available_actions: tuple[GameAction, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
    ) -> ActionCandidate:
        candidates = self._candidates(frame, available_actions, memory)
        level_memory.pending_option_plan = None
        level_memory.mark_available(signature, tuple(candidate.key for candidate in candidates))
        level_memory.note_abstract_state(
            signature,
            self._abstract_state_key(frame, available_actions, memory, level_memory),
        )
        self._maybe_refresh_mechanic_hint(frame, candidates, memory, level_memory)
        macro_candidate = self._pop_macro_candidate(signature, candidates, level_memory)
        if macro_candidate is not None:
            self.diagnostics.macro_actions_used += 1
            return macro_candidate
        posterior_plan, posterior_score, posterior_margin, posterior_kind = self._posterior_guided_decision(
            frame,
            candidates,
            available_actions,
            memory,
            level_memory,
        )
        pressure = level_memory.loop_pressure(signature)
        actor_component = component_by_actor_prior(frame, memory)
        faststart_confident = False
        if (
            pressure == 0
            and level_memory.mechanic_hint is not None
            and level_memory.mechanic_hint.source == "SYMBOLIC"
            and posterior_plan is not None
            and level_memory.mechanic_hint.confidence >= 0.8
        ):
            hint_mode = level_memory.mechanic_hint.mode
            if posterior_kind == "CLICK":
                faststart_confident = (
                    posterior_score >= self.tuning.faststart_click_score
                    and hint_mode in {"CLICK", "MIXED", "UNDO"}
                )
            elif posterior_kind == "KEYBOARD":
                faststart_confident = (
                    posterior_score >= self.tuning.faststart_keyboard_score
                    and posterior_margin >= self.tuning.faststart_keyboard_margin
                    and level_memory.step_index >= 5
                    and hint_mode in {"MOVE", "INTERACT", "MIXED"}
                )
        posterior_confident = (
            posterior_plan is not None
            and (
                faststart_confident
                or (
                    posterior_kind == "KEYBOARD"
                    and level_memory.mechanic_hint is not None
                    and level_memory.mechanic_hint.mode in {"CLICK", "MIXED"}
                    and posterior_score >= self.tuning.posterior_click_override_score
                    and posterior_margin >= self.tuning.posterior_click_override_margin
                )
                or (
                    pressure >= 1
                    and (
                        posterior_score >= self.tuning.posterior_pressure_score
                        or (
                            pressure >= 2
                            and posterior_score >= self.tuning.posterior_relaxed_score
                            and posterior_margin >= self.tuning.posterior_relaxed_margin
                        )
                    )
                )
            )
        )
        posterior_stall_suppressed = False
        if (
            posterior_confident
            and self._should_suppress_stalled_posterior(
                posterior_kind=posterior_kind,
                actor_component=actor_component,
                level_memory=level_memory,
                faststart_confident=faststart_confident,
            )
        ):
            posterior_confident = False
            posterior_stall_suppressed = True
            self.diagnostics.posterior_stall_suppressions += 1
        cached_control_map: tuple[ActionCandidate | None, float, float] | None = None
        cached_rollout: tuple[ActionCandidate | None, float, float] | None = None
        if posterior_confident and posterior_kind == "KEYBOARD" and pressure >= 2 and self.action_advisor is None:
            cached_control_map = self._control_map_decision(
                frame,
                candidates,
                memory,
                level_memory,
                actor_component=actor_component,
            )
            cached_rollout = self._planned_rollout_decision(
                frame,
                available_actions,
                memory,
                level_memory,
            )
            synthesis_plan, synthesis_score, synthesis_margin = self._control_rollout_synthesis_decision(
                posterior_plan,
                posterior_kind,
                cached_control_map[0],
                cached_control_map[1],
                cached_control_map[2],
                cached_rollout[0],
                cached_rollout[1],
                cached_rollout[2],
                pressure=pressure,
            )
            if synthesis_plan is not None:
                self.diagnostics.control_rollout_synthesis_choices += 1
                self._seed_control_commit_bundle(
                    frame,
                    available_actions,
                    candidates,
                    memory,
                    level_memory,
                    synthesis_plan,
                    control_score=synthesis_score,
                    control_margin=synthesis_margin,
                    actor_component=actor_component,
                )
                return synthesis_plan
        cached_target_macro: tuple[ActionCandidate | None, float, float, str | None] | None = None
        if posterior_confident:
            if self.action_advisor is None and actor_component is not None and posterior_kind == "KEYBOARD":
                cached_target_macro = self._target_conditioned_macro_decision(
                    frame,
                    candidates,
                    available_actions,
                    memory,
                    level_memory,
                    actor_component=actor_component,
                )
                target_macro_plan, target_macro_score, _target_macro_margin, target_macro_kind = cached_target_macro
                if (
                    target_macro_plan is not None
                    and target_macro_kind in {"MOVE_INTERACT", "OPTION_PATH"}
                    and target_macro_plan.action == posterior_plan.action
                ):
                    return self._apply_target_macro_choice(
                        frame,
                        available_actions,
                        candidates,
                        memory,
                        level_memory,
                        target_macro_plan,
                        target_macro_kind,
                        target_macro_score=target_macro_score,
                        actor_component=actor_component,
                    )
            self.diagnostics.posterior_plan_choices += 1
            if faststart_confident:
                self.diagnostics.posterior_faststart_choices += 1
            if posterior_kind == "CLICK":
                self.diagnostics.posterior_click_plan_choices += 1
            elif posterior_kind == "KEYBOARD":
                self.diagnostics.posterior_keyboard_plan_choices += 1
            self.diagnostics.mechanic_hint_applications += 1
            if level_memory.mechanic_hint is not None:
                if level_memory.mechanic_hint.source == "QWEN":
                    self.diagnostics.qwen_hint_applications += 1
                elif level_memory.mechanic_hint.source == "SYMBOLIC":
                    self.diagnostics.symbolic_hint_applications += 1
            self._seed_macro_bundle(
                frame,
                available_actions,
                candidates,
                memory,
                level_memory,
                posterior_plan,
                posterior_kind,
            )
            return posterior_plan
        current_abstract_key = level_memory.abstract_state_keys.get(signature)
        abstract_revisits = 0
        if current_abstract_key is not None:
            abstract_revisits = level_memory.abstract_revisit_count(current_abstract_key)
        abstract_frontier_ready = (
            current_abstract_key is not None
            and abstract_revisits >= 1
            and pressure >= 2
            and (
                actor_component is None
                or level_memory.family_failure_count("MOVE") >= 2
            )
        )
        if abstract_frontier_ready:
            planned = self._planned_frontier_action(frame, signature, candidates, memory, level_memory)
            if planned is not None:
                self.diagnostics.frontier_plan_routes += 1
                return planned
        if (
            actor_component is not None
            and level_memory.active_target_cells()
            and level_memory.step_index >= 5
            and level_memory.family_failure_count("MOVE") >= 1
            and not posterior_stall_suppressed
        ):
            contact_rollout, contact_rollout_score, contact_rollout_margin = self._planned_rollout_decision(
                frame,
                available_actions,
                memory,
                level_memory,
            )
            if (
                contact_rollout is not None
                and contact_rollout_score >= self.tuning.contact_rollout_score
                and contact_rollout_margin >= self.tuning.contact_rollout_margin
            ):
                self.diagnostics.rollout_plan_choices += 1
                return contact_rollout
        if self.action_advisor is None and actor_component is not None:
            if cached_target_macro is None:
                cached_target_macro = self._target_conditioned_macro_decision(
                    frame,
                    candidates,
                    available_actions,
                    memory,
                    level_memory,
                    actor_component=actor_component,
                )
            target_macro_plan, target_macro_score, _target_macro_margin, target_macro_kind = cached_target_macro
            if target_macro_plan is not None and target_macro_kind is not None:
                return self._apply_target_macro_choice(
                    frame,
                    available_actions,
                    candidates,
                    memory,
                    level_memory,
                    target_macro_plan,
                    target_macro_kind,
                    target_macro_score=target_macro_score,
                    actor_component=actor_component,
                )
        if self.action_advisor is None:
            if cached_control_map is None:
                cached_control_map = self._control_map_decision(
                    frame,
                    candidates,
                    memory,
                    level_memory,
                    actor_component=actor_component,
                )
            control_map_plan, control_map_score, control_map_margin = cached_control_map
            if (
                control_map_plan is not None
                and (
                    control_map_score >= self.tuning.control_map_accept_score
                    or (
                        control_map_score >= self.tuning.control_map_soft_score
                        and control_map_margin >= self.tuning.control_map_soft_margin
                    )
                )
            ):
                self.diagnostics.control_map_plan_choices += 1
                self._seed_control_commit_bundle(
                    frame,
                    available_actions,
                    candidates,
                    memory,
                    level_memory,
                    control_map_plan,
                    control_score=control_map_score,
                    control_margin=control_map_margin,
                    actor_component=actor_component,
                )
                return control_map_plan
        interaction_graph_plan, interaction_graph_score, interaction_graph_margin = self._interaction_graph_decision(
            frame,
            candidates,
            available_actions,
            memory,
            level_memory,
        )
        if (
            interaction_graph_plan is not None
            and (
                interaction_graph_score >= self.tuning.interaction_graph_accept_score
                or (
                    interaction_graph_score >= self.tuning.interaction_graph_soft_score
                    and interaction_graph_margin >= self.tuning.interaction_graph_soft_margin
                )
            )
        ):
            self.diagnostics.interaction_graph_plan_choices += 1
            return interaction_graph_plan
        probe = self._pop_probe_candidate(frame, signature, memory, level_memory)
        if probe is not None:
            return probe
        if cached_rollout is None:
            cached_rollout = self._planned_rollout_decision(
                frame,
                available_actions,
                memory,
                level_memory,
            )
        rollout, rollout_score, rollout_margin = cached_rollout
        rollout_confident = rollout is not None and rollout_score >= 1.2 and rollout_margin >= 0.15
        if rollout_confident:
            self.diagnostics.rollout_plan_choices += 1
            return rollout
        advised_mode = self._advisor_mode(
            frame,
            candidates,
            memory,
            level_memory,
            rollout=rollout,
            rollout_score=rollout_score,
            rollout_margin=rollout_margin,
        )
        mode_candidate = self._candidate_from_advisor_mode(
            advised_mode,
            candidates,
            signature,
            memory,
            level_memory,
            rollout,
        )
        if mode_candidate is not None:
            self.diagnostics.qwen_mode_overrides += 1
            return mode_candidate
        advised = self._advisor_action(
            frame,
            candidates,
            memory,
            level_memory,
            rollout=rollout,
            rollout_score=rollout_score,
            rollout_margin=rollout_margin,
        )
        if advised is not None:
            self.diagnostics.qwen_overrides += 1
            return advised
        if rollout is not None:
            self.diagnostics.rollout_plan_choices += 1
            return rollout
        planned = self._planned_frontier_action(frame, signature, candidates, memory, level_memory)
        if planned is not None:
            self.diagnostics.frontier_plan_routes += 1
            return planned

        interaction_targets: tuple[tuple[Coord, float], ...] = ()
        if actor_component is not None:
            interaction_targets = self._interaction_targets(frame, memory, level_memory, actor_component)
        best = candidates[0]
        best_score = float("-inf")
        best_repetition_penalty = 0.0
        best_click_deferral = 0.0
        best_mechanic_bonus = 0.0
        best_control_repeat_penalty = 0.0
        pressure = level_memory.loop_pressure(signature)
        for candidate in candidates:
            stats = memory.stat_for(candidate.key)
            novel_bonus = 1.0 if not level_memory.has_tried(signature, candidate.key) else -0.15
            stall_penalty = 0.0
            if candidate.action == GameAction.ACTION7 and level_memory.step_index - level_memory.last_changed_step < 2:
                stall_penalty = -0.5
            motion_bonus = 0.0
            if actor_component is not None and candidate.action not in {GameAction.ACTION6, GameAction.ACTION7}:
                delta, confidence = memory.best_delta(candidate.action.name)
                if delta is not None:
                    next_anchor = (
                        actor_component.anchor[0] + delta[0],
                        actor_component.anchor[1] + delta[1],
                    )
                    if 0 <= next_anchor[0] < frame.shape[0] and 0 <= next_anchor[1] < frame.shape[1]:
                        motion_bonus += 0.45 * confidence
                        if next_anchor not in level_memory.actor_positions:
                            motion_bonus += 0.35
                    else:
                        motion_bonus -= 0.35
            keyboard_context_bonus = 0.0
            projected_next_anchor: Coord | None = None
            if actor_component is not None and candidate.action not in {GameAction.ACTION6, GameAction.ACTION7}:
                context_stats = memory.keyboard_context_stats.get(
                    (candidate.action.name, coarse_cell_for_coord(actor_component.anchor))
                )
                if context_stats is not None:
                    keyboard_context_bonus = (
                        0.25 * context_stats.mean_reward
                        + 0.65 * context_stats.change_rate
                        + 0.35 * context_stats.level_gain_rate
                    )
                delta, confidence = memory.best_delta(candidate.action.name)
                if delta is not None and confidence >= 0.4:
                    projected_next_anchor = (
                        actor_component.anchor[0] + delta[0],
                        actor_component.anchor[1] + delta[1],
                    )
            transition_states = {
                next_sig
                for (state_sig, key), next_sig in level_memory.transitions.items()
                if key == candidate.key
            }
            transition_bonus = 0.2 * (1.0 - min(len(transition_states), 5) / 5.0)
            next_signature = level_memory.transitions.get((signature, candidate.key))
            future_frontier_bonus = 0.0
            if next_signature is not None:
                future_frontier_bonus = 0.18 * min(len(level_memory.untried_keys(next_signature)), 3)
            effect_diversity = memory.effect_diversity(candidate.key)
            subgoal_bonus = 0.0
            if (
                candidate.action == GameAction.ACTION6
                and candidate.data is not None
                and not level_memory.probe_queue
            ):
                subgoal_bonus = level_memory.subgoal_bonus(
                    coarse_cell_for_coord((int(candidate.data["y"]), int(candidate.data["x"])))
                )
            interact_bonus = 0.0
            if candidate.action == GameAction.ACTION5 and actor_component is not None:
                for target, target_score in interaction_targets[:4]:
                    distance = abs(actor_component.anchor[0] - target[0]) + abs(actor_component.anchor[1] - target[1])
                    if distance <= 1:
                        interact_bonus = max(interact_bonus, 1.4 * target_score)
                    elif distance == 2:
                        interact_bonus = max(interact_bonus, 0.6 * target_score)
                for feature in level_memory.active_target_contacts():
                    interact_bonus += 0.9 * memory.target_affordance_score(feature, candidate.action.name)
            repetition_penalty = 0.0
            if candidate.action == GameAction.ACTION6 and stats.attempts >= 2 and stats.level_gain_rate == 0.0:
                repetition_penalty = max(0.0, 0.65 - effect_diversity) * 0.8 * min(stats.attempts, 6)
            keyboard_stall_penalty = 0.0
            if candidate.action not in {GameAction.ACTION6, GameAction.ACTION7}:
                keyboard_stall_penalty = 0.45 * min(
                    level_memory.keyboard_no_change_streaks.get(candidate.action.name, 0),
                    3,
                )
            click_deferral = 0.0
            if (
                candidate.action == GameAction.ACTION6
                and keyboard_actions(available_actions)
                and level_memory.keyboard_control_confidence < 0.75
                and level_memory.step_index < 8
            ):
                click_deferral = 1.5
            control_repeat_penalty = 0.0
            if actor_component is not None and projected_next_anchor is not None:
                control_repeat_penalty = self._control_repeat_penalty(
                    action_key=candidate.key,
                    current_anchor=actor_component.anchor,
                    next_anchor=projected_next_anchor,
                    level_memory=level_memory,
                    interaction_targets=interaction_targets,
                    pressure=pressure,
                )
            mechanic_bonus = self._candidate_mechanic_bonus(
                candidate,
                frame,
                memory,
                level_memory,
                actor_component=actor_component,
                interaction_targets=interaction_targets,
            )
            tie_breaker = key_stability_score(candidate.key)
            score = (
                stats.mean_reward
                + 0.6 * stats.level_gain_rate
                + 0.35 * stats.change_rate
                + novel_bonus
                + stall_penalty
                + motion_bonus
                + keyboard_context_bonus
                + transition_bonus
                + future_frontier_bonus
                + 0.45 * subgoal_bonus
                + interact_bonus
                + mechanic_bonus
                + 0.15 * effect_diversity
                - repetition_penalty
                - keyboard_stall_penalty
                - click_deferral
                - control_repeat_penalty
                + tie_breaker
            )
            if score > best_score:
                best_score = score
                best = candidate
                best_repetition_penalty = repetition_penalty
                best_click_deferral = click_deferral
                best_mechanic_bonus = mechanic_bonus
                best_control_repeat_penalty = control_repeat_penalty
        if best_repetition_penalty > 0.0:
            self.diagnostics.repeated_action6_penalty_events += 1
        if best_click_deferral > 0.0:
            self.diagnostics.mixed_action_click_deferrals += 1
        if best_control_repeat_penalty > 0.0:
            self.diagnostics.control_repeat_suppressions += 1
        if best_mechanic_bonus > 0.0:
            self.diagnostics.mechanic_hint_applications += 1
            if level_memory.mechanic_hint is not None:
                if level_memory.mechanic_hint.source == "QWEN":
                    self.diagnostics.qwen_hint_applications += 1
                elif level_memory.mechanic_hint.source == "SYMBOLIC":
                    self.diagnostics.symbolic_hint_applications += 1
        return best

    def _pop_macro_candidate(
        self,
        signature: bytes,
        candidates: tuple[ActionCandidate, ...],
        level_memory: LevelMemory,
    ) -> ActionCandidate | None:
        if not level_memory.macro_queue:
            return None
        if (
            level_memory.macro_source_step >= 0
            and level_memory.step_index - level_memory.macro_source_step > 2
        ):
            level_memory.macro_queue.clear()
            level_memory.macro_source_step = -1
            level_memory.pending_option_plan = None
            level_memory.clear_control_commit()
            return None
        current_by_key = {candidate.key: candidate for candidate in candidates}
        while level_memory.macro_queue:
            queued = level_memory.macro_queue.popleft()
            candidate = current_by_key.get(queued.key)
            if candidate is None:
                continue
            if level_memory.has_tried(signature, candidate.key):
                continue
            if not level_memory.macro_queue:
                level_memory.macro_source_step = -1
                if level_memory.control_commit_steps_remaining <= 1:
                    level_memory.clear_control_commit()
            return candidate
        level_memory.macro_source_step = -1
        level_memory.pending_option_plan = None
        level_memory.clear_control_commit()
        return None

    def _seed_macro_bundle(
        self,
        frame: np.ndarray,
        available_actions: tuple[GameAction, ...],
        candidates: tuple[ActionCandidate, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        chosen: ActionCandidate,
        chosen_kind: str | None,
    ) -> None:
        hint = level_memory.mechanic_hint
        if hint is None or hint.source != "SYMBOLIC" or hint.confidence < 0.82:
            return
        if chosen_kind != "KEYBOARD":
            return
        if chosen.action in {GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7}:
            return
        if level_memory.macro_queue:
            return
        if not (4 <= level_memory.step_index <= 12):
            return

        follow_ups: list[ActionCandidate] = []
        if hint.mode in {"INTERACT", "MIXED"} and hint.goal in {"CONTACT", "COLLECT", "TOGGLE"}:
            interact_candidate, interact_score, _ = self._planned_interact_decision(
                frame,
                candidates,
                memory,
                level_memory,
                actor_component=resolved_actor_component(frame, memory, level_memory),
            )
            if (
                interact_candidate is not None
                and interact_candidate.action == GameAction.ACTION5
                and GameAction.ACTION5 in available_actions
                and interact_score >= 1.05
            ):
                follow_ups.append(interact_candidate)

        keyboard_stall = level_memory.keyboard_no_change_streaks.get(chosen.action.name, 0)
        if (
            not follow_ups
            and hint.mode in {"MOVE", "INTERACT", "MIXED"}
            and keyboard_stall == 0
            and level_memory.step_index - level_memory.last_changed_step <= 3
        ):
            follow_ups.append(chosen)

        if not follow_ups:
            return

        level_memory.macro_queue.extend(follow_ups)
        level_memory.macro_source_step = level_memory.step_index
        self.diagnostics.macro_bundle_injections += 1

    def _seed_control_commit_bundle(
        self,
        frame: np.ndarray,
        available_actions: tuple[GameAction, ...],
        candidates: tuple[ActionCandidate, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        chosen: ActionCandidate,
        *,
        control_score: float,
        control_margin: float,
        actor_component: FrameComponent | None,
    ) -> None:
        if actor_component is None:
            return
        if chosen.action in {GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7}:
            return
        if level_memory.macro_queue:
            return
        if not (4 <= level_memory.step_index <= 14):
            return
        if control_score < 1.12 or control_margin < 0.12:
            return
        if level_memory.keyboard_no_change_streaks.get(chosen.action.name, 0) > 0:
            return
        delta, confidence = memory.best_delta(chosen.action.name)
        if delta is None or confidence < 0.62:
            return
        targets = self._interaction_targets(frame, memory, level_memory, actor_component)
        if not targets:
            return
        current_target, current_target_score = targets[0]
        second_target_score = targets[1][1] if len(targets) > 1 else 0.0
        current_distance = abs(actor_component.anchor[0] - current_target[0]) + abs(actor_component.anchor[1] - current_target[1])
        next_anchor = (actor_component.anchor[0] + delta[0], actor_component.anchor[1] + delta[1])
        if not (0 <= next_anchor[0] < frame.shape[0] and 0 <= next_anchor[1] < frame.shape[1]):
            return
        next_distance = abs(next_anchor[0] - current_target[0]) + abs(next_anchor[1] - current_target[1])
        if next_distance >= current_distance:
            return
        target_margin = current_target_score - second_target_score
        if current_distance > 2 and current_target_score < 1.35:
            self.diagnostics.target_value_commit_blocks += 1
            return
        if len(targets) > 1 and current_target_score < 1.75 and target_margin < 0.12:
            self.diagnostics.target_value_commit_blocks += 1
            return

        follow_ups: list[ActionCandidate] = []
        allow_interact = False
        if next_distance <= 1 and GameAction.ACTION5 in available_actions:
            interact_candidate, interact_score, _ = self._planned_interact_decision(
                frame,
                candidates,
                memory,
                level_memory,
                actor_component=actor_component,
            )
            if interact_candidate is not None and interact_candidate.action == GameAction.ACTION5 and interact_score >= 1.0:
                follow_ups.append(interact_candidate)
                allow_interact = True
        elif next_distance > 1:
            projected_anchor = next_anchor
            projected_distance = next_distance
            for _ in range(2):
                next_projected = (projected_anchor[0] + delta[0], projected_anchor[1] + delta[1])
                if not (0 <= next_projected[0] < frame.shape[0] and 0 <= next_projected[1] < frame.shape[1]):
                    break
                next_projected_distance = (
                    abs(next_projected[0] - current_target[0]) + abs(next_projected[1] - current_target[1])
                )
                if next_projected_distance > projected_distance:
                    break
                follow_ups.append(chosen)
                projected_anchor = next_projected
                projected_distance = next_projected_distance
                if projected_distance <= 1 and GameAction.ACTION5 in available_actions:
                    interact_candidate, interact_score, _ = self._planned_interact_decision(
                        frame,
                        candidates,
                        memory,
                        level_memory,
                        actor_component=actor_component,
                    )
                    if (
                        interact_candidate is not None
                        and interact_candidate.action == GameAction.ACTION5
                        and interact_score >= 1.0
                    ):
                        follow_ups.append(interact_candidate)
                        allow_interact = True
                    break

        if not follow_ups and next_distance < current_distance:
            follow_ups.append(chosen)

        if not follow_ups:
            return

        self.diagnostics.target_value_commit_gates += 1
        level_memory.macro_queue.extend(follow_ups)
        level_memory.macro_source_step = level_memory.step_index
        level_memory.start_control_commit(
            primary_action=chosen.action.name,
            target=current_target,
            last_distance=current_distance,
            steps_remaining=1 + len(follow_ups),
            allow_interact=allow_interact,
        )
        self.diagnostics.macro_bundle_injections += 1
        self.diagnostics.control_commit_injections += 1

    def _recent_event_count(
        self,
        level_memory: LevelMemory,
        fragment: str,
        *,
        window: int = 4,
    ) -> int:
        if window <= 0:
            return 0
        events = list(level_memory.recent_events)[-window:]
        return sum(1 for event in events if fragment in event)

    def _update_control_commit_after_transition(
        self,
        level_memory: LevelMemory,
        executed: ActionCandidate,
        *,
        current_actor_component: FrameComponent | None,
        next_actor_component: FrameComponent | None,
        changed: bool,
        level_gain: int,
    ) -> None:
        if level_memory.control_commit_steps_remaining <= 0:
            return
        target = level_memory.control_commit_target
        primary_action = level_memory.control_commit_primary_action
        if target is None or primary_action is None:
            level_memory.clear_control_commit()
            return

        allowed_interact = level_memory.control_commit_allow_interact and executed.action == GameAction.ACTION5
        if executed.action.name != primary_action and not allowed_interact:
            level_memory.macro_queue.clear()
            level_memory.clear_control_commit()
            self.diagnostics.control_commit_aborts += 1
            return

        progress = level_gain > 0
        next_distance = level_memory.control_commit_last_distance
        if not progress and current_actor_component is not None:
            current_distance = abs(current_actor_component.anchor[0] - target[0]) + abs(current_actor_component.anchor[1] - target[1])
            next_distance = current_distance
            if next_actor_component is not None:
                next_distance = (
                    abs(next_actor_component.anchor[0] - target[0]) + abs(next_actor_component.anchor[1] - target[1])
                )
            if executed.action == GameAction.ACTION5:
                progress = changed and current_distance <= 1
            else:
                progress = next_distance < current_distance or (changed and next_distance <= current_distance - 0)
        elif not progress and changed:
            progress = True

        if not progress:
            level_memory.note_blocked_target(target, weight=1.0)
            level_memory.macro_queue.clear()
            level_memory.clear_control_commit()
            self.diagnostics.control_commit_aborts += 1
            return

        if next_distance is not None:
            level_memory.control_commit_last_distance = next_distance
        level_memory.control_commit_steps_remaining = max(level_memory.control_commit_steps_remaining - 1, 0)
        self.diagnostics.control_commit_validations += 1
        if level_memory.control_commit_steps_remaining <= 0:
            level_memory.clear_control_commit()

    def _pop_probe_candidate(
        self,
        frame: np.ndarray,
        signature: bytes,
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
    ) -> ActionCandidate | None:
        if not level_memory.probe_queue:
            return None
        untried = [
            (index, probe)
            for index, probe in enumerate(level_memory.probe_queue)
            if not level_memory.has_tried(signature, probe.key)
        ]
        if not untried:
            level_memory.probe_queue.clear()
            return None
        hint = level_memory.mechanic_hint
        if hint is None or hint.confidence <= 0.0:
            chosen_index = untried[0][0]
        else:
            actor_component = component_by_actor_prior(frame, memory)
            interaction_targets: tuple[tuple[Coord, float], ...] = ()
            if actor_component is not None:
                interaction_targets = self._interaction_targets(frame, memory, level_memory, actor_component)
            chosen_index = untried[0][0]
            best_score = float("-inf")
            for index, probe in untried:
                stats = memory.stat_for(probe.key)
                score = (
                    self._candidate_mechanic_bonus(
                        probe,
                        frame,
                        memory,
                        level_memory,
                        actor_component=actor_component,
                        interaction_targets=interaction_targets,
                    )
                    + 0.15 * max(stats.mean_reward, 0.0)
                    + 0.35 * stats.change_rate
                    + 0.6 * stats.level_gain_rate
                )
                control_bonus = self._control_map_candidate_score(
                    probe,
                    frame,
                    memory,
                    level_memory,
                    actor_component=actor_component,
                    interaction_targets=interaction_targets,
                )
                if control_bonus != float("-inf"):
                    score += 0.9 * control_bonus
                if probe.action == GameAction.ACTION6 and probe.data is not None:
                    score += 0.25 * level_memory.subgoal_bonus(
                        coarse_cell_for_coord((int(probe.data["y"]), int(probe.data["x"])))
                    )
                if score > best_score:
                    best_score = score
                    chosen_index = index
        chosen = level_memory.probe_queue[chosen_index]
        level_memory.probe_queue = deque(
            probe for index, probe in enumerate(level_memory.probe_queue) if index != chosen_index
        )
        return chosen

    def _maybe_refresh_mechanic_hint(
        self,
        frame: np.ndarray,
        candidates: tuple[ActionCandidate, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
    ) -> None:
        stall_steps = level_memory.step_index - level_memory.last_changed_step
        if level_memory.step_index < 4:
            return
        current_signature = frame_signature(frame)
        pressure = level_memory.loop_pressure(current_signature)
        if (
            level_memory.mechanic_hint is not None
            and stall_steps < 3
            and pressure - level_memory.mechanic_hint_pressure < 2
        ):
            return
        if stall_steps < 2 and level_memory.step_index < 7:
            return

        def apply_hint(hint: MechanicHint, *, qwen_refresh: bool) -> None:
            level_memory.mechanic_hint = hint
            level_memory.mechanic_hint_step = level_memory.step_index
            level_memory.mechanic_hint_pressure = pressure
            level_memory.mechanic_hint_raw = hint.raw_text
            if qwen_refresh:
                self.diagnostics.qwen_hint_refreshes += 1
            else:
                self.diagnostics.symbolic_hint_refreshes += 1

        symbolic_hint = self._symbolic_mechanic_hint(frame, candidates, memory, level_memory)
        if self.mechanic_advisor is None:
            hint = symbolic_hint
            if hint is None:
                return
            apply_hint(hint, qwen_refresh=False)
            return
        if (
            symbolic_hint is not None
            and symbolic_hint.confidence >= 0.8
            and pressure < 3
            and stall_steps < 5
        ):
            apply_hint(symbolic_hint, qwen_refresh=False)
            return
        if level_memory.mechanic_qwen_calls_used >= self.mechanic_advisor_budget_per_level:
            if symbolic_hint is not None:
                apply_hint(symbolic_hint, qwen_refresh=False)
            return

        prompt = self._mechanic_hint_prompt(
            frame,
            candidates,
            memory,
            level_memory,
            symbolic_hint=symbolic_hint,
        )
        level_memory.mechanic_qwen_calls_used += 1
        self.diagnostics.qwen_calls += 1
        self.diagnostics.qwen_hint_calls += 1
        payload = self.mechanic_advisor.summarize_mechanic(
            prompt,
            allowed_modes=MECHANIC_MODES,
            allowed_goals=MECHANIC_GOALS,
            allowed_focuses=MECHANIC_FOCUSES,
        )
        if payload is None:
            if symbolic_hint is not None:
                apply_hint(symbolic_hint, qwen_refresh=False)
            return
        confidence = max(0.0, min(float(payload.get("confidence", 0.0)), 1.0))
        qwen_hint = MechanicHint(
            mode=str(payload.get("mode", "UNKNOWN")).upper(),
            goal=str(payload.get("goal", "UNKNOWN")).upper(),
            focus=str(payload.get("focus", "UNKNOWN")).upper(),
            confidence=confidence,
            source_step=level_memory.step_index,
            source="QWEN",
            raw_text=str(payload.get("raw_text", "")),
        )
        if (
            qwen_hint.mode == "UNKNOWN"
            and qwen_hint.goal == "UNKNOWN"
            and qwen_hint.focus == "UNKNOWN"
        ):
            if symbolic_hint is not None:
                apply_hint(symbolic_hint, qwen_refresh=False)
            return

        chosen_hint = symbolic_hint
        chosen_from_qwen = False
        if symbolic_hint is None:
            chosen_hint = qwen_hint
            chosen_from_qwen = True
        else:
            qwen_disagreement = any(
                (
                    getattr(qwen_hint, field) != "UNKNOWN"
                    and getattr(qwen_hint, field) != getattr(symbolic_hint, field)
                )
                for field in ("mode", "goal", "focus")
            )
            confidence_margin = 0.03 if pressure >= 4 else 0.06
            if (
                (
                    pressure >= 4
                    and qwen_disagreement
                    and qwen_hint.confidence >= 0.9
                )
                or (
                    qwen_hint.confidence >= symbolic_hint.confidence + confidence_margin
                )
                or (
                    qwen_disagreement
                    and qwen_hint.confidence >= symbolic_hint.confidence + 0.02
                )
            ):
                chosen_hint = qwen_hint
                chosen_from_qwen = True
        if chosen_hint is None:
            return
        apply_hint(chosen_hint, qwen_refresh=chosen_from_qwen)

    def _posterior_guided_decision(
        self,
        frame: np.ndarray,
        candidates: tuple[ActionCandidate, ...],
        available_actions: tuple[GameAction, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
    ) -> tuple[ActionCandidate | None, float, float, str | None]:
        hint = level_memory.mechanic_hint
        if (
            hint is None
            or hint.confidence < self.tuning.posterior_hint_min_confidence
            or level_memory.step_index < 4
        ):
            return None, 0.0, 0.0, None
        signature = frame_signature(frame)
        pressure = level_memory.loop_pressure(signature)
        actor_component = component_by_actor_prior(frame, memory)

        click_plan: tuple[ActionCandidate | None, float, float] = (None, 0.0, 0.0)
        if any(candidate.action == GameAction.ACTION6 for candidate in candidates):
            click_plan = self._planned_click_decision(frame, candidates, memory, level_memory)

        interact_plan: tuple[ActionCandidate | None, float, float] = (None, 0.0, 0.0)
        if actor_component is not None and hint.mode in {"INTERACT", "MIXED"}:
            interact_plan = self._planned_interact_decision(
                frame,
                candidates,
                memory,
                level_memory,
                actor_component=actor_component,
            )

        keyboard_plan: tuple[ActionCandidate | None, float, float] = (None, 0.0, 0.0)
        keyboard_modes = {"MOVE", "INTERACT", "MIXED", "CLICK"}
        if (
            actor_component is not None
            and keyboard_actions(available_actions)
            and hint.mode in keyboard_modes
        ):
            keyboard_plan = self._planned_rollout_decision(frame, available_actions, memory, level_memory)
            keyboard_plan = (
                keyboard_plan[0],
                keyboard_plan[1] + 0.06 * hint.confidence + 0.03 * min(pressure, 4),
                keyboard_plan[2],
            )

        click_candidate, click_score, click_margin = click_plan
        interact_candidate, interact_score, interact_margin = interact_plan
        keyboard_candidate, keyboard_score, keyboard_margin = keyboard_plan
        adjacent_interact_affordance = 0.0
        if interact_candidate is not None and actor_component is not None:
            adjacent_interact_affordance = max(
                (
                    memory.target_affordance_score(component.feature, interact_candidate.action.name)
                    for component in extract_components(frame)
                    if component.feature != actor_component.feature
                    and bounds_gap(actor_component.bounds, component.bounds) <= 1
                ),
                default=0.0,
            )

        late_click_detour = self._late_click_detour_active(
            available_actions=available_actions,
            actor_component=actor_component,
            level_memory=level_memory,
            click_candidate=click_candidate,
            click_score=click_score,
            keyboard_candidate=keyboard_candidate,
            keyboard_score=keyboard_score,
        )

        if hint.mode in {"CLICK", "UNDO"}:
            if hint.mode == "CLICK" and late_click_detour and keyboard_candidate is not None:
                self.diagnostics.late_click_detours += 1
                detour_score = max(keyboard_score, click_score - 0.02) + 0.08 * hint.confidence
                detour_margin = max(keyboard_margin, 0.08)
                return keyboard_candidate, detour_score, detour_margin, "KEYBOARD"
            return click_candidate, click_score, click_margin, "CLICK" if click_candidate is not None else None
        if hint.mode in {"MOVE", "INTERACT"}:
            if (
                hint.mode == "INTERACT"
                and interact_candidate is not None
                and (
                    interact_score >= keyboard_score - 0.02
                    or (
                        adjacent_interact_affordance >= 0.9
                        and level_memory.family_failure_count("MOVE") >= 2
                    )
                )
            ):
                return interact_candidate, interact_score, interact_margin, "KEYBOARD"
            return (
                keyboard_candidate,
                keyboard_score,
                keyboard_margin,
                "KEYBOARD" if keyboard_candidate is not None else None,
            )
        if hint.mode == "MIXED":
            if late_click_detour and keyboard_candidate is not None and click_candidate is not None:
                self.diagnostics.late_click_detours += 1
                detour_score = max(keyboard_score, click_score - 0.02) + 0.06 * hint.confidence
                detour_margin = max(keyboard_margin, 0.08)
                return keyboard_candidate, detour_score, detour_margin, "KEYBOARD"
            click_fail = level_memory.family_failure_count("CLICK")
            move_fail = level_memory.family_failure_count("MOVE")
            recent_click_change = any(
                event.startswith("ACTION6 ") and "changed=1" in event and "level_gain=0" in event
                for event in level_memory.recent_events
            )
            recent_keyboard_stall = any(
                event.startswith("ACTION")
                and not event.startswith("ACTION6 ")
                and "changed=0" in event
                and "level_gain=0" in event
                for event in level_memory.recent_events
            )
            if (
                interact_candidate is not None
                and recent_click_change
                and interact_score >= max(click_score, keyboard_score) - 0.04
            ):
                return interact_candidate, interact_score, interact_margin, "KEYBOARD"
            if (
                keyboard_candidate is not None
                and recent_click_change
                and keyboard_score >= click_score - 0.05
            ):
                return keyboard_candidate, keyboard_score, keyboard_margin, "KEYBOARD"
            if (
                click_candidate is not None
                and recent_keyboard_stall
                and move_fail >= click_fail + 2
                and click_score >= keyboard_score - 0.04
            ):
                return click_candidate, click_score, click_margin, "CLICK"
            if (
                keyboard_candidate is not None
                and click_fail >= move_fail + 2
                and keyboard_score >= click_score - 0.04
            ):
                return keyboard_candidate, keyboard_score, keyboard_margin, "KEYBOARD"
            if interact_candidate is not None and interact_score >= max(click_score, keyboard_score) + 0.02:
                return interact_candidate, interact_score, interact_margin, "KEYBOARD"
            if click_candidate is None:
                return keyboard_candidate, keyboard_score, keyboard_margin, "KEYBOARD" if keyboard_candidate else None
            if keyboard_candidate is None:
                return click_candidate, click_score, click_margin, "CLICK"
        if click_score >= keyboard_score + 0.05:
            return click_candidate, click_score, click_margin, "CLICK"
        return keyboard_candidate, keyboard_score, keyboard_margin, "KEYBOARD"
        return None, 0.0, 0.0, None

    def _should_suppress_stalled_posterior(
        self,
        *,
        posterior_kind: str | None,
        actor_component: FrameComponent | None,
        level_memory: LevelMemory,
        faststart_confident: bool,
    ) -> bool:
        if posterior_kind != "KEYBOARD" or actor_component is None or faststart_confident:
            return False
        if (
            level_memory.family_failure_count("MOVE")
            < self.tuning.posterior_keyboard_stall_failure_count
        ):
            return False
        recent_keyboard_stalls = sum(
            1
            for event in level_memory.recent_events
            if event.startswith("ACTION")
            and not event.startswith("ACTION5 ")
            and not event.startswith("ACTION6 ")
            and not event.startswith("ACTION7 ")
            and "changed=0" in event
            and "level_gain=0" in event
        )
        return recent_keyboard_stalls >= self.tuning.posterior_keyboard_stall_event_count

    def _symbolic_mechanic_hint(
        self,
        frame: np.ndarray,
        candidates: tuple[ActionCandidate, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
    ) -> MechanicHint | None:
        components = extract_components(frame)
        if not components:
            return None
        current_signature = frame_signature(frame)
        actor_component = component_by_actor_prior(frame, memory)
        click_candidates = tuple(candidate for candidate in candidates if candidate.action == GameAction.ACTION6)
        keyboard_candidates = keyboard_actions(tuple(candidate.action for candidate in candidates))
        move_candidates = tuple(action for action in keyboard_candidates if action != GameAction.ACTION5)
        interact_available = GameAction.ACTION5 in keyboard_candidates
        undo_available = any(candidate.action == GameAction.ACTION7 for candidate in candidates)
        non_actor_components = [
            component for component in components if actor_component is None or component.feature != actor_component.feature
        ]
        ranked_components = sorted(
            non_actor_components,
            key=lambda component: interaction_target_priority(component, memory) + min(component.size, 16) / 20.0,
            reverse=True,
        )
        top_component = ranked_components[0] if ranked_components else components[0]
        top_priority = interaction_target_priority(top_component, memory)
        top_click_stats = memory.click_stats.get(top_component.feature)
        top_click_change = top_click_stats.change_rate if top_click_stats is not None else 0.0
        top_click_level = top_click_stats.level_gain_rate if top_click_stats is not None else 0.0
        top_click_reward = max(top_click_stats.mean_reward, 0.0) if top_click_stats is not None else 0.0
        click_hotspot = next(
            (
                score
                for cell, score in hotspot_cells(memory)
                if cell == coarse_cell_for_coord(top_component.anchor)
            ),
            0.0,
        )
        top_color_mass = int(np.count_nonzero(frame == top_component.color))
        rarity = top_component.size / max(top_color_mass, top_component.size)

        move_stats = [memory.stat_for((action.name, None)) for action in move_candidates]
        move_change = max((stats.change_rate for stats in move_stats), default=0.0)
        move_level = max((stats.level_gain_rate for stats in move_stats), default=0.0)
        move_reward = max((max(stats.mean_reward, 0.0) for stats in move_stats), default=0.0)
        interact_stats = memory.stat_for((GameAction.ACTION5.name, None)) if interact_available else ActionStats()
        undo_stats = memory.stat_for((GameAction.ACTION7.name, None)) if undo_available else ActionStats()

        actor_target_distance = 99
        aligned_with_target = False
        if actor_component is not None and ranked_components:
            actor_target_distance = manhattan_to_bounds(actor_component.anchor, top_component.bounds)
            aligned_with_target = (
                actor_component.anchor[0] in range(top_component.bounds[0], top_component.bounds[2] + 1)
                or actor_component.anchor[1] in range(top_component.bounds[1], top_component.bounds[3] + 1)
            )
        click_fail = level_memory.family_failure_count("CLICK")
        move_fail = level_memory.family_failure_count("MOVE")
        interact_fail = level_memory.family_failure_count("INTERACT")
        undo_fail = level_memory.family_failure_count("UNDO")
        revisit_pressure = max(0, level_memory.state_visit_counts.get(current_signature, 0) - 1)
        loop_pressure = level_memory.loop_pressure(current_signature)
        recent_click_dead_ends = self._recent_event_count(
            level_memory,
            "ACTION6 changed=1 level_gain=0",
        )

        mode_scores: dict[MechanicMode, float] = {"UNKNOWN": 0.0}
        if click_candidates:
            mode_scores["CLICK"] = (
                0.18
                + 0.85 * top_click_change
                + 1.35 * top_click_level
                + 0.18 * top_click_reward
                + 0.18 * click_hotspot
                + (0.12 if not move_candidates else 0.0)
                - 0.16 * click_fail
            )
        if move_candidates:
            mode_scores["MOVE"] = (
                0.15
                + 0.75 * move_change
                + 1.2 * move_level
                + 0.22 * move_reward
                + 0.45 * level_memory.keyboard_control_confidence
                + (0.18 if actor_component is not None else 0.0)
                - 0.14 * move_fail
            )
        if interact_available:
            close_bonus = 0.4 if actor_target_distance <= 1 else 0.18 if actor_target_distance == 2 else 0.0
            mode_scores["INTERACT"] = (
                0.08
                + 0.7 * interact_stats.change_rate
                + 1.25 * interact_stats.level_gain_rate
                + 0.25 * max(interact_stats.mean_reward, 0.0)
                + close_bonus
                + 0.16 * min(click_fail + move_fail, 4)
                - 0.12 * interact_fail
            )
        if undo_available:
            stall_steps = level_memory.step_index - level_memory.last_changed_step
            mode_scores["UNDO"] = (
                0.05
                + 0.25 * undo_stats.change_rate
                + 0.3 * undo_stats.level_gain_rate
                + (0.22 if stall_steps >= 5 else 0.0)
                + 0.16 * min(recent_click_dead_ends, 3)
                + 0.12 * min(click_fail + move_fail, 5)
                + 0.08 * min(revisit_pressure, 3)
                - 0.1 * undo_fail
            )
        single_mode_scores = [(mode, score) for mode, score in mode_scores.items() if mode != "UNKNOWN"]
        single_mode_scores.sort(key=lambda item: item[1], reverse=True)
        if not single_mode_scores or single_mode_scores[0][1] < 0.22:
            return None
        best_mode, best_mode_score = single_mode_scores[0]
        second_mode_score = single_mode_scores[1][1] if len(single_mode_scores) > 1 else 0.0
        if len(single_mode_scores) > 1 and single_mode_scores[1][1] >= 0.62 and best_mode_score - second_mode_score <= 0.18:
            mode = "MIXED"
        else:
            mode = best_mode
        if (
            undo_available
            and (
                (click_fail + move_fail >= 5 and revisit_pressure >= 1)
                or (recent_click_dead_ends >= 2 and click_fail >= 2)
            )
            and mode_scores.get("UNDO", 0.0) >= 0.38
        ):
            mode = "UNDO"
        if (
            interact_available
            and actor_component is not None
            and actor_target_distance <= 1
            and click_fail + move_fail >= 4
            and mode_scores.get("INTERACT", 0.0) >= 0.4
        ):
            mode = "INTERACT"

        goal_scores: dict[MechanicGoal, float] = {
            "CONTACT": 0.0,
            "ALIGN": 0.0,
            "COLLECT": 0.0,
            "TOGGLE": 0.0,
            "CLEAR": 0.0,
            "UNKNOWN": 0.0,
        }
        if click_candidates:
            goal_scores["CLEAR"] = 0.18 + 0.75 * mode_scores.get("CLICK", 0.0) + 0.22 * min(top_component.size, 24) / 24.0
            goal_scores["TOGGLE"] = 0.12 + 0.42 * mode_scores.get("CLICK", 0.0) + (0.18 if undo_available else 0.0)
        if actor_component is not None:
            goal_scores["CONTACT"] = (
                0.12
                + 0.6 * mode_scores.get("MOVE", 0.0)
                + 0.45 * mode_scores.get("INTERACT", 0.0)
                + (0.28 if actor_target_distance > 1 else 0.0)
            )
            goal_scores["ALIGN"] = (
                0.08
                + 0.45 * mode_scores.get("MOVE", 0.0)
                + (0.32 if aligned_with_target else 0.0)
            )
            goal_scores["COLLECT"] = (
                0.08
                + 0.35 * mode_scores.get("MOVE", 0.0)
                + 0.4 * mode_scores.get("INTERACT", 0.0)
                + 0.4 * rarity
                + (0.22 if actor_target_distance <= 2 else 0.0)
            )
            goal_scores["TOGGLE"] += 0.24 * mode_scores.get("INTERACT", 0.0) + (
                0.28 if actor_target_distance <= 1 else 0.0
            )
        else:
            goal_scores["COLLECT"] = 0.04 + 0.25 * rarity + 0.15 * mode_scores.get("CLICK", 0.0)
        goal, best_goal_score = max(goal_scores.items(), key=lambda item: item[1])

        center = (frame.shape[0] / 2.0, frame.shape[1] / 2.0)
        center_score = max(
            0.0,
            1.0 - (
                abs(top_component.anchor[0] - center[0]) + abs(top_component.anchor[1] - center[1])
            ) / (frame.shape[0] + frame.shape[1]),
        )
        focus_scores: dict[MechanicFocus, float] = {
            "RARE_COLOR": rarity,
            "LARGE_OBJECT": min(top_component.size, 16) / 16.0,
            "SMALL_OBJECT": max(0.0, (6 - min(top_component.size, 6)) / 6.0),
            "HOTSPOT": click_hotspot,
            "CENTER": center_score,
            "MOVING_OBJECT": min(1.0, 0.25 + level_memory.keyboard_control_confidence + 0.4 * move_change),
            "UNKNOWN": 0.0,
        }
        if actor_component is None:
            focus_scores["MOVING_OBJECT"] *= 0.45
        focus, best_focus_score = max(focus_scores.items(), key=lambda item: item[1])

        dominance = max(0.0, best_mode_score - second_mode_score)
        confidence = min(
            0.94,
            0.2
            + 0.22 * min(best_mode_score, 1.5)
            + 0.2 * min(best_goal_score, 1.2)
            + 0.12 * min(best_focus_score, 1.0)
            + 0.18 * min(dominance, 0.8),
        )
        confidence = min(0.96, confidence + 0.02 * min(loop_pressure, 6))
        if confidence < 0.32:
            return None
        raw_text = (
            f"symbolic mode={mode} goal={goal} focus={focus} "
            f"mode_score={best_mode_score:.2f} goal_score={best_goal_score:.2f} focus_score={best_focus_score:.2f}"
        )
        return MechanicHint(
            mode=mode,
            goal=goal,
            focus=focus,
            confidence=confidence,
            source_step=level_memory.step_index,
            source="SYMBOLIC",
            raw_text=raw_text,
        )

    def _mechanic_hint_prompt(
        self,
        frame: np.ndarray,
        candidates: tuple[ActionCandidate, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        *,
        symbolic_hint: MechanicHint | None = None,
    ) -> str:
        stall_steps = level_memory.step_index - level_memory.last_changed_step
        actor_component = component_by_actor_prior(frame, memory)
        click_candidates = tuple(candidate for candidate in candidates if candidate.action == GameAction.ACTION6)
        keyboard_candidates = tuple(
            candidate for candidate in candidates if candidate.action not in {GameAction.ACTION6, GameAction.ACTION7}
        )
        components = sorted(
            extract_components(frame),
            key=lambda component: interaction_target_priority(component, memory) + min(component.size, 16) / 16.0,
            reverse=True,
        )[:6]
        color_counts = {component.color: 0 for component in components}
        for component in extract_components(frame):
            color_counts[component.color] = color_counts.get(component.color, 0) + component.size

        prompt_lines = [
            "Infer a soft mechanic hypothesis for an unseen arcade level.",
            "Do not choose an action. Summarize the likely interaction family only.",
            "Return exactly four lines:",
            f"MODE=<{ '|'.join(MECHANIC_MODES) }>",
            f"GOAL=<{ '|'.join(MECHANIC_GOALS) }>",
            f"FOCUS=<{ '|'.join(MECHANIC_FOCUSES) }>",
            "CONFIDENCE=<0.0-1.0>",
            f"step={level_memory.step_index}",
            f"stall_steps={stall_steps}",
            f"keyboard_actions={','.join(candidate.action.name for candidate in keyboard_candidates) or 'none'}",
            f"click_candidates={len(click_candidates)}",
        ]
        if symbolic_hint is None:
            prompt_lines.append("symbolic_baseline=none")
        else:
            prompt_lines.append(
                "symbolic_baseline="
                f"mode:{symbolic_hint.mode} goal:{symbolic_hint.goal} focus:{symbolic_hint.focus} "
                f"conf:{symbolic_hint.confidence:.2f}"
            )
        if actor_component is None:
            prompt_lines.append("known_actor=unknown")
        else:
            prompt_lines.append(
                f"known_actor=anchor:{actor_component.anchor} cell:{coarse_cell_for_coord(actor_component.anchor)}"
            )
        active_target_cells = level_memory.active_target_cells()
        if active_target_cells:
            prompt_lines.append(
                "active_target_cells=" + ",".join(f"{row}:{col}" for row, col in active_target_cells[:6])
            )
        else:
            prompt_lines.append("active_target_cells=none")
        if level_memory.family_no_progress_counts:
            prompt_lines.append(
                "family_failures="
                + ",".join(
                    f"{family}:{count}"
                    for family, count in sorted(level_memory.family_no_progress_counts.items())
                    if count > 0
                )
            )
        else:
            prompt_lines.append("family_failures=none")
        prompt_lines.append("recent_events:")
        if level_memory.recent_events:
            prompt_lines.extend(f"- {event}" for event in level_memory.recent_events)
        else:
            prompt_lines.append("- none")
        prompt_lines.append("action_stats:")
        for candidate in keyboard_candidates:
            stats = memory.stat_for(candidate.key)
            delta, confidence = memory.best_delta(candidate.action.name)
            prompt_lines.append(
                f"- {candidate.action.name} reward={stats.mean_reward:.2f} change={stats.change_rate:.2f} "
                f"level={stats.level_gain_rate:.2f} delta={delta} conf={confidence:.2f} "
                f"streak={level_memory.keyboard_no_change_streaks.get(candidate.action.name, 0)}"
            )
        if click_candidates:
            click_stats = [memory.stat_for(candidate.key) for candidate in click_candidates[:4]]
            avg_click_reward = sum(stats.mean_reward for stats in click_stats) / max(len(click_stats), 1)
            avg_click_change = sum(stats.change_rate for stats in click_stats) / max(len(click_stats), 1)
            prompt_lines.append(
                f"click_summary: sample={min(len(click_candidates), 4)} "
                f"avg_reward={avg_click_reward:.2f} avg_change={avg_click_change:.2f}"
            )
        prompt_lines.append("component_summary:")
        for component in components:
            stats = memory.click_stats.get(component.feature)
            prompt_lines.append(
                f"- color={component.color} size={component.size} anchor={component.anchor} "
                f"cell={coarse_cell_for_coord(component.anchor)} "
                f"color_mass={color_counts.get(component.color, component.size)} "
                f"click_change={(stats.change_rate if stats is not None else 0.0):.2f} "
                f"click_level={(stats.level_gain_rate if stats is not None else 0.0):.2f} "
                f"priority={interaction_target_priority(component, memory):.2f}"
            )
        return "\n".join(prompt_lines)

    def _mechanic_focus_bonus(
        self,
        component: FrameComponent,
        frame: np.ndarray,
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
    ) -> float:
        hint = level_memory.mechanic_hint
        if hint is None or hint.confidence <= 0.0:
            return 0.0
        scale = 0.45 * hint.confidence
        focus = hint.focus
        if focus == "LARGE_OBJECT":
            return scale * min(component.size, 16) / 16.0
        if focus == "SMALL_OBJECT":
            return scale * max(0.0, (6 - min(component.size, 6)) / 6.0)
        if focus == "CENTER":
            center = (frame.shape[0] / 2.0, frame.shape[1] / 2.0)
            dist = abs(component.anchor[0] - center[0]) + abs(component.anchor[1] - center[1])
            return scale * max(0.0, 1.0 - dist / (frame.shape[0] + frame.shape[1]))
        if focus == "HOTSPOT":
            cell = coarse_cell_for_coord(component.anchor)
            return scale * next((score for hotspot_cell, score in hotspot_cells(memory) if hotspot_cell == cell), 0.0)
        if focus == "MOVING_OBJECT":
            return scale * min(1.0, 0.25 + interaction_target_priority(component, memory))
        if focus == "RARE_COLOR":
            color_mass = int(np.count_nonzero(frame == component.color))
            if color_mass <= 0:
                return 0.0
            rarity = component.size / max(color_mass, component.size)
            return scale * min(1.0, rarity)
        return 0.0

    def _candidate_mechanic_bonus(
        self,
        candidate: ActionCandidate,
        frame: np.ndarray,
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        *,
        actor_component: FrameComponent | None,
        interaction_targets: tuple[tuple[Coord, float], ...],
    ) -> float:
        hint = level_memory.mechanic_hint
        if hint is None or hint.confidence <= 0.0:
            return 0.0
        scale = 0.75 * hint.confidence
        bonus = 0.0
        if hint.mode == "CLICK" and candidate.action == GameAction.ACTION6:
            bonus += 0.85 * scale
        elif hint.mode == "MOVE" and candidate.action not in {GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7}:
            bonus += 0.55 * scale
        elif hint.mode == "INTERACT" and candidate.action == GameAction.ACTION5:
            bonus += 0.8 * scale
        elif hint.mode == "UNDO" and candidate.action == GameAction.ACTION7:
            bonus += 0.4 * scale
        elif hint.mode == "MIXED":
            if candidate.action == GameAction.ACTION6:
                bonus += 0.2 * scale
            elif candidate.action not in {GameAction.ACTION6, GameAction.ACTION7}:
                bonus += 0.18 * scale

        top_targets = interaction_targets[:3]
        if candidate.action == GameAction.ACTION6 and candidate.data is not None and top_targets:
            click_coord = (int(candidate.data["y"]), int(candidate.data["x"]))
            best_target_distance = min(
                abs(click_coord[0] - target[0]) + abs(click_coord[1] - target[1]) for target, _ in top_targets
            )
            if hint.goal in {"CLEAR", "COLLECT", "TOGGLE"}:
                bonus += scale * max(0.0, 0.45 - 0.08 * best_target_distance)
        if actor_component is None:
            return bonus
        if candidate.action not in {GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7} and top_targets:
            delta, confidence = memory.best_delta(candidate.action.name)
            if delta is not None:
                next_anchor = (actor_component.anchor[0] + delta[0], actor_component.anchor[1] + delta[1])
                current_distance = min(
                    abs(actor_component.anchor[0] - target[0]) + abs(actor_component.anchor[1] - target[1])
                    for target, _ in top_targets
                )
                next_distance = min(
                    abs(next_anchor[0] - target[0]) + abs(next_anchor[1] - target[1]) for target, _ in top_targets
                )
                if hint.goal in {"CONTACT", "ALIGN", "COLLECT", "TOGGLE"} and next_distance < current_distance:
                    bonus += scale * min(0.55, 0.18 * (current_distance - next_distance)) * max(confidence, 0.35)
        if candidate.action == GameAction.ACTION5 and top_targets:
            distance = min(
                abs(actor_component.anchor[0] - target[0]) + abs(actor_component.anchor[1] - target[1])
                for target, _ in top_targets
            )
            if hint.goal in {"TOGGLE", "COLLECT", "CONTACT"} and distance <= 1:
                bonus += 0.7 * scale
            elif hint.goal in {"TOGGLE", "COLLECT"} and distance > 2:
                bonus -= 0.15 * scale
        return bonus

    def _control_map_candidate_score(
        self,
        candidate: ActionCandidate,
        frame: np.ndarray,
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        *,
        actor_component: FrameComponent | None,
        interaction_targets: tuple[tuple[Coord, float], ...],
    ) -> float:
        if actor_component is None or candidate.action in {GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7}:
            return float("-inf")
        delta, confidence = memory.best_delta(candidate.action.name)
        if delta is None or confidence < self.tuning.control_map_delta_confidence_min:
            return float("-inf")

        next_anchor = (actor_component.anchor[0] + delta[0], actor_component.anchor[1] + delta[1])
        if not (0 <= next_anchor[0] < frame.shape[0] and 0 <= next_anchor[1] < frame.shape[1]):
            return -1.0

        pressure = level_memory.loop_pressure(frame_signature(frame))
        score = 0.55 * confidence
        if next_anchor not in level_memory.actor_positions:
            score += 0.32
        else:
            score -= 0.08

        stats = memory.stat_for(candidate.key)
        score += 0.18 * max(stats.change_rate, 0.0)
        score += 0.24 * max(stats.level_gain_rate, 0.0)

        active_target_cells = level_memory.active_target_cells()
        actor_cell = coarse_cell_for_coord(actor_component.anchor)
        next_cell = coarse_cell_for_coord(next_anchor)
        if interaction_targets:
            best_target_bonus = float("-inf")
            target_cells = frozenset(coarse_cell_for_coord(target) for target, _ in interaction_targets[:4])
            blocked_cells = coarse_blocked_cells_for_components(
                frame,
                actor_feature=actor_component.feature,
                target_cells=target_cells,
            )
            for target, target_score in interaction_targets[:4]:
                current_distance = abs(actor_component.anchor[0] - target[0]) + abs(actor_component.anchor[1] - target[1])
                next_distance = abs(next_anchor[0] - target[0]) + abs(next_anchor[1] - target[1])
                pixel_improvement = current_distance - next_distance
                target_cell = coarse_cell_for_coord(target)
                current_cell_distance = abs(actor_cell[0] - target_cell[0]) + abs(actor_cell[1] - target_cell[1])
                next_cell_distance = abs(next_cell[0] - target_cell[0]) + abs(next_cell[1] - target_cell[1])
                cell_improvement = current_cell_distance - next_cell_distance
                current_path = coarse_path_distance(actor_cell, target_cell, blocked=blocked_cells - {actor_cell, target_cell})
                next_path = coarse_path_distance(next_cell, target_cell, blocked=blocked_cells - {next_cell, target_cell})
                path_improvement = current_path - next_path
                path_detour = max(0, current_path - current_cell_distance)
                bonus = 0.28 * target_score / (1.0 + next_cell_distance)
                if current_cell_distance <= 1:
                    if pixel_improvement > 0:
                        bonus += 1.8 * target_score * min(pixel_improvement / 8.0, 2.0)
                    elif pixel_improvement < 0:
                        bonus -= 0.72 * min((-pixel_improvement) / 8.0, 2.0)
                if cell_improvement > 0:
                    bonus += 0.34 * target_score * min(cell_improvement, 2)
                elif cell_improvement < 0:
                    bonus -= 0.14 * min(-cell_improvement, 2)
                if path_improvement > 0:
                    bonus += (0.24 + 0.05 * min(path_detour, 4)) * target_score * min(path_improvement, 3)
                elif path_improvement < 0:
                    bonus -= (0.16 + 0.03 * min(path_detour, 4)) * min(-path_improvement, 3)
                if current_path < 99 and next_path == 99:
                    bonus -= 0.75 * target_score
                best_target_bonus = max(best_target_bonus, bonus)
            score += max(best_target_bonus, 0.0)
        elif active_target_cells:
            best_contact_bonus = 0.0
            blocked_cells = coarse_blocked_cells_for_components(
                frame,
                actor_feature=actor_component.feature,
                target_cells=frozenset(active_target_cells),
            )
            for target_cell in active_target_cells:
                cell_distance = abs(next_cell[0] - target_cell[0]) + abs(next_cell[1] - target_cell[1])
                contact_bonus = 0.4 / (1.0 + cell_distance)
                if next_cell == target_cell:
                    contact_bonus += 0.25
                current_path = coarse_path_distance(actor_cell, target_cell, blocked=blocked_cells - {actor_cell, target_cell})
                next_path = coarse_path_distance(next_cell, target_cell, blocked=blocked_cells - {next_cell, target_cell})
                path_improvement = current_path - next_path
                if path_improvement > 0:
                    contact_bonus += 0.12 * min(path_improvement, 3)
                best_contact_bonus = max(best_contact_bonus, contact_bonus)
            score += best_contact_bonus

        score += 0.2 * level_memory.subgoal_bonus(coarse_cell_for_coord(next_anchor))
        context_stats = memory.keyboard_context_stats.get((candidate.action.name, actor_cell))
        if context_stats is not None:
            score += 0.18 * max(context_stats.change_rate, 0.0)
            score += 0.12 * max(context_stats.level_gain_rate, 0.0)

        score -= 0.25 * min(level_memory.keyboard_no_change_streaks.get(candidate.action.name, 0), 3)
        score -= self._control_repeat_penalty(
            action_key=candidate.key,
            current_anchor=actor_component.anchor,
            next_anchor=next_anchor,
            level_memory=level_memory,
            interaction_targets=interaction_targets,
            pressure=pressure,
        )
        return score

    def _control_map_decision(
        self,
        frame: np.ndarray,
        candidates: tuple[ActionCandidate, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        *,
        actor_component: FrameComponent | None,
    ) -> tuple[ActionCandidate | None, float, float]:
        if actor_component is None:
            actor_component = resolved_actor_component(frame, memory, level_memory)
        if actor_component is None or level_memory.step_index < 4:
            return None, 0.0, 0.0

        interaction_targets = self._interaction_targets(frame, memory, level_memory, actor_component)
        if not interaction_targets and not level_memory.active_target_cells():
            return None, 0.0, 0.0

        best_candidate: ActionCandidate | None = None
        best_score = float("-inf")
        second_score = float("-inf")
        best_confidence = 0.0
        for candidate in candidates:
            score = self._control_map_candidate_score(
                candidate,
                frame,
                memory,
                level_memory,
                actor_component=actor_component,
                interaction_targets=interaction_targets,
            )
            if score == float("-inf"):
                continue
            _delta, confidence = memory.best_delta(candidate.action.name)
            if score > best_score:
                second_score = best_score
                best_score = score
                best_candidate = candidate
                best_confidence = confidence
            elif score > second_score:
                second_score = score
        if best_candidate is None:
            return None, 0.0, 0.0

        margin = best_score - (second_score if second_score != float("-inf") else 0.0)
        confident_enough = (
            best_confidence >= self.tuning.control_map_candidate_confidence_min
            or level_memory.keyboard_control_confidence >= self.tuning.control_map_context_confidence_min
        )
        pressure = level_memory.loop_pressure(frame_signature(frame))
        if not confident_enough and pressure < 2:
            return None, best_score, margin
        if best_score < self.tuning.control_map_score_floor and margin < self.tuning.control_map_margin_floor:
            return None, best_score, margin
        return best_candidate, best_score, margin

    def _control_rollout_synthesis_decision(
        self,
        posterior_plan: ActionCandidate | None,
        posterior_kind: str | None,
        control_map_plan: ActionCandidate | None,
        control_map_score: float,
        control_map_margin: float,
        rollout_plan: ActionCandidate | None,
        rollout_score: float,
        rollout_margin: float,
        *,
        pressure: int,
    ) -> tuple[ActionCandidate | None, float, float]:
        if posterior_kind != "KEYBOARD" or posterior_plan is None:
            return None, 0.0, 0.0
        if posterior_plan.action in {GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7}:
            return None, 0.0, 0.0
        if control_map_plan is None or rollout_plan is None:
            return None, 0.0, 0.0
        if control_map_plan.action != rollout_plan.action:
            return None, 0.0, 0.0
        if control_map_plan.action == posterior_plan.action:
            return None, 0.0, 0.0
        if control_map_score < 0.95 or rollout_score < 1.0:
            return None, 0.0, 0.0
        if control_map_margin < 0.1 or rollout_margin < 0.08:
            return None, 0.0, 0.0
        score = 0.55 * control_map_score + 0.45 * rollout_score + 0.05 * min(pressure, 4)
        margin = min(control_map_margin, rollout_margin)
        return control_map_plan, score, margin

    def _target_conditioned_macro_decision(
        self,
        frame: np.ndarray,
        candidates: tuple[ActionCandidate, ...],
        available_actions: tuple[GameAction, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        *,
        actor_component: FrameComponent | None,
    ) -> tuple[ActionCandidate | None, float, float, str | None]:
        if actor_component is None or level_memory.step_index < 4:
            return None, 0.0, 0.0, None
        if not keyboard_actions(available_actions):
            return None, 0.0, 0.0, None

        targets = self._interaction_targets(frame, memory, level_memory, actor_component)
        if not targets and not level_memory.active_target_cells():
            return None, 0.0, 0.0, None

        top_target: Coord
        top_target_score: float
        if targets:
            top_target, top_target_score = targets[0]
        else:
            target_cell = level_memory.active_target_cells()[0]
            top_target = coarse_cell_center(target_cell)
            top_target_score = 0.75 + level_memory.target_cell_value_bonus(target_cell)
        target_cell = coarse_cell_for_coord(top_target)
        target_value = level_memory.target_cell_value_bonus(target_cell)
        blocked_penalty = level_memory.blocked_target_penalty(target_cell)
        active_contacts = bool(level_memory.active_target_contacts() or level_memory.active_target_cells())
        if target_value < 0.25 and not active_contacts:
            return None, 0.0, 0.0, None
        raw_target_distance = abs(actor_component.anchor[0] - top_target[0]) + abs(actor_component.anchor[1] - top_target[1])
        planning_target = top_target
        planning_target_score = top_target_score
        planning_target_value = target_value
        planning_blocked_penalty = blocked_penalty
        bridge_bonus = 0.0
        pressure = level_memory.loop_pressure(frame_signature(frame))

        control_map_plan, control_map_score, control_map_margin = self._control_map_decision(
            frame,
            candidates,
            memory,
            level_memory,
            actor_component=actor_component,
        )
        interact_action = next((candidate for candidate in candidates if candidate.action == GameAction.ACTION5), None)
        interact_candidate, interact_score, _interact_margin = self._planned_interact_decision(
            frame,
            candidates,
            memory,
            level_memory,
            actor_component=actor_component,
        )
        target_components = [
            component
            for component in extract_components(frame)
            if component.feature != actor_component.feature
        ]
        approach_cells = self._target_approach_cells(
            frame,
            memory,
            level_memory,
            actor_component,
            top_target=top_target,
        )
        if approach_cells:
            self.diagnostics.target_adjacency_cells_considered += len(approach_cells)
            best_approach, best_approach_score = approach_cells[0]
            if best_approach_score >= self.tuning.target_adjacency_bridge_floor and raw_target_distance > 1:
                planning_target = best_approach
                planning_target_score = top_target_score + 0.35 * best_approach_score
                planning_target_value = max(target_value, level_memory.target_cell_value_bonus(coarse_cell_for_coord(best_approach)))
                planning_blocked_penalty = max(blocked_penalty, level_memory.blocked_target_penalty(coarse_cell_for_coord(best_approach)))
                bridge_bonus = best_approach_score
        click_or_undo_candidate, click_or_undo_score, click_or_undo_margin = self._planned_click_decision(
            frame,
            candidates,
            memory,
            level_memory,
        )
        macro_control_plan = control_map_plan
        macro_control_score = control_map_score
        macro_control_margin = control_map_margin
        if bridge_bonus >= self.tuning.target_adjacency_bridge_floor:
            best_bridge_candidate: ActionCandidate | None = None
            best_bridge_score = float("-inf")
            second_bridge_score = float("-inf")
            current_bridge_distance = (
                abs(actor_component.anchor[0] - planning_target[0]) + abs(actor_component.anchor[1] - planning_target[1])
            )
            for candidate in candidates:
                if candidate.action in {GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7}:
                    continue
                delta, confidence = memory.best_delta(candidate.action.name)
                if delta is None:
                    continue
                next_anchor = (actor_component.anchor[0] + delta[0], actor_component.anchor[1] + delta[1])
                if not (0 <= next_anchor[0] < frame.shape[0] and 0 <= next_anchor[1] < frame.shape[1]):
                    continue
                next_distance = abs(next_anchor[0] - planning_target[0]) + abs(next_anchor[1] - planning_target[1])
                improvement = current_bridge_distance - next_distance
                stats = memory.stat_for(candidate.key)
                bridge_score = (
                    0.52 * max(confidence, 0.35)
                    + 0.28 * improvement
                    + 0.12 * planning_target_score
                    + 0.08 * planning_target_value
                    + 0.16 * bridge_bonus
                    + 0.12 * max(stats.change_rate, 0.0)
                    + 0.08 * max(stats.level_gain_rate, 0.0)
                    - 0.08 * planning_blocked_penalty
                    - 0.08 * min(level_memory.keyboard_no_change_streaks.get(candidate.action.name, 0), 3)
                )
                if improvement <= 0:
                    bridge_score -= 0.22
                if bridge_score > best_bridge_score:
                    second_bridge_score = best_bridge_score
                    best_bridge_score = bridge_score
                    best_bridge_candidate = candidate
                elif bridge_score > second_bridge_score:
                    second_bridge_score = bridge_score
            if best_bridge_candidate is not None:
                macro_control_plan = best_bridge_candidate
                macro_control_score = best_bridge_score
                macro_control_margin = best_bridge_score - max(second_bridge_score, 0.0)
        option_plan = self._target_option_bundle_plan(
            frame,
            candidates,
            available_actions,
            memory,
            level_memory,
            actor_component=actor_component,
            top_target=top_target,
            top_target_score=top_target_score,
            planning_target=planning_target,
            planning_target_score=planning_target_score,
            planning_target_value=planning_target_value,
            planning_blocked_penalty=planning_blocked_penalty,
            bridge_bonus=bridge_bonus,
        )

        best_candidate: ActionCandidate | None = None
        best_score = float("-inf")
        second_score = float("-inf")
        best_kind: str | None = None

        def consider(candidate: ActionCandidate | None, score: float, kind: str) -> None:
            nonlocal best_candidate, best_score, second_score, best_kind
            if candidate is None:
                return
            if score > best_score:
                second_score = best_score
                best_score = score
                best_candidate = candidate
                best_kind = kind
            elif score > second_score:
                second_score = score

        if macro_control_plan is not None and macro_control_plan.action not in {GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7}:
            delta, confidence = memory.best_delta(macro_control_plan.action.name)
            if delta is not None:
                next_anchor = (actor_component.anchor[0] + delta[0], actor_component.anchor[1] + delta[1])
                if 0 <= next_anchor[0] < frame.shape[0] and 0 <= next_anchor[1] < frame.shape[1]:
                    current_distance = abs(actor_component.anchor[0] - planning_target[0]) + abs(actor_component.anchor[1] - planning_target[1])
                    next_distance = abs(next_anchor[0] - planning_target[0]) + abs(next_anchor[1] - planning_target[1])
                    improvement = max(0, current_distance - next_distance)
                    base_move_score = (
                        0.68 * macro_control_score
                        + 0.22 * planning_target_score
                        + 0.18 * planning_target_value
                        - 0.12 * planning_blocked_penalty
                        + 0.06 * min(pressure, 4)
                    )
                    base_move_score += 0.14 * min(improvement, 3)
                    base_move_score += 0.1 * max(confidence, 0.35)
                    base_move_score += 0.14 * bridge_bonus
                    interact_steps: int | None = None
                    projected_interact_support = 0.0
                    if interact_action is not None:
                        hint = level_memory.mechanic_hint
                        hint_interact_bias = 0.0
                        if (
                            hint is not None
                            and hint.mode in {"INTERACT", "MIXED"}
                            and hint.goal in {"CONTACT", "COLLECT", "TOGGLE"}
                        ):
                            hint_interact_bias = 0.22 * hint.confidence
                        projected_bounds = actor_component.bounds
                        projected_anchor = actor_component.anchor
                        projected_distance = current_distance
                        for step in range(3):
                            projected_bounds = (
                                projected_bounds[0] + delta[0],
                                projected_bounds[1] + delta[1],
                                projected_bounds[2] + delta[0],
                                projected_bounds[3] + delta[1],
                            )
                            projected_anchor = (
                                projected_anchor[0] + delta[0],
                                projected_anchor[1] + delta[1],
                            )
                            if not (
                                0 <= projected_anchor[0] < frame.shape[0]
                                and 0 <= projected_anchor[1] < frame.shape[1]
                            ):
                                break
                            next_projected_distance = (
                                abs(projected_anchor[0] - planning_target[0]) + abs(projected_anchor[1] - planning_target[1])
                            )
                            if next_projected_distance > projected_distance:
                                break
                            projected_distance = next_projected_distance
                            step_support = 0.0
                            for component in target_components:
                                gap = bounds_gap(projected_bounds, component.bounds)
                                if gap > 1:
                                    continue
                                support = memory.target_affordance_score(component.feature, interact_action.action.name)
                                support += 0.28 * level_memory.target_value_bonus(
                                    component.feature,
                                    coarse_cell_for_coord(component.anchor),
                                )
                                if coarse_cell_for_coord(component.anchor) == target_cell:
                                    support += 0.1 * max(planning_target_score, 0.0)
                                step_support = max(step_support, support + hint_interact_bias)
                            if step_support > 0.0:
                                interact_steps = step
                                projected_interact_support = step_support
                                break
                    if interact_steps is not None:
                        interact_eta_bonus = 0.22 if interact_steps == 0 else 0.15 if interact_steps == 1 else 0.1
                        move_interact_score = (
                            base_move_score
                            + 0.18 * max(interact_score, 0.0)
                            + 0.32 * projected_interact_support
                            + 0.12 * max(planning_target_score, 0.0)
                            + 0.08 * max(planning_target_value, 0.0)
                            + interact_eta_bonus
                        )
                        consider(macro_control_plan, move_interact_score, "MOVE_INTERACT")
                    repeat_move_score = base_move_score + (0.08 if current_distance > 1 else -0.04)
                    if interact_steps is not None:
                        repeat_move_score -= 0.08
                    consider(macro_control_plan, repeat_move_score, "REPEAT_MOVE")

        if option_plan is not None:
            consider(option_plan.first_candidate, option_plan.score, "OPTION_PATH")

        if interact_candidate is not None and raw_target_distance <= 1:
            direct_interact_score = (
                interact_score
                + 0.24 * top_target_score
                + 0.16 * target_value
                - 0.08 * blocked_penalty
                + 0.04 * min(pressure, 4)
            )
            consider(interact_candidate, direct_interact_score, "MOVE_INTERACT")

        if click_or_undo_candidate is not None:
            if click_or_undo_candidate.action == GameAction.ACTION6:
                allow_click_macro = (
                    macro_control_plan is None
                    or macro_control_score < 0.95
                    or raw_target_distance >= 4
                    or planning_blocked_penalty >= 0.4
                    or planning_target_value < 0.9
                    or level_memory.family_failure_count("MOVE") >= level_memory.family_failure_count("CLICK") + 2
                )
                if allow_click_macro:
                    click_score = (
                        click_or_undo_score
                        + 0.14 * max(level_memory.family_failure_count("MOVE") - level_memory.family_failure_count("CLICK"), 0)
                        + 0.1 * planning_blocked_penalty
                        + (0.12 if raw_target_distance >= 4 else -0.06)
                    )
                    if macro_control_plan is not None and macro_control_score >= 1.0 and planning_blocked_penalty < 0.5:
                        click_score -= 0.18
                    consider(click_or_undo_candidate, click_score, "CLICK_PROBE")
            elif click_or_undo_candidate.action == GameAction.ACTION7:
                undo_score = (
                    click_or_undo_score
                    + 0.12 * min(pressure, 4)
                    + 0.16 * max(level_memory.family_failure_count("CLICK"), level_memory.family_failure_count("MOVE"))
                    + 0.18 * planning_blocked_penalty
                )
                consider(click_or_undo_candidate, undo_score, "UNDO_RECOVERY")

        if best_candidate is None or best_score < self.tuning.target_macro_accept_score:
            return None, max(best_score, 0.0), 0.0, None
        margin = best_score - max(second_score, float("-inf"))
        if second_score == float("-inf"):
            margin = best_score
        if margin < self.tuning.target_macro_margin_floor:
            return None, best_score, margin, None
        if bridge_bonus >= self.tuning.target_adjacency_bridge_floor:
            self.diagnostics.target_adjacency_bridge_activations += 1
        if best_kind == "OPTION_PATH":
            level_memory.pending_option_plan = option_plan
        return best_candidate, best_score, margin, best_kind

    def _apply_target_macro_choice(
        self,
        frame: np.ndarray,
        available_actions: tuple[GameAction, ...],
        candidates: tuple[ActionCandidate, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        chosen: ActionCandidate,
        kind: str,
        *,
        target_macro_score: float,
        actor_component: FrameComponent | None,
    ) -> ActionCandidate:
        self.diagnostics.target_macro_plan_choices += 1
        if kind == "REPEAT_MOVE":
            self.diagnostics.target_macro_repeat_move_choices += 1
        elif kind == "MOVE_INTERACT":
            self.diagnostics.target_macro_move_interact_choices += 1
        elif kind == "OPTION_PATH":
            self.diagnostics.target_option_plan_choices += 1
        elif kind == "CLICK_PROBE":
            self.diagnostics.target_macro_click_choices += 1
        elif kind == "UNDO_RECOVERY":
            self.diagnostics.target_macro_undo_choices += 1
        if kind == "OPTION_PATH":
            plan = level_memory.pending_option_plan
            level_memory.pending_option_plan = None
            if (
                plan is not None
                and plan.first_candidate.key == chosen.key
                and plan.follow_ups
                and not level_memory.macro_queue
            ):
                level_memory.macro_queue.extend(plan.follow_ups)
                level_memory.macro_source_step = level_memory.step_index
                self.diagnostics.target_option_bundle_injections += 1
                self.diagnostics.target_option_bundle_actions += len(plan.follow_ups)
        elif kind in {"REPEAT_MOVE", "MOVE_INTERACT"} and chosen.action not in {GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7}:
            self._seed_control_commit_bundle(
                frame,
                available_actions,
                candidates,
                memory,
                level_memory,
                chosen,
                control_score=target_macro_score,
                control_margin=max(self.tuning.target_macro_margin_floor, 0.12),
                actor_component=actor_component,
            )
        return chosen

    def _control_repeat_penalty(
        self,
        *,
        action_key: ActionKey,
        current_anchor: Coord,
        next_anchor: Coord,
        level_memory: LevelMemory,
        interaction_targets: tuple[tuple[Coord, float], ...],
        pressure: int,
    ) -> float:
        if pressure < 2:
            return 0.0

        repeated_key = level_memory.keyboard_repeat_key == action_key and level_memory.keyboard_repeat_steps >= 2
        revisiting_anchor = next_anchor in level_memory.actor_positions
        if not repeated_key and not revisiting_anchor:
            return 0.0

        penalty = 0.0
        if repeated_key:
            penalty += 0.08 * min(level_memory.keyboard_repeat_steps, 4)
            penalty += 0.06 * min(level_memory.keyboard_plateau_steps, 4)
        if revisiting_anchor:
            penalty += 0.12 + 0.04 * min(pressure, 4)

        if interaction_targets:
            current_distance = min(
                abs(current_anchor[0] - target[0]) + abs(current_anchor[1] - target[1])
                for target, _target_score in interaction_targets[:4]
            )
            next_distance = min(
                abs(next_anchor[0] - target[0]) + abs(next_anchor[1] - target[1])
                for target, _target_score in interaction_targets[:4]
            )
            if repeated_key and next_distance >= current_distance:
                penalty += 0.16
            elif revisiting_anchor and next_distance > current_distance:
                penalty += 0.08

        return penalty

    def _late_click_detour_active(
        self,
        *,
        available_actions: tuple[GameAction, ...],
        actor_component: FrameComponent | None,
        level_memory: LevelMemory,
        click_candidate: ActionCandidate | None,
        click_score: float,
        keyboard_candidate: ActionCandidate | None,
        keyboard_score: float,
    ) -> bool:
        if actor_component is None or click_candidate is None or keyboard_candidate is None:
            return False
        if not keyboard_actions(available_actions):
            return False
        if not (4 <= level_memory.step_index <= 12):
            return False
        if level_memory.active_target_contacts():
            return False
        control_confidence = level_memory.keyboard_control_confidence
        if control_confidence < 0.45 or control_confidence >= 0.9:
            return False
        if level_memory.family_failure_count("MOVE") > level_memory.family_failure_count("CLICK") + 1:
            return False
        return keyboard_score >= click_score - 0.12

    def _candidate_from_advisor_mode(
        self,
        mode: str | None,
        candidates: tuple[ActionCandidate, ...],
        signature: bytes,
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        rollout: ActionCandidate | None,
    ) -> ActionCandidate | None:
        if mode is None:
            return None
        if mode == "UNDO":
            return next((candidate for candidate in candidates if candidate.action == GameAction.ACTION7), None)
        if mode == "CLICK":
            click_candidates = tuple(candidate for candidate in candidates if candidate.action == GameAction.ACTION6)
            if not click_candidates:
                return None
            for candidate in click_candidates:
                if not level_memory.has_tried(signature, candidate.key):
                    return candidate
            return click_candidates[0]
        if mode == "MOVE":
            if rollout is not None:
                return rollout
            keyboard_candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.action not in {GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7}
            )
            for candidate in keyboard_candidates:
                if not level_memory.has_tried(signature, candidate.key):
                    return candidate
            return keyboard_candidates[0] if keyboard_candidates else None
        if mode == "INTERACT":
            interact = next((candidate for candidate in candidates if candidate.action == GameAction.ACTION5), None)
            if interact is not None:
                return interact
            click_candidates = tuple(candidate for candidate in candidates if candidate.action == GameAction.ACTION6)
            if click_candidates:
                return click_candidates[0]
        return None

    def _advisor_mode(
        self,
        frame: np.ndarray,
        candidates: tuple[ActionCandidate, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        *,
        rollout: ActionCandidate | None = None,
        rollout_score: float = 0.0,
        rollout_margin: float = 0.0,
    ) -> str | None:
        if self.action_advisor is None or level_memory.qwen_calls_used >= self.action_advisor_budget_per_level:
            return None
        stall_steps = level_memory.step_index - level_memory.last_changed_step
        if level_memory.step_index < 4 or stall_steps < 2:
            return None
        keyboard_candidates = tuple(
            candidate for candidate in candidates if candidate.action not in {GameAction.ACTION6, GameAction.ACTION7}
        )
        allowed_modes: list[str] = []
        if any(candidate.action == GameAction.ACTION6 for candidate in candidates):
            allowed_modes.append("CLICK")
        if any(candidate.action == GameAction.ACTION5 for candidate in keyboard_candidates):
            allowed_modes.append("INTERACT")
        if any(candidate.action not in {GameAction.ACTION5} for candidate in keyboard_candidates):
            allowed_modes.append("MOVE")
        if any(candidate.action == GameAction.ACTION7 for candidate in candidates) and stall_steps >= 4:
            allowed_modes.append("UNDO")
        if len(allowed_modes) < 2:
            return None
        actor_component = component_by_actor_prior(frame, memory)
        prompt_lines = [
            "Choose the best exploration mode for the next ARC action burst.",
            f"Reply with exactly one mode from: {', '.join(allowed_modes)}",
            f"Current step: {level_memory.step_index}",
            f"Stall steps since meaningful change: {stall_steps}",
        ]
        if actor_component is None:
            prompt_lines.append("Known controllable object: unknown")
        else:
            prompt_lines.append(f"Known controllable object anchor: {actor_component.anchor}")
        prompt_lines.append("Action summaries:")
        for candidate in keyboard_candidates:
            stats = memory.stat_for(candidate.key)
            delta, confidence = memory.best_delta(candidate.action.name)
            delta_text = f"delta={delta} conf={confidence:.2f}" if delta is not None else "delta=unknown"
            prompt_lines.append(
                f"- {candidate.action.name}: reward={stats.mean_reward:.2f} change={stats.change_rate:.2f} "
                f"level={stats.level_gain_rate:.2f} {delta_text} "
                f"streak={level_memory.keyboard_no_change_streaks.get(candidate.action.name, 0)}"
            )
        click_count = sum(1 for candidate in candidates if candidate.action == GameAction.ACTION6)
        if click_count:
            prompt_lines.append(f"- ACTION6 candidates currently available: {click_count}")
        if rollout is not None:
            prompt_lines.append(
                f"Heuristic rollout suggestion: {rollout.action.name} score={rollout_score:.2f} margin={rollout_margin:.2f}"
            )
        if actor_component is not None:
            targets = self._interaction_targets(frame, memory, level_memory, actor_component)
            if targets:
                prompt_lines.append("Interaction targets:")
                prompt_lines.extend(
                    f"- target@{coord[0]},{coord[1]} priority={score:.2f}"
                    for coord, score in targets[:4]
                )
        if level_memory.recent_events:
            prompt_lines.append("Recent action outcomes:")
            prompt_lines.extend(f"- {event}" for event in level_memory.recent_events)
        chosen_mode = self.action_advisor.choose_option("\n".join(prompt_lines), tuple(allowed_modes))
        level_memory.qwen_calls_used += 1
        self.diagnostics.qwen_calls += 1
        self.diagnostics.qwen_mode_calls += 1
        return chosen_mode

    def _advisor_action(
        self,
        frame: np.ndarray,
        candidates: tuple[ActionCandidate, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        *,
        rollout: ActionCandidate | None = None,
        rollout_score: float = 0.0,
        rollout_margin: float = 0.0,
    ) -> ActionCandidate | None:
        if self.action_advisor is None or level_memory.qwen_calls_used >= self.action_advisor_budget_per_level:
            return None
        stall_steps = level_memory.step_index - level_memory.last_changed_step
        if stall_steps < 3 or level_memory.step_index < 5 or not memory.interaction_target_stats:
            return None
        keyboard_candidates = tuple(
            candidate for candidate in candidates if candidate.action not in {GameAction.ACTION6, GameAction.ACTION7}
        )
        if len(keyboard_candidates) < 2:
            return None
        actor_component = component_by_actor_prior(frame, memory)
        if actor_component is None:
            return None
        targets = self._interaction_targets(frame, memory, level_memory, actor_component)
        if not targets:
            return None
        allowed_actions = tuple(candidate.action.name for candidate in keyboard_candidates)
        streak_text = ", ".join(
            f"{candidate.action.name}={level_memory.keyboard_no_change_streaks.get(candidate.action.name, 0)}"
            for candidate in keyboard_candidates
        )
        target_lines = [
            f"- target@{target[0]},{target[1]} priority={target_score:.2f}"
            for target, target_score in targets[:4]
        ]
        action_lines = []
        for candidate in keyboard_candidates:
            delta, confidence = memory.best_delta(candidate.action.name)
            if delta is None:
                continue
            next_anchor = (
                actor_component.anchor[0] + delta[0],
                actor_component.anchor[1] + delta[1],
            )
            if not (0 <= next_anchor[0] < frame.shape[0] and 0 <= next_anchor[1] < frame.shape[1]):
                continue
            best_target, best_value = max(
                (
                    target,
                    target_score / (1.0 + abs(next_anchor[0] - target[0]) + abs(next_anchor[1] - target[1])),
                )
                for target, target_score in targets
            )
            action_lines.append(
                f"- {candidate.action.name}: delta={delta} next_anchor={next_anchor} "
                f"best_target={best_target} reach_value={best_value:.2f} confidence={confidence:.2f}"
            )
        if not action_lines:
            return None
        prompt_lines = [
            "You are helping explore a deterministic arcade environment.",
            f"Choose exactly one next action from: {', '.join(allowed_actions)}.",
            "Prefer actions most likely to move the controllable object toward the most promising interaction target.",
            f"Current step: {level_memory.step_index}",
            f"Stall steps since meaningful change: {stall_steps}",
            f"Keyboard no-change streaks: {streak_text or 'none'}",
            f"Known actor anchor: {actor_component.anchor}",
            f"Known actor coarse cell: {coarse_cell_for_coord(actor_component.anchor)}",
        ]
        prompt_lines.append("Highest-priority interaction targets:")
        prompt_lines.extend(target_lines)
        prompt_lines.append("Predicted action consequences:")
        prompt_lines.extend(action_lines[:6])
        if rollout is not None:
            prompt_lines.append(
                "Current heuristic rollout suggestion: "
                f"{rollout.action.name} (score={rollout_score:.2f}, margin={rollout_margin:.2f})"
            )
            prompt_lines.append(
                "Only override the heuristic if another action looks clearly better for reaching the best target."
            )
        if level_memory.recent_events:
            prompt_lines.append("Recent action outcomes:")
            prompt_lines.extend(f"- {event}" for event in level_memory.recent_events)
        prompt_lines.append(f"Reply with only one action name from: {', '.join(allowed_actions)}")
        chosen_name = self.action_advisor.choose_action("\n".join(prompt_lines), allowed_actions)
        level_memory.qwen_calls_used += 1
        self.diagnostics.qwen_calls += 1
        if chosen_name is None:
            return None
        return next((candidate for candidate in keyboard_candidates if candidate.action.name == chosen_name), None)

    def _candidates(
        self,
        frame: np.ndarray,
        available_actions: tuple[GameAction, ...],
        memory: EnvironmentMemory,
    ) -> tuple[ActionCandidate, ...]:
        candidates: list[ActionCandidate] = []
        click_points = candidate_clicks(frame, memory)
        for action in available_actions:
            if action == GameAction.ACTION6:
                for coord, feature in click_points:
                    candidates.append(
                        ActionCandidate(
                            action=action,
                            data={"x": coord[1], "y": coord[0]},
                            key=(action.name, feature),
                            label=f"{action.name}@{coord[0]},{coord[1]}",
                        )
                    )
                continue
            candidates.append(
                ActionCandidate(
                    action=action,
                    data=None,
                    key=(action.name, None),
                    label=action.name,
                )
            )
        return tuple(candidates)

    def _bootstrap_probe_hint(
        self,
        frame: np.ndarray,
        available_actions: tuple[GameAction, ...],
        memory: EnvironmentMemory,
    ) -> MechanicHint | None:
        keyboard = keyboard_actions(available_actions)
        if GameAction.ACTION6 not in available_actions or not keyboard:
            return None
        if component_by_actor_prior(frame, memory) is not None:
            return None
        components = extract_components(frame)
        if not components:
            return None
        top_component = max(
            components,
            key=lambda component: (
                max(0.0, (8 - min(component.size, 8)) / 8.0),
                component.size / max(int(np.count_nonzero(frame == component.color)), component.size),
                interaction_target_priority(component, memory),
            ),
        )
        color_mass = int(np.count_nonzero(frame == top_component.color))
        small_score = max(0.0, (8 - min(top_component.size, 8)) / 8.0)
        rarity = top_component.size / max(color_mass, top_component.size)
        click_stats = memory.click_stats.get(top_component.feature)
        click_prior = 0.0
        if click_stats is not None:
            click_prior = (
                0.2 * max(click_stats.mean_reward, 0.0)
                + 0.45 * click_stats.change_rate
                + 0.55 * click_stats.level_gain_rate
            )
        move_known = any(
            memory.best_delta(action.name)[1] >= 0.55
            for action in keyboard
            if action != GameAction.ACTION5
        )
        clicky_score = 0.4 * small_score + 0.3 * rarity + 0.25 * click_prior
        if clicky_score < 0.52:
            return None
        if move_known and clicky_score < 0.72:
            return None
        focus = "SMALL_OBJECT" if small_score >= rarity else "RARE_COLOR"
        goal: MechanicGoal = "CLEAR" if top_component.size <= 4 else "TOGGLE" if top_component.size <= 8 else "COLLECT"
        confidence = min(0.64, 0.22 + 0.45 * clicky_score - (0.08 if move_known else 0.0))
        return MechanicHint(
            mode="CLICK",
            goal=goal,
            focus=focus,
            confidence=confidence,
            source_step=-1,
            source="SYMBOLIC",
            raw_text=f"bootstrap mode=CLICK goal={goal} focus={focus} score={clicky_score:.2f}",
        )

    def _prime_probe_queue(
        self,
        obs,
        action_space: list[GameAction],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
    ) -> None:
        frame = primary_frame(obs.frame)
        available = tuple(action_space)
        queue: list[ActionCandidate] = []
        click_probe_points: tuple[tuple[Coord, ClickFeature], ...] = ()
        keyboard = keyboard_actions(available)
        if not keyboard and available == (GameAction.ACTION6,):
            merged: list[tuple[Coord, ClickFeature]] = []
            seen_coords: set[Coord] = set()
            sweep_points = sweep_clicks()
            self.diagnostics.sweep_probe_levels += 1
            self.diagnostics.sweep_probe_points += len(sweep_points)
            for coord, feature in (*sweep_points, *candidate_clicks(frame, memory, limit=10)):
                if coord in seen_coords:
                    continue
                seen_coords.add(coord)
                merged.append((coord, feature))
            click_probe_points = tuple(merged)
        elif not keyboard:
            click_probe_points = candidate_clicks(frame, memory, limit=8)
        else:
            click_probe_points = candidate_clicks(frame, memory, limit=2 if GameAction.ACTION6 in available else 0)
        for action in keyboard:
            queue.append(
                ActionCandidate(
                    action=action,
                    data=None,
                    key=(action.name, None),
                    label=action.name,
                )
            )
        if GameAction.ACTION6 in available:
            for coord, feature in click_probe_points:
                queue.append(
                    ActionCandidate(
                        action=GameAction.ACTION6,
                        data={"x": coord[1], "y": coord[0]},
                        key=(GameAction.ACTION6.name, feature),
                        label=f"{GameAction.ACTION6.name}@{coord[0]},{coord[1]}",
                    )
                )
        bootstrap_hint = self._bootstrap_probe_hint(frame, available, memory)
        if bootstrap_hint is not None:
            click_queue = [candidate for candidate in queue if candidate.action == GameAction.ACTION6]
            other_queue = [candidate for candidate in queue if candidate.action != GameAction.ACTION6]
            if click_queue and other_queue:
                queue = [*click_queue, *other_queue]
                self.diagnostics.bootstrap_probe_reorders += 1
        level_memory.probe_queue = deque(queue)

    def _enqueue_keyboard_followups(
        self,
        *,
        executed: ActionCandidate,
        next_signature: bytes,
        next_frame: np.ndarray,
        available_actions: tuple[GameAction, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        changed_pixels: int,
    ) -> None:
        actor_component = component_by_actor_prior(next_frame, memory)
        if actor_component is None:
            return
        _delta, confidence = memory.best_delta(executed.action.name)
        if confidence <= 0.0:
            return
        level_memory.keyboard_control_confidence = max(level_memory.keyboard_control_confidence, confidence)
        key = executed.key
        discovered_new_anchor = actor_component.anchor not in level_memory.actor_positions
        plateau = not discovered_new_anchor or changed_pixels <= 2
        if level_memory.keyboard_repeat_key == key:
            level_memory.keyboard_repeat_steps += 1
            if plateau:
                level_memory.keyboard_plateau_steps += 1
            else:
                level_memory.keyboard_plateau_steps = 0
        else:
            level_memory.keyboard_repeat_key = key
            level_memory.keyboard_repeat_steps = 1
            level_memory.keyboard_plateau_steps = 1 if plateau else 0

        if (
            confidence >= 0.55
            and discovered_new_anchor
            and level_memory.keyboard_repeat_steps < 4
            and not level_memory.has_tried(next_signature, key)
        ):
            level_memory.probe_queue.appendleft(executed)
            self.diagnostics.keyboard_followups_enqueued += 1
            return

        if level_memory.keyboard_plateau_steps < 1 and level_memory.keyboard_repeat_steps < 3:
            return

        alternate_candidates: list[ActionCandidate] = []
        for action in keyboard_actions(available_actions):
            if action == executed.action:
                continue
            alternate = ActionCandidate(
                action=action,
                data=None,
                key=(action.name, None),
                label=action.name,
            )
            if level_memory.has_tried(next_signature, alternate.key):
                continue
            alternate_candidates.append(alternate)
        if alternate_candidates:
            interaction_targets = self._interaction_targets(next_frame, memory, level_memory, actor_component)
            alternate_candidates.sort(
                key=lambda candidate: self._control_map_candidate_score(
                    candidate,
                    next_frame,
                    memory,
                    level_memory,
                    actor_component=actor_component,
                    interaction_targets=interaction_targets,
                ),
                reverse=True,
            )
        for alternate in reversed(alternate_candidates):
            level_memory.probe_queue.appendleft(alternate)
            self.diagnostics.keyboard_turn_probes_enqueued += 1

    def _planned_frontier_action(
        self,
        frame: np.ndarray,
        signature: bytes,
        candidates: tuple[ActionCandidate, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
    ) -> ActionCandidate | None:
        if not candidates:
            return None
        candidate_by_key = {candidate.key: candidate for candidate in candidates}
        if level_memory.frontier_plan and level_memory.frontier_plan[0] not in candidate_by_key:
            level_memory.frontier_plan.clear()
        if level_memory.frontier_plan:
            return candidate_by_key.get(level_memory.frontier_plan.popleft())

        local_untried = level_memory.untried_keys(signature)
        stalled = level_memory.step_index - level_memory.last_changed_step >= 6
        current_abstract_key = level_memory.abstract_state_keys.get(signature)
        if current_abstract_key is None:
            current_abstract_key = self._abstract_state_key(frame, tuple(candidate.action for candidate in candidates), memory, level_memory)
            level_memory.note_abstract_state(signature, current_abstract_key)
        abstract_revisits = level_memory.abstract_revisit_count(current_abstract_key)
        abstract_pressure = level_memory.loop_pressure(signature)

        current_actor = level_memory.state_actor_anchors.get(signature)
        abstract_loop_ready = abstract_revisits >= 1 and abstract_pressure >= 2
        actor_loop_ready = (
            current_actor is not None
            and abstract_loop_ready
            and level_memory.family_failure_count("MOVE") >= 2
        )
        abstract_stalled = stalled or (current_actor is None and abstract_loop_ready) or actor_loop_ready
        if local_untried and not abstract_stalled:
            return None
        target_components = []
        if current_actor is not None:
            current_feature = component_by_actor_prior(frame, memory)
            target_components = [
                component
                for component in extract_components(frame)
                if current_feature is None or component.feature != current_feature.feature
            ]

        def frontier_state_score(state_signature: bytes) -> float:
            actor_anchor = level_memory.state_actor_anchors.get(state_signature)
            if actor_anchor is None or not target_components:
                return 0.0
            best = 0.0
            actor_cell = coarse_cell_for_coord(actor_anchor)
            for component in target_components:
                target_priority = interaction_target_priority(component, memory)
                distance = manhattan_to_bounds(actor_anchor, component.bounds)
                score = target_priority * (0.8 / (1.0 + distance))
                if coarse_cell_for_coord(component.anchor) == actor_cell:
                    score += 0.5 * target_priority
                best = max(best, score)
            return best

        best_abstract_frontier: tuple[float, int, tuple[ActionKey, ...]] | None = None
        best_raw_frontier: tuple[float, int, tuple[ActionKey, ...]] | None = None

        frontier = deque([(signature, ())])
        visited = {signature}
        while frontier:
            current, path = frontier.popleft()
            if current != signature:
                frontier_actions = level_memory.untried_keys(current)
                if frontier_actions:
                    best_frontier = max(frontier_actions, key=key_stability_score)
                    full_path = path + (best_frontier,)
                    if local_untried and len(full_path) > 3:
                        continue
                    state_score = frontier_state_score(current)
                    plan_score = state_score - 0.08 * len(full_path)
                    if best_raw_frontier is None or plan_score > best_raw_frontier[0]:
                        best_raw_frontier = (plan_score, len(full_path), full_path)
                    abstract_key = level_memory.abstract_state_keys.get(current)
                    if abstract_key is not None:
                        abstract_score = self._abstract_frontier_state_score(abstract_key, current_abstract_key) - 0.06 * len(full_path)
                        if best_abstract_frontier is None or abstract_score > best_abstract_frontier[0]:
                            best_abstract_frontier = (abstract_score, len(full_path), full_path)

            outgoing = [
                (action_key, next_signature)
                for (state_signature, action_key), next_signature in level_memory.transitions.items()
                if state_signature == current and next_signature not in visited and next_signature != current
            ]
            outgoing.sort(key=lambda item: key_stability_score(item[0]), reverse=True)
            for action_key, next_signature in outgoing:
                visited.add(next_signature)
                frontier.append((next_signature, path + (action_key,)))
        chosen_plan = best_raw_frontier
        used_abstract_frontier = False
        if best_abstract_frontier is not None and (
            best_raw_frontier is None or best_abstract_frontier[0] >= best_raw_frontier[0] + 0.05
        ):
            chosen_plan = best_abstract_frontier
            used_abstract_frontier = True
        if chosen_plan is None:
            return None
        _plan_score, _path_len, full_path = chosen_plan
        level_memory.frontier_plan = deque(full_path)
        if used_abstract_frontier:
            self.diagnostics.abstract_frontier_plan_routes += 1
        return candidate_by_key.get(level_memory.frontier_plan.popleft())

    def _planned_click_decision(
        self,
        frame: np.ndarray,
        candidates: tuple[ActionCandidate, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
    ) -> tuple[ActionCandidate | None, float, float]:
        hint = level_memory.mechanic_hint
        if hint is None or hint.confidence <= 0.0:
            return None, 0.0, 0.0
        click_candidates = tuple(
            candidate for candidate in candidates if candidate.action == GameAction.ACTION6 and candidate.data is not None
        )
        undo_candidate = next((candidate for candidate in candidates if candidate.action == GameAction.ACTION7), None)
        if not click_candidates and undo_candidate is None:
            return None, 0.0, 0.0

        actor_component = component_by_actor_prior(frame, memory)
        interaction_targets: tuple[tuple[Coord, float], ...] = ()
        if actor_component is not None:
            interaction_targets = self._interaction_targets(frame, memory, level_memory, actor_component)
        hotspot_map = dict(hotspot_cells(memory))
        signature = frame_signature(frame)
        pressure = level_memory.loop_pressure(signature)

        component_scores: list[tuple[FrameComponent, float]] = []
        for component in extract_components(frame):
            if actor_component is not None and component.feature == actor_component.feature:
                continue
            score = interaction_target_priority(component, memory)
            score += self._mechanic_focus_bonus(component, frame, memory, level_memory)
            click_stats = memory.click_stats.get(component.feature)
            if click_stats is not None:
                score += 0.25 * max(click_stats.mean_reward, 0.0)
                score += 0.35 * click_stats.change_rate
                score += 0.55 * click_stats.level_gain_rate
            component_scores.append((component, score))
        component_scores.sort(key=lambda item: item[1], reverse=True)
        top_components = component_scores[:6]

        best_candidate: ActionCandidate | None = None
        best_score = float("-inf")
        second_score = float("-inf")
        for candidate in click_candidates:
            coord = (int(candidate.data["y"]), int(candidate.data["x"]))
            cell = coarse_cell_for_coord(coord)
            stats = memory.stat_for(candidate.key)
            score = 0.2 + self._candidate_mechanic_bonus(
                candidate,
                frame,
                memory,
                level_memory,
                actor_component=actor_component,
                interaction_targets=interaction_targets,
            )
            score += 0.35 * max(stats.mean_reward, 0.0)
            score += 0.65 * stats.change_rate
            score += 1.15 * stats.level_gain_rate
            score += 0.45 * level_memory.subgoal_bonus(cell)
            score += 0.25 * hotspot_map.get(cell, 0.0)
            if top_components:
                best_target_score = 0.0
                for component, target_score in top_components:
                    distance = manhattan_to_bounds(coord, component.bounds)
                    target_bonus = target_score / (1.0 + distance)
                    if hint.goal == "CLEAR" and distance == 0:
                        target_bonus += 0.2 * hint.confidence
                    if hint.goal == "TOGGLE" and distance <= 1:
                        target_bonus += 0.16 * hint.confidence
                    best_target_score = max(best_target_score, target_bonus)
                score += best_target_score
            if level_memory.has_tried(signature, candidate.key):
                score -= 0.25
            if stats.attempts >= 3:
                score -= 0.25 * max(0.0, 0.55 - memory.effect_diversity(candidate.key))
            if pressure >= 2:
                score += 0.05 * min(pressure, 4)
            if score > best_score:
                second_score = best_score
                best_score = score
                best_candidate = candidate
            elif score > second_score:
                second_score = score

        if undo_candidate is not None:
            undo_stats = memory.stat_for(undo_candidate.key)
            undo_score = 0.0
            click_fail = level_memory.family_failure_count("CLICK")
            undo_fail = level_memory.family_failure_count("UNDO")
            recent_click_dead_ends = self._recent_event_count(
                level_memory,
                "ACTION6 changed=1 level_gain=0",
            )
            if (
                hint.mode == "UNDO"
                or (click_fail >= 4 and pressure >= 2)
                or (recent_click_dead_ends >= 2 and click_fail >= 2)
            ):
                undo_score += 0.45 + 0.45 * hint.confidence
                undo_score += 0.12 * min(click_fail, 5)
                undo_score += 0.08 * min(pressure, 4)
                undo_score += 0.18 * min(recent_click_dead_ends, 3)
                undo_score += 0.2 * undo_stats.change_rate
                undo_score += 0.35 * undo_stats.level_gain_rate
                undo_score -= 0.1 * min(undo_fail, 3)
                if undo_score > best_score:
                    second_score = best_score
                    best_score = undo_score
                    best_candidate = undo_candidate
                elif undo_score > second_score:
                    second_score = undo_score

        if best_candidate is None or best_score < 0.78:
            return None, max(best_score, 0.0), 0.0
        margin = best_score - max(second_score, float("-inf"))
        if second_score == float("-inf"):
            margin = best_score
        return best_candidate, best_score, margin

    def _planned_interact_decision(
        self,
        frame: np.ndarray,
        candidates: tuple[ActionCandidate, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        *,
        actor_component: FrameComponent | None = None,
    ) -> tuple[ActionCandidate | None, float, float]:
        hint = level_memory.mechanic_hint
        interact_candidate = next((candidate for candidate in candidates if candidate.action == GameAction.ACTION5), None)
        if interact_candidate is None or hint is None or hint.confidence <= 0.0:
            return None, 0.0, 0.0
        if actor_component is None:
            actor_component = resolved_actor_component(frame, memory, level_memory)
        if actor_component is None:
            return None, 0.0, 0.0
        targets = self._interaction_targets(frame, memory, level_memory, actor_component)
        if not targets:
            return None, 0.0, 0.0
        target_components = [
            component
            for component in extract_components(frame)
            if component.feature != actor_component.feature
        ]
        distance = min(
            bounds_gap(actor_component.bounds, component.bounds)
            for component in target_components
        ) if target_components else 99
        if distance > 1 and hint.mode == "INTERACT":
            return None, 0.0, 0.0

        interact_stats = memory.stat_for(interact_candidate.key)
        score = 0.25 + self._candidate_mechanic_bonus(
            interact_candidate,
            frame,
            memory,
            level_memory,
            actor_component=actor_component,
            interaction_targets=targets,
        )
        score += 0.25 * max(interact_stats.mean_reward, 0.0)
        score += 0.55 * interact_stats.change_rate
        score += 0.9 * interact_stats.level_gain_rate
        score += max(
            (
                memory.target_affordance_score(component.feature, interact_candidate.action.name)
                for component in target_components
                if bounds_gap(actor_component.bounds, component.bounds) <= 1
            ),
            default=0.0,
        )
        if hint.goal in {"TOGGLE", "COLLECT", "CONTACT"} and distance <= 1:
            score += 0.55 * hint.confidence
        if distance == 0:
            score += 0.18 * hint.confidence
        elif distance == 1:
            score += 0.12 * hint.confidence
        if level_memory.family_failure_count("MOVE") >= 2:
            score += 0.08 * min(level_memory.family_failure_count("MOVE"), 4)
        if score < 0.85:
            return None, score, 0.0
        return interact_candidate, score, score

    def _interaction_targets(
        self,
        frame: np.ndarray,
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        actor_component: FrameComponent,
    ) -> tuple[tuple[Coord, float], ...]:
        actor_cell = coarse_cell_for_coord(actor_component.anchor)
        active_contacts = set(level_memory.active_target_contacts())
        contact_cells = set(level_memory.active_target_cells())
        merged: dict[CoarseCell, tuple[Coord, float]] = {}

        for component in extract_components(frame):
            if component.anchor == actor_component.anchor:
                continue
            cell = coarse_cell_for_coord(component.anchor)
            stats = memory.click_stats.get(component.feature)
            score = self.tuning.interaction_target_base + min(component.size, 16) / 64.0
            if stats is not None:
                score += (
                    self.tuning.interaction_target_reward_weight * max(stats.mean_reward, 0.0)
                    + self.tuning.interaction_target_change_weight * stats.change_rate
                    + self.tuning.interaction_target_level_gain_weight * stats.level_gain_rate
                )
            score += self.tuning.interaction_target_subgoal_weight * level_memory.subgoal_bonus(cell)
            score += self.tuning.interaction_target_value_weight * level_memory.target_value_bonus(component.feature, cell)
            score -= self.tuning.interaction_target_blocked_penalty * level_memory.blocked_target_penalty(cell)
            score += self._mechanic_focus_bonus(component, frame, memory, level_memory)
            if component.feature in active_contacts:
                score += self.tuning.interaction_target_active_contact_bonus
            elif cell in contact_cells:
                score += self.tuning.interaction_target_contact_cell_bonus
            if cell == actor_cell or score <= 0.0:
                continue
            merged[cell] = (component.anchor, max(score, merged.get(cell, (component.anchor, 0.0))[1]))

        for cell, hotspot_score in hotspot_cells(memory):
            if cell == actor_cell:
                continue
            coord = coarse_cell_center(cell)
            base = 0.35 * hotspot_score + 0.35 * level_memory.subgoal_bonus(cell)
            base += 0.55 * level_memory.target_cell_value_bonus(cell)
            base -= 0.45 * level_memory.blocked_target_penalty(cell)
            if cell in contact_cells:
                base += 0.55
            existing = merged.get(cell)
            if existing is None:
                merged[cell] = (coord, base)
            else:
                merged[cell] = (existing[0], existing[1] + base)

        for cell in contact_cells:
            if cell == actor_cell:
                continue
            coord = coarse_cell_center(cell)
            base = 0.55 + 0.45 * level_memory.subgoal_bonus(cell)
            base += 0.7 * level_memory.target_cell_value_bonus(cell)
            base -= 0.5 * level_memory.blocked_target_penalty(cell)
            existing = merged.get(cell)
            if existing is None:
                merged[cell] = (coord, base)
            else:
                merged[cell] = (existing[0], existing[1] + base)

        targets = sorted(merged.values(), key=lambda item: item[1], reverse=True)
        if not targets:
            targets = sorted(
                (
                    (component.anchor, 0.4 + min(component.size, 16) / 64.0)
                    for component in extract_components(frame)
                    if component.anchor != actor_component.anchor
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        return tuple(targets[:8])

    def _approach_cell_exhaustion(self, level_memory: LevelMemory, cell: CoarseCell) -> float:
        exhaustion = sum(1 for anchor in level_memory.actor_positions if coarse_cell_for_coord(anchor) == cell)
        row, col = cell
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (row + delta_row, col + delta_col)
            neighbor_count = sum(1 for anchor in level_memory.actor_positions if coarse_cell_for_coord(anchor) == neighbor)
            exhaustion = max(exhaustion, 0.45 * neighbor_count)
        return min(float(exhaustion), 4.0)

    def _target_approach_cells(
        self,
        frame: np.ndarray,
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        actor_component: FrameComponent,
        *,
        top_target: Coord,
    ) -> tuple[tuple[Coord, float], ...]:
        actor_cell = coarse_cell_for_coord(actor_component.anchor)
        merged: dict[CoarseCell, tuple[Coord, float]] = {}
        candidate_components = sorted(
            (
                component
                for component in extract_components(frame)
                if component.feature != actor_component.feature
            ),
            key=lambda component: (
                abs(component.anchor[0] - top_target[0]) + abs(component.anchor[1] - top_target[1]),
                -interaction_target_priority(component, memory),
            ),
        )[:4]
        for component in candidate_components:
            top, left, bottom, right = component.bounds
            center_row = (top + bottom) // 2
            center_col = (left + right) // 2
            approach_coords = {
                (center_row, left - 1),
                (center_row, right + 1),
                (top - 1, center_col),
                (bottom + 1, center_col),
            }
            for coord in approach_coords:
                row, col = coord
                if not (0 <= row < frame.shape[0] and 0 <= col < frame.shape[1]):
                    continue
                cell = coarse_cell_for_coord(coord)
                if cell == actor_cell:
                    continue
                score = 0.18
                score += 0.24 * level_memory.target_value_bonus(component.feature, cell)
                score += 0.14 * level_memory.subgoal_bonus(cell)
                score += 0.18 * memory.target_affordance_score(component.feature, GameAction.ACTION5.name)
                score += 0.08 * interaction_target_priority(component, memory)
                score -= 0.2 * level_memory.blocked_target_penalty(cell)
                score -= 0.14 * self._approach_cell_exhaustion(level_memory, cell)
                if coord not in level_memory.actor_positions:
                    score += 0.12
                if cell in level_memory.active_target_cells():
                    score += 0.08
                current = merged.get(cell)
                if current is None or score > current[1]:
                    merged[cell] = (coord, score)
        ranked = sorted(merged.values(), key=lambda item: item[1], reverse=True)
        return tuple(ranked[:6])

    def _target_option_bundle_plan(
        self,
        frame: np.ndarray,
        candidates: tuple[ActionCandidate, ...],
        available_actions: tuple[GameAction, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        *,
        actor_component: FrameComponent,
        top_target: Coord,
        top_target_score: float,
        planning_target: Coord,
        planning_target_score: float,
        planning_target_value: float,
        planning_blocked_penalty: float,
        bridge_bonus: float,
    ) -> OptionBundlePlan | None:
        motion_models: list[tuple[ActionCandidate, Coord, float, ActionStats]] = []
        for candidate in candidates:
            if candidate.action in {GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7}:
                continue
            delta, confidence = memory.best_delta(candidate.action.name)
            if delta is None or confidence < self.tuning.target_option_action_confidence_min:
                continue
            motion_models.append((candidate, delta, confidence, memory.stat_for(candidate.key)))
        if not motion_models:
            return None

        actor_cell = coarse_cell_for_coord(actor_component.anchor)
        planning_cell = coarse_cell_for_coord(planning_target)
        target_cell = coarse_cell_for_coord(top_target)
        target_cells = set(level_memory.active_target_cells())
        target_cells.add(planning_cell)
        target_cells.add(target_cell)
        blocked_cells = coarse_blocked_cells_for_components(
            frame,
            actor_feature=actor_component.feature,
            target_cells=frozenset(target_cells),
        )
        initial_path = coarse_shortest_path(
            actor_cell,
            planning_cell,
            blocked=blocked_cells - {actor_cell, planning_cell},
        )
        if actor_cell != planning_cell and not initial_path:
            return None

        target_components = [
            component
            for component in extract_components(frame)
            if component.feature != actor_component.feature
        ]
        interact_candidate = next((candidate for candidate in candidates if candidate.action == GameAction.ACTION5), None)
        click_candidates = tuple(candidate for candidate in candidates if candidate.action == GameAction.ACTION6)
        hint = level_memory.mechanic_hint
        max_actions = max(2, self.tuning.target_option_total_actions)
        beam_width = 8
        accepted: list[OptionBundlePlan] = []

        @dataclass(frozen=True, slots=True)
        class OptionState:
            anchor: Coord
            bounds: tuple[int, int, int, int]
            sequence: tuple[ActionCandidate, ...]
            visited_cells: tuple[CoarseCell, ...]
            score: float

        beam: list[OptionState] = [
            OptionState(
                anchor=actor_component.anchor,
                bounds=actor_component.bounds,
                sequence=(),
                visited_cells=(actor_cell,),
                score=0.0,
            )
        ]

        def register_option(
            state: OptionState,
            finisher: ActionCandidate | None,
            *,
            finisher_kind: str,
            finisher_score: float,
        ) -> None:
            sequence = state.sequence if finisher is None else state.sequence + (finisher,)
            if len(sequence) < 2:
                return
            path = coarse_shortest_path(
                actor_cell,
                coarse_cell_for_coord(state.anchor),
                blocked=blocked_cells - {actor_cell, coarse_cell_for_coord(state.anchor)},
            )
            accepted.append(
                OptionBundlePlan(
                    first_candidate=sequence[0],
                    follow_ups=sequence[1:],
                    score=state.score + finisher_score,
                    margin=0.0,
                    target=planning_target,
                    path=path,
                    finisher_kind=finisher_kind,
                )
            )

        for _depth in range(max_actions):
            expanded: list[OptionState] = []
            for state in beam:
                current_cell = coarse_cell_for_coord(state.anchor)
                current_path = coarse_path_distance(
                    current_cell,
                    planning_cell,
                    blocked=blocked_cells - {current_cell, planning_cell},
                )
                remaining_path = coarse_shortest_path(
                    current_cell,
                    planning_cell,
                    blocked=blocked_cells - {current_cell, planning_cell},
                )
                waypoint_cell = remaining_path[0] if remaining_path else planning_cell
                current_waypoint_distance = abs(current_cell[0] - waypoint_cell[0]) + abs(current_cell[1] - waypoint_cell[1])
                current_target_distance = abs(state.anchor[0] - top_target[0]) + abs(state.anchor[1] - top_target[1])
                self.diagnostics.target_option_paths_considered += 1

                if interact_candidate is not None:
                    best_support = 0.0
                    for component in target_components:
                        gap = bounds_gap(state.bounds, component.bounds)
                        if gap > 1:
                            continue
                        support = memory.target_affordance_score(component.feature, interact_candidate.action.name)
                        support += 0.22 * level_memory.target_value_bonus(
                            component.feature,
                            coarse_cell_for_coord(component.anchor),
                        )
                        if coarse_cell_for_coord(component.anchor) == target_cell:
                            support += 0.14 * max(top_target_score, 0.0)
                        best_support = max(best_support, support)
                    if best_support > 0.0:
                        interact_score = 0.28 * planning_target_score + 0.22 * planning_target_value + 0.38 * best_support
                        if hint is not None and hint.mode in {"INTERACT", "MIXED"}:
                            interact_score += 0.18 * hint.confidence
                        register_option(
                            state,
                            interact_candidate,
                            finisher_kind="INTERACT",
                            finisher_score=interact_score,
                        )

                if click_candidates and current_target_distance <= 10:
                    best_click: ActionCandidate | None = None
                    best_click_score = float("-inf")
                    for click_candidate in click_candidates:
                        if click_candidate.data is None:
                            continue
                        click_coord = (int(click_candidate.data["y"]), int(click_candidate.data["x"]))
                        click_distance = abs(click_coord[0] - top_target[0]) + abs(click_coord[1] - top_target[1])
                        if click_distance > 12:
                            continue
                        click_stats = memory.stat_for(click_candidate.key)
                        score = (
                            0.22 * max(click_stats.mean_reward, 0.0)
                            + 0.4 * click_stats.change_rate
                            + 0.55 * click_stats.level_gain_rate
                            + 0.26 * planning_target_score
                            + 0.16 * planning_target_value
                            + max(0.0, 0.3 - 0.03 * click_distance)
                        )
                        if hint is not None and hint.mode in {"CLICK", "MIXED"}:
                            score += 0.14 * hint.confidence
                        if score > best_click_score:
                            best_click_score = score
                            best_click = click_candidate
                    if best_click is not None:
                        register_option(
                            state,
                            best_click,
                            finisher_kind="CLICK",
                            finisher_score=best_click_score,
                        )

                if len(state.sequence) >= max_actions:
                    register_option(state, None, finisher_kind="MOVE", finisher_score=0.0)
                    continue

                for candidate, delta, confidence, stats in motion_models:
                    next_anchor = (state.anchor[0] + delta[0], state.anchor[1] + delta[1])
                    if not (0 <= next_anchor[0] < frame.shape[0] and 0 <= next_anchor[1] < frame.shape[1]):
                        continue
                    next_cell = coarse_cell_for_coord(next_anchor)
                    next_path = coarse_path_distance(
                        next_cell,
                        planning_cell,
                        blocked=blocked_cells - {next_cell, planning_cell},
                    )
                    if next_path >= 99:
                        continue
                    next_waypoint_distance = abs(next_cell[0] - waypoint_cell[0]) + abs(next_cell[1] - waypoint_cell[1])
                    next_target_distance = abs(next_anchor[0] - top_target[0]) + abs(next_anchor[1] - top_target[1])
                    path_improvement = current_path - next_path if current_path < 99 else 0
                    waypoint_improvement = current_waypoint_distance - next_waypoint_distance
                    pixel_improvement = current_target_distance - next_target_distance
                    revisit_penalty = 0.18 if next_anchor in level_memory.actor_positions else 0.0
                    if next_cell in state.visited_cells:
                        revisit_penalty += 0.1
                    step_score = (
                        0.42 * confidence
                        + 0.18 * max(stats.change_rate, 0.0)
                        + 0.18 * max(stats.level_gain_rate, 0.0)
                        + 0.22 * max(path_improvement, 0)
                        + 0.14 * max(waypoint_improvement, 0)
                        + 0.08 * max(pixel_improvement, 0)
                        + 0.1 * planning_target_score
                        + 0.12 * planning_target_value
                        + 0.08 * bridge_bonus
                        + 0.08 * level_memory.subgoal_bonus(next_cell)
                        - 0.1 * planning_blocked_penalty
                        - revisit_penalty
                        - 0.08 * min(level_memory.keyboard_no_change_streaks.get(candidate.action.name, 0), 3)
                    )
                    if next_cell == planning_cell:
                        step_score += 0.2
                    if next_cell == target_cell:
                        step_score += 0.12
                    if hint is not None and hint.mode in {"MOVE", "MIXED"}:
                        step_score += 0.08 * hint.confidence
                    if path_improvement <= 0 and waypoint_improvement <= 0 and pixel_improvement <= 0:
                        step_score -= 0.22
                    expanded.append(
                        OptionState(
                            anchor=next_anchor,
                            bounds=(
                                state.bounds[0] + delta[0],
                                state.bounds[1] + delta[1],
                                state.bounds[2] + delta[0],
                                state.bounds[3] + delta[1],
                            ),
                            sequence=state.sequence + (candidate,),
                            visited_cells=state.visited_cells + (next_cell,),
                            score=state.score + step_score,
                        )
                    )
            if not expanded:
                break
            expanded.sort(key=lambda item: item.score, reverse=True)
            beam = expanded[:beam_width]
            for state in beam:
                if len(state.sequence) >= 2:
                    register_option(state, None, finisher_kind="MOVE", finisher_score=0.0)

        if not accepted:
            return None
        accepted.sort(key=lambda item: item.score, reverse=True)
        best = accepted[0]
        second_score = accepted[1].score if len(accepted) > 1 else 0.0
        margin = best.score - second_score
        if best.score < self.tuning.target_option_min_score or margin < self.tuning.target_option_margin_floor:
            return None
        return OptionBundlePlan(
            first_candidate=best.first_candidate,
            follow_ups=best.follow_ups,
            score=best.score,
            margin=margin,
            target=best.target,
            path=best.path,
            finisher_kind=best.finisher_kind,
        )

    def _seed_target_option_bundle(
        self,
        frame: np.ndarray,
        candidates: tuple[ActionCandidate, ...],
        available_actions: tuple[GameAction, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
        *,
        actor_component: FrameComponent,
        top_target: Coord,
        top_target_score: float,
        planning_target: Coord,
        planning_target_score: float,
        planning_target_value: float,
        planning_blocked_penalty: float,
        bridge_bonus: float,
    ) -> bool:
        if level_memory.macro_queue:
            return False
        plan = self._target_option_bundle_plan(
            frame,
            candidates,
            available_actions,
            memory,
            level_memory,
            actor_component=actor_component,
            top_target=top_target,
            top_target_score=top_target_score,
            planning_target=planning_target,
            planning_target_score=planning_target_score,
            planning_target_value=planning_target_value,
            planning_blocked_penalty=planning_blocked_penalty,
            bridge_bonus=bridge_bonus,
        )
        if plan is None or not plan.follow_ups:
            return False
        level_memory.macro_queue.extend(plan.follow_ups)
        level_memory.macro_source_step = level_memory.step_index
        self.diagnostics.target_option_bundle_injections += 1
        self.diagnostics.target_option_bundle_actions += len(plan.follow_ups)
        return True

    def _interaction_state_key(
        self,
        frame: np.ndarray,
        available_actions: tuple[GameAction, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
    ) -> InteractionStateKey:
        actor_component = resolved_actor_component(frame, memory, level_memory)
        actor_row = actor_col = -1
        target_row = target_col = -1
        if actor_component is not None:
            actor_row, actor_col = coarse_cell_for_coord(actor_component.anchor)
            targets = self._interaction_targets(frame, memory, level_memory, actor_component)
            if targets:
                target_row, target_col = coarse_cell_for_coord(targets[0][0])
        else:
            target_cell = self._target_cell_without_actor(frame, memory, level_memory)
            if target_cell is not None:
                target_row, target_col = target_cell
        contact_flag = int(bool(level_memory.active_target_contacts() or level_memory.active_target_cells()))
        actor_on_target = int(
            actor_row >= 0 and target_row >= 0 and actor_row == target_row and actor_col == target_col
        )
        return (actor_row, actor_col, target_row, target_col, contact_flag, actor_on_target, action_mask(available_actions))

    def _interaction_graph_state_distance(self, interaction_key: InteractionStateKey) -> int | None:
        actor_row, actor_col, target_row, target_col, _contact_flag, _actor_on_target, _mask = interaction_key
        if actor_row < 0 or target_row < 0:
            return None
        return abs(actor_row - target_row) + abs(actor_col - target_col)

    def _interaction_graph_transition_value(
        self,
        current_key: InteractionStateKey,
        next_key: InteractionStateKey,
    ) -> float:
        current_distance = self._interaction_graph_state_distance(current_key)
        next_distance = self._interaction_graph_state_distance(next_key)
        current_contact = current_key[4]
        next_contact = next_key[4]
        current_actor_on_target = current_key[5]
        next_actor_on_target = next_key[5]
        value = 0.0
        if current_distance is not None and next_distance is not None:
            improvement = current_distance - next_distance
            value += 0.22 * improvement
            if improvement > 0:
                value += 0.08 * min(improvement, 2)
            elif improvement < 0:
                value -= 0.1 * min(-improvement, 2)
        elif next_distance is not None:
            value += 0.08
        if next_contact > current_contact:
            value += 0.42
        elif next_contact < current_contact:
            value -= 0.1
        if next_actor_on_target > current_actor_on_target:
            value += 0.48
        actor_changed = next_key[:2] != current_key[:2] and next_key[0] >= 0
        target_changed = next_key[2:4] != current_key[2:4] and next_key[2] >= 0
        if actor_changed:
            value += 0.14
        if target_changed:
            value += 0.08
        return value

    def _interaction_graph_decision(
        self,
        frame: np.ndarray,
        candidates: tuple[ActionCandidate, ...],
        available_actions: tuple[GameAction, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
    ) -> tuple[ActionCandidate | None, float, float]:
        if level_memory.step_index < 5:
            return None, 0.0, 0.0
        current_key = self._interaction_state_key(frame, available_actions, memory, level_memory)
        revisit_count = level_memory.interaction_revisit_count(current_key)
        pressure = level_memory.loop_pressure(frame_signature(frame))
        if revisit_count < 2 and pressure < 4:
            return None, 0.0, 0.0

        best_candidate: ActionCandidate | None = None
        best_score = float("-inf")
        second_score = float("-inf")
        for candidate in candidates:
            if candidate.action == GameAction.ACTION6:
                continue
            action_stats = level_memory.interaction_graph_action_stats.get((current_key, candidate.action.name))
            family = action_family(candidate.action)
            family_stats = level_memory.interaction_graph_family_stats.get((current_key, family))
            if action_stats is None and family_stats is None:
                continue
            if (
                (action_stats is None or action_stats.attempts < 2)
                and (family_stats is None or family_stats.attempts < 3)
                and pressure < 5
            ):
                continue
            score = 0.0
            if action_stats is not None:
                score += (
                    0.25 * max(action_stats.mean_reward, 0.0)
                    + 0.7 * action_stats.change_rate
                    + 1.1 * action_stats.level_gain_rate
                    + 0.03 * min(action_stats.attempts, 6)
                )
            if family_stats is not None:
                score += (
                    0.12 * max(family_stats.mean_reward, 0.0)
                    + 0.35 * family_stats.change_rate
                    + 0.55 * family_stats.level_gain_rate
                )

            transition_edges = [
                (next_key, count)
                for (state_key, action_name, next_key), count in level_memory.interaction_graph_transitions.items()
                if state_key == current_key and action_name == candidate.action.name
            ]
            transition_total = sum(count for _next_key, count in transition_edges)
            if transition_total < 2 and (action_stats is None or action_stats.level_gain_rate == 0.0) and pressure < 5:
                continue
            transition_bonus = 0.0
            if transition_edges:
                transition_bonus = sum(
                    count * self._interaction_graph_transition_value(current_key, next_key)
                    for next_key, count in transition_edges
                ) / max(transition_total, 1)
                score += transition_bonus
                if any(next_key[5] > current_key[5] for next_key, _count in transition_edges):
                    score += 0.18
                if any(next_key[:2] != current_key[:2] for next_key, _count in transition_edges):
                    score += 0.08
            if (
                transition_bonus <= 0.1
                and (action_stats is None or action_stats.level_gain_rate == 0.0)
                and (family_stats is None or family_stats.level_gain_rate == 0.0)
            ):
                continue
            if family == "UNDO" and current_key[4] == 0:
                score -= 0.12
            if family == "INTERACT" and current_key[3] >= 0 and current_key[5] == 1:
                score += 0.08
            score += 0.05 * min(revisit_count, 3)
            score += 0.04 * min(pressure, 4)

            if score > best_score:
                second_score = best_score
                best_score = score
                best_candidate = candidate
            elif score > second_score:
                second_score = score

        if best_candidate is None:
            return None, 0.0, 0.0
        margin = best_score - (second_score if second_score != float("-inf") else 0.0)
        return best_candidate, best_score, margin

    def _target_cell_without_actor(
        self,
        frame: np.ndarray,
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
    ) -> CoarseCell | None:
        active_cells = level_memory.active_target_cells()
        if active_cells:
            ranked_active = sorted(
                active_cells,
                key=lambda cell: (
                    level_memory.subgoal_bonus(cell)
                    + next((score for hotspot_cell, score in hotspot_cells(memory) if hotspot_cell == cell), 0.0)
                ),
                reverse=True,
            )
            return ranked_active[0]
        components = extract_components(frame)
        ranked_components = sorted(
            components,
            key=lambda component: interaction_target_priority(component, memory) + self._mechanic_focus_bonus(
                component, frame, memory, level_memory
            ),
            reverse=True,
        )
        if ranked_components:
            return coarse_cell_for_coord(ranked_components[0].anchor)
        hotspots = hotspot_cells(memory)
        if hotspots:
            return hotspots[0][0]
        return None

    def _abstract_state_key(
        self,
        frame: np.ndarray,
        available_actions: tuple[GameAction, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
    ) -> AbstractStateKey:
        hint = level_memory.mechanic_hint
        mode = hint.mode if hint is not None else "UNKNOWN"
        goal = hint.goal if hint is not None else "UNKNOWN"
        actor_component = resolved_actor_component(frame, memory, level_memory)
        actor_row = actor_col = -1
        target_row = target_col = -1
        if actor_component is not None:
            actor_row, actor_col = coarse_cell_for_coord(actor_component.anchor)
            targets = self._interaction_targets(frame, memory, level_memory, actor_component)
            if targets:
                target_row, target_col = coarse_cell_for_coord(targets[0][0])
        else:
            target_cell = self._target_cell_without_actor(frame, memory, level_memory)
            if target_cell is not None:
                target_row, target_col = target_cell
        return (mode, goal, actor_row, actor_col, target_row, target_col, action_mask(available_actions))

    def _abstract_frontier_state_score(
        self,
        abstract_key: AbstractStateKey,
        current_abstract_key: AbstractStateKey,
    ) -> float:
        mode, goal, actor_row, actor_col, target_row, target_col, mask = abstract_key
        current_mode, current_goal, current_actor_row, current_actor_col, current_target_row, current_target_col, _ = (
            current_abstract_key
        )
        score = 0.0
        if target_row >= 0 and target_col >= 0:
            score += 0.25
        if mode in {"CLICK", "MIXED"} and (mask & 1):
            score += 0.18
        if mode in {"MOVE", "INTERACT", "MIXED"} and (mask & 2):
            score += 0.16
        if mode == "INTERACT" and (mask & 4):
            score += 0.18
        if mode == "UNDO" and (mask & 8):
            score += 0.1
        if actor_row >= 0 and target_row >= 0:
            distance = abs(actor_row - target_row) + abs(actor_col - target_col)
            score += 0.8 / (1.0 + distance)
        elif target_row >= 0:
            score += 0.15
        if goal in {"CLEAR", "COLLECT", "TOGGLE", "CONTACT"}:
            score += 0.08
        if goal != current_goal:
            score += 0.05
        if mode != current_mode:
            score += 0.08
        if (target_row, target_col) != (current_target_row, current_target_col):
            score += 0.08
        if (actor_row, actor_col) != (current_actor_row, current_actor_col):
            score += 0.05
        return score

    def _planned_rollout_decision(
        self,
        frame: np.ndarray,
        available_actions: tuple[GameAction, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
    ) -> tuple[ActionCandidate | None, float, float]:
        hint = level_memory.mechanic_hint
        keyboard = keyboard_actions(available_actions)
        if not keyboard or level_memory.step_index < len(keyboard):
            return None, 0.0, 0.0
        actor_component = component_by_actor_prior(frame, memory)
        if actor_component is None:
            return None, 0.0, 0.0

        action_models: list[tuple[GameAction, Coord, float, bool]] = []
        for action in keyboard:
            if action == GameAction.ACTION5:
                stats = memory.stat_for((action.name, None))
                confidence = 0.15 + 0.35 * stats.change_rate + 0.5 * stats.level_gain_rate
                action_models.append((action, (0, 0), confidence, True))
                continue
            delta, confidence = memory.best_delta(action.name)
            if delta is None or confidence < 0.4:
                continue
            action_models.append((action, delta, confidence, False))
        if not action_models:
            return None, 0.0, 0.0

        target_components = [
            component
            for component in extract_components(frame)
            if component.feature != actor_component.feature
        ]
        if not target_components:
            return None, 0.0, 0.0

        target_start_distances = {
            component.feature: manhattan_to_bounds(actor_component.anchor, component.bounds)
            for component in target_components
        }
        active_target_cells = level_memory.active_target_cells()
        max_depth = 2
        if hint is not None and hint.confidence >= 0.42:
            if hint.mode in {"INTERACT", "MIXED"} or hint.goal in {"CONTACT", "COLLECT", "TOGGLE"}:
                max_depth = 3
        pressure = level_memory.loop_pressure(frame_signature(frame))

        best_action: GameAction | None = None
        best_score = 0.0
        second_score = float("-inf")
        for first_action, first_delta, first_confidence, first_is_interact in action_models:
            sequences = [((first_action, first_delta, first_confidence, first_is_interact),)]
            for second_action, second_delta, second_confidence, second_is_interact in action_models:
                second_sequence = (
                    (first_action, first_delta, first_confidence, first_is_interact),
                    (second_action, second_delta, second_confidence, second_is_interact),
                )
                sequences.append(second_sequence)
                if max_depth >= 3:
                    for third_action, third_delta, third_confidence, third_is_interact in action_models:
                        sequences.append(
                            second_sequence
                            + ((third_action, third_delta, third_confidence, third_is_interact),)
                        )
            for sequence in sequences:
                anchor = actor_component.anchor
                seen_predicted: set[Coord] = set()
                score = -0.3 * level_memory.keyboard_no_change_streaks.get(first_action.name, 0)
                if hint is not None:
                    score += 0.08 * min(pressure, 4)
                    if hint.mode == "MOVE" and not first_is_interact:
                        score += 0.14 * hint.confidence
                    elif hint.mode == "INTERACT" and first_is_interact:
                        score += 0.22 * hint.confidence
                previous_action: GameAction | None = None
                for step_index, (action, delta, confidence, is_interact) in enumerate(sequence, start=1):
                    previous_anchor = anchor
                    anchor = (anchor[0] + delta[0], anchor[1] + delta[1])
                    if not (0 <= anchor[0] < frame.shape[0] and 0 <= anchor[1] < frame.shape[1]):
                        score -= 2.0
                        break
                    if hint is not None:
                        if hint.mode == "MOVE":
                            score += (0.12 if not is_interact else -0.18) * hint.confidence
                        elif hint.mode == "INTERACT":
                            score += (0.18 if is_interact else 0.05) * hint.confidence
                        elif hint.mode == "MIXED":
                            score += (0.1 if not is_interact else 0.12) * hint.confidence
                    if is_interact:
                        score -= 0.1
                    elif anchor in seen_predicted or anchor in level_memory.actor_positions:
                        score -= 0.25
                    else:
                        score += 1.0 / step_index
                    if not is_interact:
                        score -= self._control_repeat_penalty(
                            action_key=(action.name, None),
                            current_anchor=previous_anchor,
                            next_anchor=anchor,
                            level_memory=level_memory,
                            interaction_targets=tuple(
                                (component.anchor, interaction_target_priority(component, memory))
                                for component in target_components[:4]
                            ),
                            pressure=pressure,
                        )
                    score += 0.35 * confidence
                    if previous_action is not None and action != previous_action:
                        score += 0.2
                    current_cell = coarse_cell_for_coord(anchor)
                    if active_target_cells:
                        best_contact_bonus = 0.0
                        for contact_cell in active_target_cells:
                            cell_distance = abs(current_cell[0] - contact_cell[0]) + abs(current_cell[1] - contact_cell[1])
                            contact_bonus = 0.55 / (1.0 + cell_distance)
                            contact_bonus += 0.25 * level_memory.subgoal_bonus(contact_cell)
                            if current_cell == contact_cell:
                                contact_bonus += 0.45
                                if is_interact:
                                    contact_bonus += 0.2
                            best_contact_bonus = max(best_contact_bonus, contact_bonus)
                        score += best_contact_bonus
                    best_target_bonus = 0.0
                    for component in target_components:
                        start_distance = target_start_distances[component.feature]
                        distance = manhattan_to_bounds(anchor, component.bounds)
                        target_priority = interaction_target_priority(component, memory)
                        next_action_name = None
                        if step_index < len(sequence):
                            next_action_name = sequence[step_index][0].name
                        affordance_bonus = 0.0
                        if distance <= 1:
                            affordance_bonus += 0.4 * memory.target_affordance_score(component.feature, action.name)
                            if next_action_name is not None:
                                affordance_bonus += memory.target_affordance_score(
                                    component.feature,
                                    next_action_name,
                                )
                        improvement = max(0, start_distance - distance)
                        bonus = target_priority * (0.18 * improvement + 0.45 / (1.0 + distance))
                        bonus += 0.9 * affordance_bonus
                        if is_interact and distance > 1:
                            bonus -= min(distance, 4) * 0.2
                        if distance <= 1:
                            bonus += target_priority * (0.7 if distance == 0 else 0.35)
                            if step_index < len(sequence):
                                bonus += 0.25
                        if hint is not None:
                            if hint.goal in {"CONTACT", "COLLECT", "TOGGLE"}:
                                bonus += hint.confidence * 0.12 * improvement
                            if hint.goal == "ALIGN":
                                row_aligned = component.bounds[0] <= anchor[0] <= component.bounds[2]
                                col_aligned = component.bounds[1] <= anchor[1] <= component.bounds[3]
                                if row_aligned or col_aligned:
                                    bonus += 0.22 * hint.confidence
                            if hint.mode == "INTERACT":
                                if is_interact and distance <= 1:
                                    bonus += 0.55 * hint.confidence
                                elif is_interact and distance > 1:
                                    bonus -= 0.18 * hint.confidence * min(distance, 4) / 4.0
                            elif hint.mode == "MOVE" and is_interact:
                                bonus -= 0.08 * hint.confidence
                        best_target_bonus = max(best_target_bonus, bonus)
                    score += best_target_bonus
                    seen_predicted.add(anchor)
                    previous_action = action
                if score > best_score:
                    second_score = best_score
                    best_score = score
                    best_action = first_action
                elif score > second_score:
                    second_score = score

        min_score = 0.9
        if best_action is None or best_score < min_score:
            return None, best_score, 0.0
        margin = best_score - max(second_score, float("-inf"))
        if second_score == float("-inf"):
            margin = best_score
        return (
            ActionCandidate(
                action=best_action,
                data=None,
                key=(best_action.name, None),
                label=best_action.name,
            ),
            best_score,
            margin,
        )

    def _planned_rollout_action(
        self,
        frame: np.ndarray,
        available_actions: tuple[GameAction, ...],
        memory: EnvironmentMemory,
        level_memory: LevelMemory,
    ) -> ActionCandidate | None:
        decision, _score, _margin = self._planned_rollout_decision(
            frame,
            available_actions,
            memory,
            level_memory,
        )
        return decision
