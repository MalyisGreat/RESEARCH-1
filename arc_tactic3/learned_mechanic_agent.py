from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from .agents import BaseAgent, FrontierGraphPolicy, LevelOutcome
from .core import Action, ClickAction, Coord, Family
from .dsl import EnvironmentCase, HiddenMechanicEnvironment
from .hypotheses import MechanicPosterior
from .parser import ParsedFrame, build_state, derive_static_scene, parse_frame
from .planner import PlannerResult, plan_with_hypothesis
from .prior import PriorPrediction, summarize_state_for_model


FAMILY_TO_INDEX: dict[Family, int] = {
    "reach_goal": 0,
    "key_goal": 1,
    "switch_goal": 2,
    "push_box": 3,
    "portal_goal": 4,
}
INDEX_TO_FAMILY: dict[int, Family] = {value: key for key, value in FAMILY_TO_INDEX.items()}

CLICK_TO_INDEX: dict[str | None, int] = {None: 0, "switch": 1, "teleport": 2}
INDEX_TO_CLICK: dict[int, str | None] = {value: key for key, value in CLICK_TO_INDEX.items()}

DIRECTION_TO_INDEX: dict[Coord | None, int] = {
    (-1, 0): 0,
    (1, 0): 1,
    (0, -1): 2,
    (0, 1): 3,
    None: 4,
}
INDEX_TO_DIRECTION: dict[int, Coord | None] = {value: key for key, value in DIRECTION_TO_INDEX.items()}

SPECIAL_TOKENS = ("<pad>", "<unk>", "<step>", "<bos>")
TEXT_TOKEN_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class SharedTrajectorySample:
    tokens: tuple[str, ...]
    family_index: int
    click_index: int
    direction_indices: tuple[int, ...]
    direction_mask: tuple[int, ...]
    effect_indices: tuple[int, ...]
    effect_mask: tuple[int, ...]
    action_target: int
    action_mask: tuple[int, ...]
    trace_steps: int


@dataclass(frozen=True, slots=True)
class TransitionSummary:
    action_kind: str
    action_slot: int
    clicked_kind: str | None
    player_move: Coord
    keys_delta: int
    doors_delta: int
    boxes_delta: int
    solved_delta: int
    frame_changed: bool
    has_key_changed: bool
    switch_changed: bool
    boxes_moved: bool


@dataclass(frozen=True, slots=True)
class AffordanceTransferSummary:
    family_scores: dict[Family, float]
    click_mode_scores: dict[str | None, float]
    action_kind_scores: dict[str, float]
    clicked_kind_scores: dict[str, float]
    state_change_scores: dict[str, float]

    def to_prior_prediction(self, *, mode: str = "family_click") -> PriorPrediction:
        if mode == "family_click":
            family_scores = dict(self.family_scores)
            click_mode_scores = dict(self.click_mode_scores)
        elif mode == "click_only":
            family_scores = {}
            click_mode_scores = dict(self.click_mode_scores)
        elif mode == "family_only":
            family_scores = dict(self.family_scores)
            click_mode_scores = {}
        elif mode == "none":
            family_scores = {}
            click_mode_scores = {}
        else:
            raise ValueError(f"unsupported affordance prior mode: {mode}")
        return PriorPrediction(
            family_scores=family_scores,
            click_mode_scores=click_mode_scores,
            button_direction_scores={},
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int
    embedding_dim: int = 64
    hidden_dim: int = 128
    slot_count: int = 5
    dropout: float = 0.1
    num_layers: int = 1
    pooling: str = "last"


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = 1337
    train_traces_per_level: int = 6
    eval_traces_per_level: int = 3
    rollout_steps: int = 8
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    embedding_dim: int = 64
    hidden_dim: int = 128
    explore_budget: int = 5
    confidence_threshold: float = 0.58
    text_manual_probability: float = 0.0
    teacher_rollin_probability: float = 0.0
    num_layers: int = 1
    pooling: str = "last"
    direction_loss_weight: float = 1.0
    effect_loss_weight: float = 0.75
    action_loss_weight: float = 0.75
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True, slots=True)
class PredictionBundle:
    family_probs: tuple[float, ...]
    click_probs: tuple[float, ...]
    direction_probs: tuple[tuple[float, ...], ...]
    effect_probs: tuple[tuple[float, ...], ...]
    action_probs: tuple[float, ...]

    @property
    def confidence(self) -> float:
        family = max(self.family_probs) if self.family_probs else 0.0
        click = max(self.click_probs) if self.click_probs else 0.0
        direction = (
            sum(max(slot) for slot in self.direction_probs) / max(len(self.direction_probs), 1)
            if self.direction_probs
            else 0.0
        )
        action = max(self.action_probs) if self.action_probs else 0.0
        return float((family + click + direction + action) / 4.0)


class LearnedProposalPrior:
    def __init__(
        self,
        agent: "LearnedMechanicAgent",
        *,
        use_family_prior: bool = True,
        use_click_prior: bool = True,
        use_direction_prior: bool = True,
        floor: float = 0.05,
        power: float = 1.0,
        family_weight: float = 1.0,
        click_weight: float = 1.0,
        direction_weight: float = 1.0,
    ) -> None:
        self.agent = agent
        self.use_family_prior = use_family_prior
        self.use_click_prior = use_click_prior
        self.use_direction_prior = use_direction_prior
        self.floor = floor
        self.power = power
        self.family_weight = max(0.0, float(family_weight))
        self.click_weight = max(0.0, float(click_weight))
        self.direction_weight = max(0.0, float(direction_weight))

    def _scale(self, score: float) -> float:
        bounded = max(0.0, float(score))
        if self.power != 1.0:
            bounded = bounded**self.power
        return max(self.floor, bounded)

    def predict(
        self,
        state,
        *,
        available_buttons: tuple[str, ...],
        allows_click: bool,
        allows_undo: bool,
    ) -> PriorPrediction:
        prediction = self.agent.predict_tokens(
            build_state_snapshot_tokens(
                state,
                available_buttons=available_buttons,
                allows_click=allows_click,
                allows_undo=allows_undo,
            )
        )
        family_scores: dict[Family, float] = {}
        click_mode_scores: dict[str | None, float] = {}
        button_direction_scores: dict[str, dict[Coord | None, float]] = {}

        if self.use_family_prior and self.family_weight > 0.0:
            family_scores = {
                INDEX_TO_FAMILY[index]: self._scale(float(score)) * self.family_weight
                for index, score in enumerate(prediction.family_probs)
            }
        if self.use_click_prior and self.click_weight > 0.0:
            click_mode_scores = {
                INDEX_TO_CLICK[index]: self._scale(float(score)) * self.click_weight
                for index, score in enumerate(prediction.click_probs)
            }
        if self.use_direction_prior and self.direction_weight > 0.0:
            for slot_index, button in enumerate(available_buttons):
                if slot_index >= len(prediction.direction_probs):
                    break
                button_direction_scores[button] = {
                    INDEX_TO_DIRECTION[direction_index]: self._scale(float(score)) * self.direction_weight
                    for direction_index, score in enumerate(prediction.direction_probs[slot_index])
                }
        return PriorPrediction(
            family_scores=family_scores,
            click_mode_scores=click_mode_scores,
            button_direction_scores=button_direction_scores,
        )


class SharedEventTokenizer:
    def __init__(self) -> None:
        self.token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = list(SPECIAL_TOKENS)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    def fit(self, samples: Iterable[SharedTrajectorySample]) -> None:
        for sample in samples:
            for token in sample.tokens:
                if token not in self.token_to_id:
                    self.token_to_id[token] = len(self.id_to_token)
                    self.id_to_token.append(token)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def to_dict(self) -> dict[str, object]:
        return {"token_to_id": self.token_to_id, "id_to_token": self.id_to_token}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SharedEventTokenizer":
        instance = cls()
        instance.token_to_id = {str(key): int(value) for key, value in dict(payload["token_to_id"]).items()}
        instance.id_to_token = [str(token) for token in list(payload["id_to_token"])]
        return instance


class SharedTrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, samples: list[SharedTrajectorySample], tokenizer: SharedEventTokenizer) -> None:
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "input_ids": torch.tensor(self.tokenizer.encode(sample.tokens), dtype=torch.long),
            "length": torch.tensor(len(sample.tokens), dtype=torch.long),
            "family": torch.tensor(sample.family_index, dtype=torch.long),
            "click": torch.tensor(sample.click_index, dtype=torch.long),
            "directions": torch.tensor(sample.direction_indices, dtype=torch.long),
            "direction_mask": torch.tensor(sample.direction_mask, dtype=torch.float32),
            "effects": torch.tensor(sample.effect_indices, dtype=torch.long),
            "effect_mask": torch.tensor(sample.effect_mask, dtype=torch.float32),
            "action_target": torch.tensor(sample.action_target, dtype=torch.long),
            "action_mask": torch.tensor(sample.action_mask, dtype=torch.float32),
        }


def collate_shared_batch(batch: list[dict[str, torch.Tensor]], *, pad_id: int) -> dict[str, torch.Tensor]:
    max_length = max(int(row["length"].item()) for row in batch)
    input_ids = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    family = torch.zeros(len(batch), dtype=torch.long)
    click = torch.zeros(len(batch), dtype=torch.long)
    directions = torch.zeros((len(batch), 5), dtype=torch.long)
    direction_mask = torch.zeros((len(batch), 5), dtype=torch.float32)
    effects = torch.zeros((len(batch), 5), dtype=torch.long)
    effect_mask = torch.zeros((len(batch), 5), dtype=torch.float32)
    action_target = torch.zeros(len(batch), dtype=torch.long)
    action_mask = torch.zeros((len(batch), 7), dtype=torch.float32)
    for index, row in enumerate(batch):
        size = int(row["length"].item())
        input_ids[index, :size] = row["input_ids"]
        lengths[index] = size
        family[index] = row["family"]
        click[index] = row["click"]
        directions[index] = row["directions"]
        direction_mask[index] = row["direction_mask"]
        effects[index] = row["effects"]
        effect_mask[index] = row["effect_mask"]
        action_target[index] = row["action_target"]
        action_mask[index] = row["action_mask"]
    return {
        "input_ids": input_ids,
        "lengths": lengths,
        "family": family,
        "click": click,
        "directions": directions,
        "direction_mask": direction_mask,
        "effects": effects,
        "effect_mask": effect_mask,
        "action_target": action_target,
        "action_mask": action_mask,
    }


class MechanicInferenceModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(
            config.embedding_dim,
            config.hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        fused_dim = config.hidden_dim * 2
        self.dropout = nn.Dropout(config.dropout)
        self.pool_attention = nn.Linear(fused_dim, 1) if config.pooling in {"attention", "hybrid"} else None
        pooled_dim = fused_dim * 2 if config.pooling == "hybrid" else fused_dim
        self.family_head = nn.Linear(pooled_dim, len(FAMILY_TO_INDEX))
        self.click_head = nn.Linear(pooled_dim, len(CLICK_TO_INDEX))
        self.direction_head = nn.Linear(pooled_dim, config.slot_count * len(DIRECTION_TO_INDEX))
        self.effect_head = nn.Linear(pooled_dim, config.slot_count * len(DIRECTION_TO_INDEX))
        self.action_head = nn.Linear(pooled_dim, 7)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        padded, _ = pad_packed_sequence(encoded, batch_first=True)
        gather_index = (lengths - 1).clamp_min(0).view(-1, 1, 1).expand(-1, 1, padded.size(-1))
        last_pooled = padded.gather(1, gather_index).squeeze(1)
        if self.pool_attention is not None:
            mask = torch.arange(padded.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
            attention_logits = self.pool_attention(padded).squeeze(-1).masked_fill(~mask, -1e9)
            attention_weights = torch.softmax(attention_logits, dim=-1).unsqueeze(-1)
            attention_pooled = (padded * attention_weights).sum(dim=1)
            if self.config.pooling == "hybrid":
                pooled = torch.cat([last_pooled, attention_pooled], dim=-1)
            else:
                pooled = attention_pooled
        else:
            pooled = last_pooled
        pooled = self.dropout(pooled)
        direction_logits = self.direction_head(pooled).view(
            pooled.size(0),
            self.config.slot_count,
            len(DIRECTION_TO_INDEX),
        )
        return {
            "family_logits": self.family_head(pooled),
            "click_logits": self.click_head(pooled),
            "direction_logits": direction_logits,
            "effect_logits": self.effect_head(pooled).view(
                pooled.size(0),
                self.config.slot_count,
                len(DIRECTION_TO_INDEX),
            ),
            "action_logits": self.action_head(pooled),
        }


def _coarse_bucket(coord: Coord | None) -> tuple[int, int]:
    if coord is None:
        return (-1, -1)
    return (coord[0] // 2, coord[1] // 2)


def _movement_direction(before: Coord, after: Coord) -> Coord:
    return (after[0] - before[0], after[1] - before[1])


def _describe_objects(parsed: ParsedFrame) -> list[str]:
    tokens: list[str] = []
    counts: dict[str, int] = {}
    for obj in parsed.objects:
        counts[obj.kind] = counts.get(obj.kind, 0) + 1
        row_bucket, col_bucket = _coarse_bucket(obj.anchor)
        tokens.extend(
            (
                f"obj:{obj.kind}",
                f"objpos:{obj.kind}:r{row_bucket}",
                f"objpos:{obj.kind}:c{col_bucket}",
                f"objsize:{obj.kind}:{min(len(obj.cells), 6)}",
            )
        )
    for kind, count in sorted(counts.items()):
        tokens.append(f"count:{kind}:{min(count, 6)}")
    return tokens


def _text_channel_tokens(summary: str) -> list[str]:
    words = [piece for piece in TEXT_TOKEN_RE.split(summary.lower()) if piece]
    return [f"txt:{word[:12]}" for word in words[:48]]


def _normalize_score_map(values: dict[object, float]) -> dict[object, float]:
    filtered = {key: float(value) for key, value in values.items() if float(value) > 0.0}
    total = sum(filtered.values())
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in filtered.items()}


def summarize_affordances_from_transitions(
    transitions: Iterable[TransitionSummary],
) -> AffordanceTransferSummary:
    family_scores: dict[Family, float] = {}
    click_mode_scores: dict[str | None, float] = {}
    action_kind_scores: dict[str, float] = {}
    clicked_kind_scores: dict[str, float] = {}
    state_change_scores: dict[str, float] = {}
    useful_move_weight = 0.0
    useful_click_weight = 0.0

    for transition in transitions:
        useful = (
            transition.solved_delta > 0
            or transition.frame_changed
            or transition.player_move != (0, 0)
            or transition.keys_delta != 0
            or transition.doors_delta != 0
            or transition.boxes_delta != 0
            or transition.has_key_changed
            or transition.switch_changed
            or transition.boxes_moved
        )
        if not useful:
            continue

        weight = 1.0 + max(0, transition.solved_delta) * 2.0
        action_kind_scores[transition.action_kind] = action_kind_scores.get(transition.action_kind, 0.0) + weight
        if transition.action_kind == "move":
            useful_move_weight += weight
        elif transition.action_kind == "click":
            useful_click_weight += weight

        if transition.clicked_kind is not None:
            clicked_kind_scores[transition.clicked_kind] = clicked_kind_scores.get(transition.clicked_kind, 0.0) + weight

        if transition.keys_delta < 0 or transition.has_key_changed or transition.doors_delta < 0:
            family_scores["key_goal"] = family_scores.get("key_goal", 0.0) + (1.6 * weight)
            state_change_scores["key_progress"] = state_change_scores.get("key_progress", 0.0) + weight
        if transition.switch_changed:
            family_scores["switch_goal"] = family_scores.get("switch_goal", 0.0) + (1.7 * weight)
            click_mode_scores["switch"] = click_mode_scores.get("switch", 0.0) + (1.4 * weight)
            state_change_scores["switch_change"] = state_change_scores.get("switch_change", 0.0) + weight
        if transition.boxes_moved or transition.boxes_delta != 0:
            family_scores["push_box"] = family_scores.get("push_box", 0.0) + (1.7 * weight)
            state_change_scores["box_motion"] = state_change_scores.get("box_motion", 0.0) + weight
        if transition.clicked_kind == "portal" and transition.player_move != (0, 0):
            family_scores["portal_goal"] = family_scores.get("portal_goal", 0.0) + (1.8 * weight)
            click_mode_scores["teleport"] = click_mode_scores.get("teleport", 0.0) + (1.5 * weight)
            state_change_scores["portal_use"] = state_change_scores.get("portal_use", 0.0) + weight
        if (
            transition.action_kind == "move"
            and transition.player_move != (0, 0)
            and transition.clicked_kind is None
            and transition.keys_delta == 0
            and transition.doors_delta == 0
            and not transition.switch_changed
            and not transition.boxes_moved
        ):
            family_scores["reach_goal"] = family_scores.get("reach_goal", 0.0) + (0.8 * weight)

    if useful_move_weight > 0.0 and useful_click_weight == 0.0:
        click_mode_scores[None] = click_mode_scores.get(None, 0.0) + useful_move_weight
    elif useful_click_weight > 0.0 and useful_move_weight > 0.0:
        click_mode_scores[None] = click_mode_scores.get(None, 0.0) + (0.4 * useful_move_weight)
    elif useful_move_weight == 0.0 and useful_click_weight == 0.0:
        click_mode_scores[None] = 1.0

    if not family_scores and useful_move_weight > 0.0:
        family_scores["reach_goal"] = useful_move_weight

    return AffordanceTransferSummary(
        family_scores=_normalize_score_map(family_scores),
        click_mode_scores=_normalize_score_map(click_mode_scores),
        action_kind_scores=_normalize_score_map(action_kind_scores),
        clicked_kind_scores=_normalize_score_map(clicked_kind_scores),
        state_change_scores=_normalize_score_map(state_change_scores),
    )


def summarize_transition(
    action: Action | None,
    before_state,
    after_state,
    before_parsed: ParsedFrame | None,
    after_parsed: ParsedFrame,
    *,
    solved_before: bool,
    solved_after: bool,
    available_buttons: tuple[str, ...],
) -> TransitionSummary | None:
    if action is None or before_state is None or before_parsed is None:
        return None
    if action == "undo":
        action_kind = "undo"
        action_slot = len(available_buttons)
        clicked_kind = None
    elif isinstance(action, ClickAction):
        action_kind = "click"
        action_slot = len(available_buttons) + 1
        clicked_kind = next((obj.kind for obj in before_parsed.objects if obj.anchor == action.coord), "unknown")
    else:
        action_kind = "move"
        action_slot = available_buttons.index(action) if action in available_buttons else len(available_buttons)
        clicked_kind = None
    return TransitionSummary(
        action_kind=action_kind,
        action_slot=action_slot,
        clicked_kind=clicked_kind,
        player_move=_movement_direction(before_state.player, after_state.player),
        keys_delta=len(after_state.keys) - len(before_state.keys),
        doors_delta=len(after_state.doors) - len(before_state.doors),
        boxes_delta=len(after_state.boxes) - len(before_state.boxes),
        solved_delta=int(solved_after) - int(solved_before),
        frame_changed=before_parsed.objects != after_parsed.objects,
        has_key_changed=before_state.has_key != after_state.has_key,
        switch_changed=before_state.switch_active != after_state.switch_active,
        boxes_moved=before_state.boxes != after_state.boxes,
    )


def build_step_tokens(
    parsed: ParsedFrame,
    state,
    *,
    available_buttons: tuple[str, ...],
    allows_click: bool,
    allows_undo: bool,
    transition: TransitionSummary | None,
) -> tuple[str, ...]:
    summary = summarize_state_for_model(
        state,
        available_buttons=tuple(f"slot{index}" for index, _ in enumerate(available_buttons)),
        allows_click=allows_click,
        allows_undo=allows_undo,
    )
    tokens: list[str] = ["<step>"]
    tokens.extend((f"buttons:{len(available_buttons)}", f"click:{int(allows_click)}", f"undo:{int(allows_undo)}"))
    tokens.extend((f"state:has_key:{int(state.has_key)}", f"state:switch:{int(state.switch_active)}"))
    tokens.extend(_describe_objects(parsed))
    if transition is None:
        tokens.append("action:none")
    else:
        tokens.extend(
            (
                f"action:{transition.action_kind}",
                f"action_slot:{transition.action_slot}",
                f"move_delta:{transition.player_move[0]},{transition.player_move[1]}",
                f"keys_delta:{max(min(transition.keys_delta, 2), -2)}",
                f"doors_delta:{max(min(transition.doors_delta, 2), -2)}",
                f"boxes_delta:{max(min(transition.boxes_delta, 2), -2)}",
                f"solved_delta:{transition.solved_delta}",
                f"frame_changed:{int(transition.frame_changed)}",
                f"has_key_changed:{int(transition.has_key_changed)}",
                f"switch_changed:{int(transition.switch_changed)}",
                f"boxes_moved:{int(transition.boxes_moved)}",
            )
        )
        if transition.clicked_kind is not None:
            tokens.append(f"clicked:{transition.clicked_kind}")
    tokens.extend(_text_channel_tokens(summary))
    return tuple(tokens)


def _describe_state_objects(state) -> list[str]:
    tokens: list[str] = []
    grouped = {
        "player": (state.player,),
        "goal": tuple(sorted(state.goals)),
        "key": tuple(sorted(state.keys)),
        "door": tuple(sorted(state.doors)),
        "box": tuple(sorted(state.boxes)),
        "target": tuple(sorted(state.targets)),
        "switch": tuple(sorted(state.switches)),
        "portal": tuple(sorted(state.portals)),
    }
    for kind, coords in grouped.items():
        if not coords:
            continue
        tokens.append(f"count:{kind}:{min(len(coords), 6)}")
        for row, col in coords:
            row_bucket, col_bucket = _coarse_bucket((row, col))
            tokens.extend(
                (
                    f"obj:{kind}",
                    f"objpos:{kind}:r{row_bucket}",
                    f"objpos:{kind}:c{col_bucket}",
                    f"objsize:{kind}:1",
                )
            )
    return tokens


def build_state_snapshot_tokens(
    state,
    *,
    available_buttons: tuple[str, ...],
    allows_click: bool,
    allows_undo: bool,
) -> tuple[str, ...]:
    summary = summarize_state_for_model(
        state,
        available_buttons=tuple(f"slot{index}" for index, _ in enumerate(available_buttons)),
        allows_click=allows_click,
        allows_undo=allows_undo,
    )
    tokens: list[str] = ["<bos>", "<step>"]
    tokens.extend((f"buttons:{len(available_buttons)}", f"click:{int(allows_click)}", f"undo:{int(allows_undo)}"))
    tokens.extend((f"state:has_key:{int(state.has_key)}", f"state:switch:{int(state.switch_active)}"))
    tokens.extend(_describe_state_objects(state))
    tokens.append("action:none")
    tokens.extend(_text_channel_tokens(summary))
    return tuple(tokens)


def build_manual_tokens(case: EnvironmentCase, level_index: int, *, variant_seed: int) -> tuple[str, ...]:
    config = case.config_for_level(level_index)
    family_phrases = {
        "reach_goal": ("manual:reach_goal", "manual:go_to_goal"),
        "key_goal": ("manual:key_goal", "manual:pick_key_then_goal"),
        "switch_goal": ("manual:switch_goal", "manual:activate_switch"),
        "push_box": ("manual:push_box", "manual:push_box_to_target"),
        "portal_goal": ("manual:portal_goal", "manual:use_portal"),
    }
    click_phrase = {
        None: ("manual:click:none", "manual:no_click"),
        "switch": ("manual:click:switch", "manual:click_switch"),
        "teleport": ("manual:click:teleport", "manual:click_portal"),
    }
    phrases = list(family_phrases[config.family])
    phrases.extend(click_phrase[config.click_mode])
    for button, direction in config.button_map().items():
        direction_name = {
            (-1, 0): "up",
            (1, 0): "down",
            (0, -1): "left",
            (0, 1): "right",
            None: "none",
        }[direction]
        phrases.append(f"manual:{button}:{direction_name}")
    rng = random.Random(variant_seed)
    rng.shuffle(phrases)
    return tuple(["<step>", "text:manual"] + phrases)


def effect_indices_for_state(state, available_buttons: tuple[str, ...], config) -> tuple[int, ...]:
    from .dsl import simulate_action

    effect_indices: list[int] = []
    for button in available_buttons:
        next_state = simulate_action(state, button, config)
        delta = _movement_direction(state.player, next_state.player)
        label = delta if delta in DIRECTION_TO_INDEX else None
        if delta == (0, 0):
            label = None
        effect_indices.append(DIRECTION_TO_INDEX[label])
    return tuple(effect_indices)


def action_class_index(action: Action | None) -> int:
    if action is None:
        return 6
    if action == "undo":
        return 6
    if isinstance(action, ClickAction):
        return 5
    return int(max(0, min(4, 0 if not isinstance(action, str) else 0)))


def action_class_index_for_button(button: str, available_buttons: tuple[str, ...]) -> int:
    if button in available_buttons:
        return available_buttons.index(button)
    return 6


def action_mask_for_obs(obs) -> tuple[int, ...]:
    mask = [0] * 7
    for index, _button in enumerate(obs.available_buttons[:5]):
        mask[index] = 1
    if obs.allows_click:
        mask[5] = 1
    mask[6] = 1
    return tuple(mask)


def configure_reproducibility(seed: int) -> torch.Generator:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def update_empirical_click_votes(
    click_votes: dict[str, int],
    *,
    clicked_kind: str | None,
    frame_changed: bool,
    player_move: Coord,
    solved_delta: int,
) -> None:
    if clicked_kind == "switch" and (frame_changed or player_move != (0, 0) or solved_delta != 0):
        click_votes["switch"] = click_votes.get("switch", 0) + 1
    if clicked_kind == "portal" and (frame_changed or player_move != (0, 0) or solved_delta != 0):
        click_votes["teleport"] = click_votes.get("teleport", 0) + 1


def derive_empirical_click_hint(click_votes: dict[str, int], *, min_votes: int) -> str | None:
    if not click_votes:
        return None
    best_label, best_votes = max(click_votes.items(), key=lambda item: item[1])
    if best_votes < min_votes:
        return None
    return best_label


def best_action_for_state(case: EnvironmentCase, level_index: int, state) -> Action | None:
    from .hypotheses import MechanicHypothesis

    config = case.config_for_level(level_index)
    hypothesis = MechanicHypothesis(
        family=config.family,
        movement_map=config.movement_map,
        click_mode=config.click_mode,
        allows_undo=config.allows_undo,
    )
    plan = plan_with_hypothesis(state, hypothesis, max_depth=24)
    if plan is None or not plan.actions:
        return None
    return plan.actions[0]


def best_action_for_state_cached(
    case: EnvironmentCase,
    level_index: int,
    state,
    cache: dict[tuple[str, int, tuple[object, ...]], Action | None],
) -> Action | None:
    key = (case.env_id, level_index, state.signature())
    if key not in cache:
        cache[key] = best_action_for_state(case, level_index, state)
    return cache[key]


def case_labels(
    case: EnvironmentCase,
    level_index: int,
    state,
    parsed: ParsedFrame,
    obs,
    *,
    planner_cache: dict[tuple[str, int, tuple[object, ...]], Action | None],
) -> tuple[int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], int, tuple[int, ...]]:
    config = case.config_for_level(level_index)
    button_map = config.button_map()
    direction_indices = tuple(DIRECTION_TO_INDEX[button_map.get(button)] for button in config.available_buttons)
    mask = tuple(1 for _ in config.available_buttons)
    effect_indices = effect_indices_for_state(state, obs.available_buttons, config)
    effect_mask = tuple(1 for _ in obs.available_buttons)
    oracle_action = best_action_for_state_cached(case, level_index, state, planner_cache)
    if isinstance(oracle_action, str):
        action_target = action_class_index_for_button(oracle_action, obs.available_buttons) if oracle_action != "undo" else 6
    elif isinstance(oracle_action, ClickAction):
        action_target = 5
    else:
        action_target = 6
    return (
        FAMILY_TO_INDEX[config.family],
        CLICK_TO_INDEX[config.click_mode],
        direction_indices,
        mask,
        effect_indices,
        effect_mask,
        action_target,
        action_mask_for_obs(obs),
    )


def choose_exploration_action(
    obs,
    parsed: ParsedFrame,
    *,
    tried_slots: set[int],
    tried_clicks: set[Coord],
    rng: random.Random,
) -> Action:
    for slot_index, button in enumerate(obs.available_buttons):
        if slot_index not in tried_slots:
            tried_slots.add(slot_index)
            return button
    if obs.allows_click:
        for coord in parsed.clickable_targets():
            if coord not in tried_clicks:
                tried_clicks.add(coord)
                return ClickAction(*coord)
    if obs.allows_undo and rng.random() < 0.15:
        return "undo"
    actions: list[Action] = list(obs.available_buttons)
    if obs.allows_click:
        actions.extend(ClickAction(*coord) for coord in parsed.clickable_targets())
    if obs.allows_undo:
        actions.append("undo")
    return rng.choice(actions)


def generate_trace_samples(
    cases: Iterable[EnvironmentCase],
    *,
    traces_per_level: int,
    rollout_steps: int,
    seed: int,
    text_manual_probability: float = 0.0,
    teacher_rollin_probability: float = 0.0,
) -> list[SharedTrajectorySample]:
    rng = random.Random(seed)
    samples: list[SharedTrajectorySample] = []
    planner_cache: dict[tuple[str, int, tuple[object, ...]], Action | None] = {}
    for case in cases:
        for level_index in range(len(case.levels)):
            for trace_index in range(traces_per_level):
                env = HiddenMechanicEnvironment(case)
                obs = env.reset(level_index)
                parsed = parse_frame(obs.frame)
                static_scene = derive_static_scene(parsed)
                state = build_state(parsed, static_scene)
                history: list[str] = ["<bos>"]
                if rng.random() < text_manual_probability:
                    history.extend(build_manual_tokens(case, level_index, variant_seed=seed + trace_index + level_index * 31))
                (
                    family_index,
                    click_index,
                    direction_indices,
                    direction_mask,
                    effect_indices,
                    effect_mask,
                    action_target,
                    action_mask,
                ) = case_labels(case, level_index, state, parsed, obs, planner_cache=planner_cache)
                history.extend(
                    build_step_tokens(
                        parsed,
                        state,
                        available_buttons=obs.available_buttons,
                        allows_click=obs.allows_click,
                        allows_undo=obs.allows_undo,
                        transition=None,
                    )
                )
                samples.append(
                    SharedTrajectorySample(
                        tokens=tuple(history),
                        family_index=family_index,
                        click_index=click_index,
                        direction_indices=direction_indices,
                        direction_mask=direction_mask,
                        effect_indices=effect_indices,
                        effect_mask=effect_mask,
                        action_target=action_target,
                        action_mask=action_mask,
                        trace_steps=0,
                    )
                )

                tried_slots: set[int] = set()
                tried_clicks: set[Coord] = set()
                prior_state = state
                prior_parsed = parsed
                solved_before = obs.solved

                for step_index in range(rollout_steps):
                    step_rng = random.Random(rng.randint(0, 10**9) + trace_index * 17 + step_index)
                    teacher_action = best_action_for_state_cached(case, level_index, state, planner_cache)
                    if teacher_action is not None and step_rng.random() < teacher_rollin_probability:
                        action = teacher_action
                    else:
                        action = choose_exploration_action(
                            obs,
                            parsed,
                            tried_slots=tried_slots,
                            tried_clicks=tried_clicks,
                            rng=step_rng,
                        )
                    next_obs = env.step(action)
                    next_parsed = parse_frame(next_obs.frame)
                    next_state = build_state(next_parsed, static_scene, prior_state=state)
                    transition = summarize_transition(
                        action,
                        prior_state,
                        next_state,
                        prior_parsed,
                        next_parsed,
                        solved_before=solved_before,
                        solved_after=next_obs.solved,
                        available_buttons=obs.available_buttons,
                    )
                    history.extend(
                        build_step_tokens(
                            next_parsed,
                            next_state,
                            available_buttons=next_obs.available_buttons,
                            allows_click=next_obs.allows_click,
                            allows_undo=next_obs.allows_undo,
                            transition=transition,
                        )
                    )
                    (
                        family_index,
                        click_index,
                        direction_indices,
                        direction_mask,
                        effect_indices,
                        effect_mask,
                        action_target,
                        action_mask,
                    ) = case_labels(case, level_index, next_state, next_parsed, next_obs, planner_cache=planner_cache)
                    samples.append(
                        SharedTrajectorySample(
                            tokens=tuple(history),
                            family_index=family_index,
                            click_index=click_index,
                            direction_indices=direction_indices,
                            direction_mask=direction_mask,
                            effect_indices=effect_indices,
                            effect_mask=effect_mask,
                            action_target=action_target,
                            action_mask=action_mask,
                            trace_steps=step_index + 1,
                        )
                    )
                    obs = next_obs
                    parsed = next_parsed
                    prior_parsed = next_parsed
                    prior_state = next_state
                    state = next_state
                    solved_before = next_obs.solved
                    if next_obs.solved:
                        break
    return samples


def compute_prediction_metrics(
    model: MechanicInferenceModel,
    tokenizer: SharedEventTokenizer,
    samples: list[SharedTrajectorySample],
    *,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    dataset = SharedTrajectoryDataset(samples, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_shared_batch(rows, pad_id=tokenizer.pad_id),
    )
    family_hits = 0
    click_hits = 0
    direction_hits = 0.0
    direction_total = 0.0
    effect_hits = 0.0
    effect_total = 0.0
    action_hits = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            outputs = model(input_ids, lengths)
            family_pred = outputs["family_logits"].argmax(dim=-1).cpu()
            click_pred = outputs["click_logits"].argmax(dim=-1).cpu()
            direction_pred = outputs["direction_logits"].argmax(dim=-1).cpu()
            effect_pred = outputs["effect_logits"].argmax(dim=-1).cpu()
            action_pred = outputs["action_logits"].argmax(dim=-1).cpu()
            family_hits += int((family_pred == batch["family"]).sum().item())
            click_hits += int((click_pred == batch["click"]).sum().item())
            matches = (direction_pred == batch["directions"]).float() * batch["direction_mask"]
            direction_hits += float(matches.sum().item())
            direction_total += float(batch["direction_mask"].sum().item())
            effect_matches = (effect_pred == batch["effects"]).float() * batch["effect_mask"]
            effect_hits += float(effect_matches.sum().item())
            effect_total += float(batch["effect_mask"].sum().item())
            action_hits += int((action_pred == batch["action_target"]).sum().item())
            total += batch["family"].size(0)
    return {
        "family_accuracy": family_hits / max(total, 1),
        "click_accuracy": click_hits / max(total, 1),
        "direction_accuracy": direction_hits / max(direction_total, 1.0),
        "effect_accuracy": effect_hits / max(effect_total, 1.0),
        "action_accuracy": action_hits / max(total, 1),
    }


def train_mechanic_model(
    train_samples: list[SharedTrajectorySample],
    val_samples: list[SharedTrajectorySample],
    *,
    config: TrainConfig,
) -> tuple[MechanicInferenceModel, SharedEventTokenizer, dict[str, object]]:
    data_generator = configure_reproducibility(config.seed)
    tokenizer = SharedEventTokenizer()
    tokenizer.fit(train_samples)
    model = MechanicInferenceModel(
        ModelConfig(
            vocab_size=len(tokenizer.id_to_token),
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            pooling=config.pooling,
        )
    ).to(config.device)

    train_dataset = SharedTrajectoryDataset(train_samples, tokenizer)
    val_dataset = SharedTrajectoryDataset(val_samples, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=data_generator,
        collate_fn=lambda rows: collate_shared_batch(rows, pad_id=tokenizer.pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_shared_batch(rows, pad_id=tokenizer.pad_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    family_loss_fn = nn.CrossEntropyLoss()
    click_loss_fn = nn.CrossEntropyLoss()
    direction_loss_fn = nn.CrossEntropyLoss(reduction="none")
    effect_loss_fn = nn.CrossEntropyLoss(reduction="none")
    action_loss_fn = nn.CrossEntropyLoss(reduction="none")

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(config.device)
            lengths = batch["lengths"].to(config.device)
            family = batch["family"].to(config.device)
            click = batch["click"].to(config.device)
            directions = batch["directions"].to(config.device)
            direction_mask = batch["direction_mask"].to(config.device)
            effects = batch["effects"].to(config.device)
            effect_mask = batch["effect_mask"].to(config.device)
            action_target = batch["action_target"].to(config.device)
            action_mask = batch["action_mask"].to(config.device)

            outputs = model(input_ids, lengths)
            loss = family_loss_fn(outputs["family_logits"], family)
            loss = loss + 0.5 * click_loss_fn(outputs["click_logits"], click)

            direction_logits = outputs["direction_logits"].reshape(-1, len(DIRECTION_TO_INDEX))
            direction_targets = directions.reshape(-1)
            per_direction = direction_loss_fn(direction_logits, direction_targets).view(directions.shape)
            direction_loss = (per_direction * direction_mask).sum() / direction_mask.sum().clamp_min(1.0)
            effect_logits = outputs["effect_logits"].reshape(-1, len(DIRECTION_TO_INDEX))
            effect_targets = effects.reshape(-1)
            per_effect = effect_loss_fn(effect_logits, effect_targets).view(effects.shape)
            effect_loss = (per_effect * effect_mask).sum() / effect_mask.sum().clamp_min(1.0)
            masked_action_logits = outputs["action_logits"].masked_fill(action_mask <= 0, -1e9)
            action_loss = action_loss_fn(masked_action_logits, action_target).mean()
            loss = loss + config.direction_loss_weight * direction_loss
            loss = loss + config.effect_loss_weight * effect_loss
            loss = loss + config.action_loss_weight * action_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += float(loss.item())
            train_batches += 1

        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(config.device)
                lengths = batch["lengths"].to(config.device)
                family = batch["family"].to(config.device)
                click = batch["click"].to(config.device)
                directions = batch["directions"].to(config.device)
                direction_mask = batch["direction_mask"].to(config.device)
                effects = batch["effects"].to(config.device)
                effect_mask = batch["effect_mask"].to(config.device)
                action_target = batch["action_target"].to(config.device)
                action_mask = batch["action_mask"].to(config.device)
                outputs = model(input_ids, lengths)
                loss = family_loss_fn(outputs["family_logits"], family)
                loss = loss + 0.5 * click_loss_fn(outputs["click_logits"], click)
                direction_logits = outputs["direction_logits"].reshape(-1, len(DIRECTION_TO_INDEX))
                direction_targets = directions.reshape(-1)
                per_direction = direction_loss_fn(direction_logits, direction_targets).view(directions.shape)
                direction_loss = (per_direction * direction_mask).sum() / direction_mask.sum().clamp_min(1.0)
                effect_logits = outputs["effect_logits"].reshape(-1, len(DIRECTION_TO_INDEX))
                effect_targets = effects.reshape(-1)
                per_effect = effect_loss_fn(effect_logits, effect_targets).view(effects.shape)
                effect_loss = (per_effect * effect_mask).sum() / effect_mask.sum().clamp_min(1.0)
                masked_action_logits = outputs["action_logits"].masked_fill(action_mask <= 0, -1e9)
                action_loss = action_loss_fn(masked_action_logits, action_target).mean()
                loss = loss + config.direction_loss_weight * direction_loss
                loss = loss + config.effect_loss_weight * effect_loss
                loss = loss + config.action_loss_weight * action_loss
                val_loss_total += float(loss.item())
                val_batches += 1

        mean_train_loss = train_loss_total / max(train_batches, 1)
        mean_val_loss = val_loss_total / max(val_batches, 1)
        history.append({"epoch": epoch + 1, "train_loss": mean_train_loss, "val_loss": mean_val_loss})
        if mean_val_loss < best_val:
            best_val = mean_val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "history": history,
        "train_metrics": compute_prediction_metrics(
            model,
            tokenizer,
            train_samples,
            batch_size=config.batch_size,
            device=config.device,
        ),
        "val_metrics": compute_prediction_metrics(
            model,
            tokenizer,
            val_samples,
            batch_size=config.batch_size,
            device=config.device,
        ),
    }
    return model, tokenizer, metrics


class LearnedMechanicAgent(BaseAgent):
    name = "learned_mechanic"

    def __init__(
        self,
        model: MechanicInferenceModel,
        tokenizer: SharedEventTokenizer,
        *,
        device: str,
        explore_budget: int = 5,
        confidence_threshold: float = 0.58,
        use_effect_for_planning: bool = False,
        use_empirical_controls: bool = False,
        use_empirical_click_hint: bool = False,
        use_empirical_click_votes: bool = False,
        empirical_click_min_votes: int = 1,
        prefer_click_probe_when_unknown: bool = False,
        use_empirical_greedy_actions: bool = False,
        empirical_greedy_min_known_controls: int = 1,
        use_action_planner_agreement: bool = False,
        action_margin_threshold: float = 0.0,
        use_symbolic_posterior: bool = False,
        symbolic_transfer: bool = False,
        symbolic_family_prior: bool = True,
        symbolic_click_prior: bool = True,
        symbolic_direction_prior: bool = True,
        symbolic_plan_confidence: float = 0.55,
        symbolic_prior_floor: float = 0.05,
        symbolic_prior_power: float = 1.0,
        symbolic_family_weight: float = 1.0,
        symbolic_click_weight: float = 1.0,
        symbolic_direction_weight: float = 1.0,
        symbolic_reprioritize: bool = False,
        symbolic_reprioritize_uncertainty: float = 0.75,
        symbolic_plan_uncertainty_ceiling: float | None = None,
        symbolic_plan_commit_steps: int = 1,
        symbolic_plan_commit_confidence: float | None = None,
        symbolic_plan_commit_uncertainty_ceiling: float | None = None,
        symbolic_transfer_confidence_floor: float | None = None,
        symbolic_transfer_uncertainty_ceiling: float | None = None,
        symbolic_transfer_requires_solved: bool = False,
        symbolic_transfer_mode: str = "full",
        symbolic_summary_transfer: bool = False,
        symbolic_summary_transfer_mode: str = "family_click",
        symbolic_affordance_transfer: bool = False,
        symbolic_affordance_prior_mode: str = "family_click",
        symbolic_affordance_bonus_weight: float = 0.35,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.explore_budget = explore_budget
        self.confidence_threshold = confidence_threshold
        self.use_effect_for_planning = use_effect_for_planning
        self.use_empirical_controls = use_empirical_controls
        self.use_empirical_click_hint = use_empirical_click_hint
        self.use_empirical_click_votes = use_empirical_click_votes
        self.empirical_click_min_votes = empirical_click_min_votes
        self.prefer_click_probe_when_unknown = prefer_click_probe_when_unknown
        self.use_empirical_greedy_actions = use_empirical_greedy_actions
        self.empirical_greedy_min_known_controls = empirical_greedy_min_known_controls
        self.use_action_planner_agreement = use_action_planner_agreement
        self.action_margin_threshold = action_margin_threshold
        self.use_symbolic_posterior = use_symbolic_posterior
        self.symbolic_transfer = symbolic_transfer
        self.symbolic_family_prior = symbolic_family_prior
        self.symbolic_click_prior = symbolic_click_prior
        self.symbolic_direction_prior = symbolic_direction_prior
        self.symbolic_plan_confidence = symbolic_plan_confidence
        self.symbolic_prior_floor = symbolic_prior_floor
        self.symbolic_prior_power = symbolic_prior_power
        self.symbolic_family_weight = max(0.0, float(symbolic_family_weight))
        self.symbolic_click_weight = max(0.0, float(symbolic_click_weight))
        self.symbolic_direction_weight = max(0.0, float(symbolic_direction_weight))
        self.symbolic_reprioritize = symbolic_reprioritize
        self.symbolic_reprioritize_uncertainty = symbolic_reprioritize_uncertainty
        self.symbolic_plan_uncertainty_ceiling = symbolic_plan_uncertainty_ceiling
        self.symbolic_plan_commit_steps = max(1, int(symbolic_plan_commit_steps))
        self.symbolic_plan_commit_confidence = symbolic_plan_commit_confidence
        self.symbolic_plan_commit_uncertainty_ceiling = symbolic_plan_commit_uncertainty_ceiling
        self.symbolic_transfer_confidence_floor = symbolic_transfer_confidence_floor
        self.symbolic_transfer_uncertainty_ceiling = symbolic_transfer_uncertainty_ceiling
        self.symbolic_transfer_requires_solved = symbolic_transfer_requires_solved
        self.symbolic_transfer_mode = symbolic_transfer_mode
        self.symbolic_summary_transfer = symbolic_summary_transfer
        self.symbolic_summary_transfer_mode = symbolic_summary_transfer_mode
        self.symbolic_affordance_transfer = symbolic_affordance_transfer
        self.symbolic_affordance_prior_mode = symbolic_affordance_prior_mode
        self.symbolic_affordance_bonus_weight = max(0.0, float(symbolic_affordance_bonus_weight))
        self._symbolic_memory: dict[str, dict] = {}
        self._symbolic_summary_memory: dict[str, PriorPrediction] = {}
        self._symbolic_affordance_memory: dict[str, AffordanceTransferSummary] = {}

    def solve_case(self, case: EnvironmentCase, *, step_limit: int = 64) -> tuple[LevelOutcome, ...]:
        if self.use_symbolic_posterior:
            return self._solve_case_with_symbolic_posterior(case, step_limit=step_limit)
        outcomes: list[LevelOutcome] = []
        env = HiddenMechanicEnvironment(case)
        for level_index in range(len(case.levels)):
            obs = env.reset(level_index)
            parsed = parse_frame(obs.frame)
            static_scene = derive_static_scene(parsed)
            state = build_state(parsed, static_scene)
            history: list[str] = ["<bos>"]
            history.extend(
                build_step_tokens(
                    parsed,
                    state,
                    available_buttons=obs.available_buttons,
                    allows_click=obs.allows_click,
                    allows_undo=obs.allows_undo,
                    transition=None,
                )
            )
            tried_slots: set[int] = set()
            tried_clicks: set[Coord] = set()
            prior_state = state
            prior_parsed = parsed
            solved_before = obs.solved
            latest_prediction = self.predict_tokens(history)
            observed_controls: dict[str, Coord | None] = {}
            observed_click_hint: str | None = None
            observed_click_votes: dict[str, int] = {}

            while not obs.solved and obs.action_count < step_limit:
                action: Action | None = None
                agreement_candidate: Action | None = None
                agreement_plan: PlannerResult | None = None
                if self.use_action_planner_agreement and (
                    obs.action_count >= self.explore_budget or latest_prediction.confidence >= self.confidence_threshold
                ):
                    agreement_candidate = self._candidate_action_from_prediction(
                        obs,
                        parsed,
                        latest_prediction,
                        observed_click_hint=observed_click_hint,
                    )
                    agreement_plan = self._plan_from_prediction(
                        state,
                        obs.available_buttons,
                        obs.allows_undo,
                        latest_prediction,
                        observed_controls=observed_controls,
                        observed_click_hint=observed_click_hint,
                    )
                    if (
                        agreement_candidate is not None
                        and agreement_plan is not None
                        and agreement_plan.actions
                        and agreement_candidate == agreement_plan.actions[0]
                    ):
                        action = agreement_candidate
                if (
                    self.use_empirical_greedy_actions
                    and len(observed_controls) >= self.empirical_greedy_min_known_controls
                    and (obs.action_count >= self.explore_budget or latest_prediction.confidence >= self.confidence_threshold)
                ):
                    action = self._empirical_greedy_action(
                        state,
                        obs.available_buttons,
                        obs.allows_undo,
                        latest_prediction,
                        observed_controls=observed_controls,
                        observed_click_hint=observed_click_hint,
                    )
                if obs.action_count >= self.explore_budget or latest_prediction.confidence >= self.confidence_threshold:
                    if action is None:
                        action = self._direct_action_from_prediction(
                            obs,
                            parsed,
                            latest_prediction,
                            observed_click_hint=observed_click_hint,
                        )
                if action is None and (obs.action_count >= self.explore_budget or latest_prediction.confidence >= self.confidence_threshold):
                    plan = agreement_plan or self._plan_from_prediction(
                        state,
                        obs.available_buttons,
                        obs.allows_undo,
                        latest_prediction,
                        observed_controls=observed_controls,
                        observed_click_hint=observed_click_hint,
                    )
                    if plan is not None and plan.actions:
                        action = plan.actions[0]
                if action is None:
                    if self.prefer_click_probe_when_unknown and obs.allows_click and observed_click_hint is None:
                        click_targets = tuple(parsed.clickable_targets())
                        if click_targets:
                            first_target = click_targets[0]
                            if first_target not in tried_clicks:
                                tried_clicks.add(first_target)
                                action = ClickAction(*first_target)
                    if action is None:
                        action = choose_exploration_action(
                            obs,
                            parsed,
                            tried_slots=tried_slots,
                            tried_clicks=tried_clicks,
                            rng=random.Random(level_index * 101 + obs.action_count),
                        )
                next_obs = env.step(action)
                next_parsed = parse_frame(next_obs.frame)
                next_state = build_state(next_parsed, static_scene, prior_state=state)
                transition = summarize_transition(
                    action,
                    prior_state,
                    next_state,
                    prior_parsed,
                    next_parsed,
                    solved_before=solved_before,
                    solved_after=next_obs.solved,
                    available_buttons=obs.available_buttons,
                )
                history.extend(
                    build_step_tokens(
                        next_parsed,
                        next_state,
                        available_buttons=next_obs.available_buttons,
                        allows_click=next_obs.allows_click,
                        allows_undo=next_obs.allows_undo,
                        transition=transition,
                    )
                )
                if self.use_empirical_controls and transition is not None and isinstance(action, str) and action in obs.available_buttons:
                    observed_controls[action] = transition.player_move if transition.player_move in DIRECTION_TO_INDEX else None
                if (
                    self.use_empirical_click_hint
                    and transition is not None
                    and isinstance(action, ClickAction)
                    and transition.clicked_kind in {"switch", "portal"}
                    and (transition.frame_changed or transition.player_move != (0, 0) or transition.solved_delta != 0)
                ):
                    if self.use_empirical_click_votes:
                        update_empirical_click_votes(
                            observed_click_votes,
                            clicked_kind=transition.clicked_kind,
                            frame_changed=transition.frame_changed,
                            player_move=transition.player_move,
                            solved_delta=transition.solved_delta,
                        )
                        observed_click_hint = derive_empirical_click_hint(
                            observed_click_votes,
                            min_votes=self.empirical_click_min_votes,
                        )
                    else:
                        observed_click_hint = "switch" if transition.clicked_kind == "switch" else "teleport"
                latest_prediction = self.predict_tokens(history)
                obs = next_obs
                parsed = next_parsed
                prior_parsed = next_parsed
                prior_state = next_state
                state = next_state
                solved_before = next_obs.solved

            outcomes.append(
                LevelOutcome(
                    solved=obs.solved,
                    action_count=obs.action_count,
                    top_confidence=latest_prediction.confidence,
                    hypothesis_count=1,
                )
            )
        return tuple(outcomes)

    def _solve_case_with_symbolic_posterior(self, case: EnvironmentCase, *, step_limit: int = 64) -> tuple[LevelOutcome, ...]:
        env = HiddenMechanicEnvironment(case)
        outcomes: list[LevelOutcome] = []
        prior = self._symbolic_memory.get(case.env_id)
        summary_prior = self._symbolic_summary_memory.get(case.env_id)
        affordance_summary = self._symbolic_affordance_memory.get(case.env_id)
        proposal_prior = LearnedProposalPrior(
            self,
            use_family_prior=self.symbolic_family_prior,
            use_click_prior=self.symbolic_click_prior,
            use_direction_prior=self.symbolic_direction_prior,
            floor=self.symbolic_prior_floor,
            power=self.symbolic_prior_power,
            family_weight=self.symbolic_family_weight,
            click_weight=self.symbolic_click_weight,
            direction_weight=self.symbolic_direction_weight,
        )

        for level_index in range(len(case.levels)):
            obs = env.reset(level_index)
            parsed = parse_frame(obs.frame)
            static_scene = derive_static_scene(parsed)
            state = build_state(parsed, static_scene)
            posterior = MechanicPosterior(
                obs.available_buttons,
                state,
                allows_click=obs.allows_click,
                allows_undo=obs.allows_undo,
                prior=prior if self.symbolic_transfer else None,
                proposal_prior=proposal_prior,
            )
            if summary_prior is not None:
                posterior.apply_prediction(summary_prior)
            if affordance_summary is not None:
                posterior.apply_prediction(
                    affordance_summary.to_prior_prediction(mode=self.symbolic_affordance_prior_mode)
                )
            frontier = FrontierGraphPolicy()
            planned_actions: tuple[Action, ...] = ()
            transitions_seen: list[TransitionSummary] = []

            while not obs.solved and obs.action_count < step_limit:
                parsed = parse_frame(obs.frame)
                state = build_state(parsed, static_scene, prior_state=state)
                actions = self._symbolic_candidate_actions(obs, parsed, state)
                top_hypothesis, confidence = posterior.top
                plan_gate_open = (
                    self.symbolic_plan_uncertainty_ceiling is None
                    or posterior.uncertainty <= self.symbolic_plan_uncertainty_ceiling
                )
                if confidence >= self.symbolic_plan_confidence and plan_gate_open and not planned_actions:
                    plan = plan_with_hypothesis(state, top_hypothesis, max_depth=step_limit - obs.action_count)
                    if plan is not None and plan.actions:
                        planned_actions = plan.actions[: self.symbolic_plan_commit_steps]

                if planned_actions:
                    action = planned_actions[0]
                    planned_actions = planned_actions[1:]
                else:
                    action = self._choose_symbolic_experiment(
                        state,
                        parsed,
                        actions,
                        posterior,
                        frontier,
                        affordance_summary=affordance_summary,
                        remaining_level_weight=float(len(case.levels) - level_index),
                    )

                next_obs = env.step(action)
                next_parsed = parse_frame(next_obs.frame)
                next_state = build_state(next_parsed, static_scene, prior_state=state)
                transition = summarize_transition(
                    action,
                    state,
                    next_state,
                    parsed,
                    next_parsed,
                    solved_before=obs.solved,
                    solved_after=next_obs.solved,
                    available_buttons=obs.available_buttons,
                )
                if transition is not None:
                    transitions_seen.append(transition)
                posterior.update(state, action, next_state)
                if (
                    self.symbolic_reprioritize
                    and posterior.uncertainty >= self.symbolic_reprioritize_uncertainty
                    and tuple(next_obs.available_buttons) == posterior.available_buttons
                ):
                    posterior.apply_prediction(
                        proposal_prior.predict(
                            next_state,
                            available_buttons=next_obs.available_buttons,
                            allows_click=next_obs.allows_click,
                            allows_undo=next_obs.allows_undo,
                        )
                    )
                frontier.observe(state, action, next_state)
                state = next_state
                obs = next_obs
                planned_actions = self._retain_symbolic_plan(planned_actions, posterior, next_obs)

            top_hypothesis, confidence = posterior.top
            if self._should_store_symbolic_prior(posterior, confidence=confidence, solved=obs.solved):
                if self.symbolic_transfer:
                    prior = posterior.transfer_prior(state, mode=self.symbolic_transfer_mode)
                if self.symbolic_summary_transfer:
                    summary_prior = posterior.transfer_summary(state, mode=self.symbolic_summary_transfer_mode)
                if self.symbolic_affordance_transfer:
                    affordance_summary = summarize_affordances_from_transitions(transitions_seen)
            else:
                prior = None
                summary_prior = None
                affordance_summary = None
            outcomes.append(
                LevelOutcome(
                    solved=obs.solved,
                    action_count=obs.action_count,
                    top_confidence=confidence,
                    hypothesis_count=posterior.hypothesis_count,
                    explored_states=len(frontier.transitions),
                )
            )

        if self.symbolic_transfer:
            self._symbolic_memory[case.env_id] = prior
        if self.symbolic_summary_transfer:
            self._symbolic_summary_memory[case.env_id] = summary_prior
        if self.symbolic_affordance_transfer:
            self._symbolic_affordance_memory[case.env_id] = affordance_summary
        return tuple(outcomes)

    def _retain_symbolic_plan(self, planned_actions: tuple[Action, ...], posterior: MechanicPosterior, next_obs) -> tuple[Action, ...]:
        if not planned_actions:
            return ()
        if tuple(next_obs.available_buttons) != posterior.available_buttons:
            return ()
        _top_hypothesis, top_confidence = posterior.top
        if self.symbolic_plan_commit_confidence is not None and top_confidence < self.symbolic_plan_commit_confidence:
            return ()
        if (
            self.symbolic_plan_commit_uncertainty_ceiling is not None
            and posterior.uncertainty > self.symbolic_plan_commit_uncertainty_ceiling
        ):
            return ()
        return planned_actions

    def _should_store_symbolic_prior(self, posterior: MechanicPosterior, *, confidence: float, solved: bool) -> bool:
        if not (self.symbolic_transfer or self.symbolic_summary_transfer or self.symbolic_affordance_transfer):
            return False
        if self.symbolic_transfer_requires_solved and not solved:
            return False
        if (
            self.symbolic_transfer_confidence_floor is not None
            and confidence < self.symbolic_transfer_confidence_floor
        ):
            return False
        if (
            self.symbolic_transfer_uncertainty_ceiling is not None
            and posterior.uncertainty > self.symbolic_transfer_uncertainty_ceiling
        ):
            return False
        return True

    def _symbolic_candidate_actions(self, obs, parsed: ParsedFrame, state) -> tuple[Action, ...]:
        actions: list[Action] = list(obs.available_buttons)
        if obs.allows_click:
            click_targets = sorted(state.switches or set(state.portals) or state.goals)
            actions.extend(ClickAction(row, col) for row, col in click_targets)
        if obs.allows_undo:
            actions.append("undo")
        if not actions:
            actions.append("undo")
        return tuple(dict.fromkeys(actions))

    def _choose_symbolic_experiment(
        self,
        state,
        parsed: ParsedFrame,
        actions: tuple[Action, ...],
        posterior: MechanicPosterior,
        frontier: FrontierGraphPolicy,
        *,
        affordance_summary: AffordanceTransferSummary | None,
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
            if affordance_summary is not None:
                score += self._affordance_action_bonus(
                    action,
                    parsed,
                    state,
                    affordance_summary,
                )
            if score > best_score:
                best_score = score
                best_action = action
        if best_score < 0.1:
            return frontier.choose(state, actions)
        return best_action

    def _affordance_action_bonus(
        self,
        action: Action,
        parsed: ParsedFrame,
        state,
        affordance_summary: AffordanceTransferSummary,
    ) -> float:
        if self.symbolic_affordance_bonus_weight <= 0.0:
            return 0.0
        weight = self.symbolic_affordance_bonus_weight
        bonus = 0.0
        if action == "undo":
            bonus += 0.35 * affordance_summary.action_kind_scores.get("undo", 0.0)
        elif isinstance(action, ClickAction):
            bonus += 0.25 * affordance_summary.action_kind_scores.get("click", 0.0)
            clicked_kind = next((obj.kind for obj in parsed.objects if obj.anchor == action.coord), None)
            if clicked_kind is not None:
                bonus += 0.45 * affordance_summary.clicked_kind_scores.get(clicked_kind, 0.0)
                if clicked_kind == "switch":
                    bonus += 0.3 * affordance_summary.state_change_scores.get("switch_change", 0.0)
                elif clicked_kind == "portal":
                    bonus += 0.3 * affordance_summary.state_change_scores.get("portal_use", 0.0)
        else:
            bonus += 0.2 * affordance_summary.action_kind_scores.get("move", 0.0)
            if state.keys:
                bonus += 0.15 * affordance_summary.state_change_scores.get("key_progress", 0.0)
            if state.switches:
                bonus += 0.1 * affordance_summary.state_change_scores.get("switch_change", 0.0)
            if state.boxes:
                bonus += 0.15 * affordance_summary.state_change_scores.get("box_motion", 0.0)
        return weight * bonus

    def predict_tokens(self, tokens: Iterable[str]) -> PredictionBundle:
        input_ids = torch.tensor([self.tokenizer.encode(tokens)], dtype=torch.long, device=self.device)
        lengths = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, lengths)
            family_probs = torch.softmax(outputs["family_logits"][0], dim=-1).cpu().tolist()
            click_probs = torch.softmax(outputs["click_logits"][0], dim=-1).cpu().tolist()
            direction_probs = [
                torch.softmax(slot_logits, dim=-1).cpu().tolist()
                for slot_logits in outputs["direction_logits"][0]
            ]
            effect_probs = [
                torch.softmax(slot_logits, dim=-1).cpu().tolist()
                for slot_logits in outputs["effect_logits"][0]
            ]
            action_probs = torch.softmax(outputs["action_logits"][0], dim=-1).cpu().tolist()
        return PredictionBundle(
            family_probs=tuple(float(value) for value in family_probs),
            click_probs=tuple(float(value) for value in click_probs),
            direction_probs=tuple(tuple(float(value) for value in slot) for slot in direction_probs),
            effect_probs=tuple(tuple(float(value) for value in slot) for slot in effect_probs),
            action_probs=tuple(float(value) for value in action_probs),
        )

    def _masked_action_stats(self, obs, prediction: PredictionBundle) -> tuple[list[float], int, float, float]:
        masked = list(prediction.action_probs)
        for index, allowed in enumerate(action_mask_for_obs(obs)):
            if not allowed:
                masked[index] = -1.0
        best_index = max(range(len(masked)), key=lambda idx: masked[idx])
        ordered = sorted(masked, reverse=True)
        best_confidence = max(0.0, masked[best_index])
        second_confidence = max(0.0, ordered[1]) if len(ordered) > 1 else 0.0
        margin = best_confidence - second_confidence
        return masked, best_index, best_confidence, margin

    def _candidate_action_from_prediction(
        self,
        obs,
        parsed: ParsedFrame,
        prediction: PredictionBundle,
        *,
        observed_click_hint: str | None = None,
    ) -> Action | None:
        _masked, best_index, _best_confidence, _margin = self._masked_action_stats(obs, prediction)
        if best_index < len(obs.available_buttons):
            return obs.available_buttons[best_index]
        if best_index == 5 and obs.allows_click:
            if observed_click_hint == "switch":
                preferred = "switch"
            elif observed_click_hint == "teleport":
                preferred = "portal"
            else:
                preferred = "switch" if prediction.click_probs[1] >= prediction.click_probs[2] else "portal"
            targets = [obj.anchor for obj in parsed.by_kind(preferred)]
            if not targets:
                targets = list(parsed.clickable_targets())
            if targets:
                return ClickAction(*targets[0])
        if best_index == 6 and obs.allows_undo:
            return "undo"
        return None

    def _direct_action_from_prediction(
        self,
        obs,
        parsed: ParsedFrame,
        prediction: PredictionBundle,
        *,
        observed_click_hint: str | None = None,
    ) -> Action | None:
        _masked, _best_index, best_confidence, margin = self._masked_action_stats(obs, prediction)
        if best_confidence < self.confidence_threshold:
            return None
        if margin < self.action_margin_threshold:
            return None
        return self._candidate_action_from_prediction(
            obs,
            parsed,
            prediction,
            observed_click_hint=observed_click_hint,
        )

    def _hypothesis_from_prediction(
        self,
        available_buttons: tuple[str, ...],
        allows_undo: bool,
        prediction: PredictionBundle,
        *,
        observed_controls: dict[str, Coord | None] | None = None,
        observed_click_hint: str | None = None,
    ):
        from .hypotheses import MechanicHypothesis

        family_index = int(max(range(len(prediction.family_probs)), key=lambda idx: prediction.family_probs[idx]))
        if observed_click_hint == "switch":
            click_index = CLICK_TO_INDEX["switch"]
        elif observed_click_hint == "teleport":
            click_index = CLICK_TO_INDEX["teleport"]
        else:
            click_index = int(max(range(len(prediction.click_probs)), key=lambda idx: prediction.click_probs[idx]))
        movement_map = []
        for slot_index, button in enumerate(available_buttons):
            if observed_controls and button in observed_controls:
                movement_map.append((button, observed_controls[button]))
                continue
            slot_probs = prediction.effect_probs[slot_index] if self.use_effect_for_planning else prediction.direction_probs[slot_index]
            direction_index = int(max(range(len(slot_probs)), key=lambda idx: slot_probs[idx]))
            movement_map.append((button, INDEX_TO_DIRECTION[direction_index]))
        return MechanicHypothesis(
            family=INDEX_TO_FAMILY[family_index],
            movement_map=tuple(movement_map),
            click_mode=INDEX_TO_CLICK[click_index],
            allows_undo=allows_undo,
        )

    def _empirical_greedy_action(
        self,
        state,
        available_buttons: tuple[str, ...],
        allows_undo: bool,
        prediction: PredictionBundle,
        *,
        observed_controls: dict[str, Coord | None] | None = None,
        observed_click_hint: str | None = None,
    ) -> Action | None:
        from .hypotheses import progress_score
        from .planner import candidate_actions
        from .dsl import simulate_action

        hypothesis = self._hypothesis_from_prediction(
            available_buttons,
            allows_undo,
            prediction,
            observed_controls=observed_controls,
            observed_click_hint=observed_click_hint,
        )
        current_score = progress_score(state, hypothesis.family)
        best_action: Action | None = None
        best_gain = 0.0
        for action in candidate_actions(state, hypothesis):
            next_state = simulate_action(
                state,
                action,
                hypothesis.config(tuple(button for button, _ in hypothesis.movement_map)),
            )
            gain = progress_score(next_state, hypothesis.family) - current_score
            if gain > best_gain + 1e-6:
                best_gain = gain
                best_action = action
        return best_action

    def _plan_from_prediction(
        self,
        state,
        available_buttons: tuple[str, ...],
        allows_undo: bool,
        prediction: PredictionBundle,
        *,
        observed_controls: dict[str, Coord | None] | None = None,
        observed_click_hint: str | None = None,
    ) -> PlannerResult | None:
        hypothesis = self._hypothesis_from_prediction(
            available_buttons,
            allows_undo,
            prediction,
            observed_controls=observed_controls,
            observed_click_hint=observed_click_hint,
        )
        return plan_with_hypothesis(state, hypothesis, max_depth=24)


def save_checkpoint(
    path: Path,
    model: MechanicInferenceModel,
    tokenizer: SharedEventTokenizer,
    train_config: TrainConfig,
    metrics: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "train_config": asdict(train_config),
        "metrics": metrics,
        "model_config": asdict(model.config),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    device: str | None = None,
) -> tuple[MechanicInferenceModel, SharedEventTokenizer, dict[str, object]]:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=resolved_device)
    tokenizer = SharedEventTokenizer.from_dict(payload["tokenizer"])
    model = MechanicInferenceModel(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.to(resolved_device)
    return model, tokenizer, payload
