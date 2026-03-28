from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

SUBJECT_POOL = tuple(range(3, 15))
VERB_POOL = tuple(range(15, 27))
OBJECT_POOL = tuple(range(27, 39))
MARKER_POOL = tuple(range(39, 47))
VOCAB_SIZE = 47

ORDER_PATTERNS = (
    ("s", "v", "o"),
    ("s", "o", "v"),
    ("v", "s", "o"),
    ("v", "o", "s"),
    ("o", "s", "v"),
    ("o", "v", "s"),
)

MARKER_RULES = ("subject_parity", "object_parity", "verb_bucket")
MARKER_POSITIONS = ("prefix", "after_first", "before_last", "suffix")


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    seed: int = 13
    train_tasks: int = 512
    val_tasks: int = 128
    test_tasks: int = 128
    max_support: int = 8
    query_examples: int = 12
    train_support_min: int = 1
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gru_embedding_dim: int = 48
    gru_hidden_dim: int = 80
    gru_memory_dim: int | None = None
    gpt2_d_model: int = 48
    gpt2_heads: int = 4
    gpt2_layers: int = 2
    gpt2_ff_dim: int = 160
    dropout: float = 0.1
    support_shots: tuple[int, ...] = (0, 1, 2, 4, 8)
    eval_batch_size: int = 32
    use_amp: bool = torch.cuda.is_available()
    sequence_focus_weight: float = 0.0
    sequence_focus_temperature: float = 0.5
    rollout_loss_weight: float = 0.0
    rollout_teacher_forcing_prob: float = 0.0
    rollout_weight_ramp_epochs: int = 0
    rollout_teacher_forcing_decay_epochs: int = 0


@dataclass(frozen=True, slots=True)
class LanguageTask:
    family: str
    support_sentences: tuple[tuple[int, ...], ...]
    query_sentences: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class ModelReport:
    name: str
    parameter_count: int
    best_val_loss: float
    adaptation_auc: float
    autoregressive_adaptation_auc: float
    eval_by_shot: dict[str, dict[str, float]]
    history: list[dict[str, float]]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_semantics(
    rng: random.Random,
    *,
    count: int,
    needs_polarity: bool,
) -> list[tuple[int, ...]]:
    semantics: set[tuple[int, ...]] = set()
    while len(semantics) < count:
        subject = rng.randrange(6)
        verb = rng.randrange(6)
        obj = rng.randrange(6)
        if needs_polarity:
            semantics.add((subject, verb, obj, rng.randrange(2)))
        else:
            semantics.add((subject, verb, obj))
    return list(semantics)


def _build_order_task(rng: random.Random, max_support: int, query_examples: int) -> LanguageTask:
    subject_tokens = rng.sample(SUBJECT_POOL, 6)
    verb_tokens = rng.sample(VERB_POOL, 6)
    object_tokens = rng.sample(OBJECT_POOL, 6)
    order = rng.choice(ORDER_PATTERNS)
    semantics = _sample_semantics(rng, count=max_support + query_examples, needs_polarity=False)

    def render(example: tuple[int, ...]) -> tuple[int, ...]:
        subject, verb, obj = example
        mapping = {
            "s": subject_tokens[subject],
            "v": verb_tokens[verb],
            "o": object_tokens[obj],
        }
        sentence = [BOS_ID]
        sentence.extend(mapping[slot] for slot in order)
        sentence.append(EOS_ID)
        return tuple(sentence)

    return LanguageTask(
        family="order_map",
        support_sentences=tuple(render(example) for example in semantics[:max_support]),
        query_sentences=tuple(render(example) for example in semantics[max_support:]),
    )


def _marker_value(rule: str, example: tuple[int, ...]) -> int:
    subject, verb, obj, polarity = example
    if rule == "subject_parity":
        return (subject + polarity) % 2
    if rule == "object_parity":
        return (obj + polarity) % 2
    return ((verb // 2) + polarity) % 2


def _insert_marker(tokens: list[int], marker_token: int, position: str) -> list[int]:
    if position == "prefix":
        return [marker_token, *tokens]
    if position == "after_first":
        return [tokens[0], marker_token, *tokens[1:]]
    if position == "before_last":
        return [*tokens[:-1], marker_token, tokens[-1]]
    return [*tokens, marker_token]


def _build_agreement_task(rng: random.Random, max_support: int, query_examples: int) -> LanguageTask:
    subject_tokens = rng.sample(SUBJECT_POOL, 6)
    verb_tokens = rng.sample(VERB_POOL, 6)
    object_tokens = rng.sample(OBJECT_POOL, 6)
    marker_pair = rng.sample(MARKER_POOL, 2)
    order = rng.choice(ORDER_PATTERNS[:3])
    marker_position = rng.choice(MARKER_POSITIONS)
    marker_rule = rng.choice(MARKER_RULES)
    semantics = _sample_semantics(rng, count=max_support + query_examples, needs_polarity=True)

    def render(example: tuple[int, ...]) -> tuple[int, ...]:
        subject, verb, obj, _polarity = example
        base = {
            "s": subject_tokens[subject],
            "v": verb_tokens[verb],
            "o": object_tokens[obj],
        }
        marker_token = marker_pair[_marker_value(marker_rule, example)]
        sentence = _insert_marker([base[slot] for slot in order], marker_token, marker_position)
        return tuple([BOS_ID, *sentence, EOS_ID])

    return LanguageTask(
        family="agreement_marker",
        support_sentences=tuple(render(example) for example in semantics[:max_support]),
        query_sentences=tuple(render(example) for example in semantics[max_support:]),
    )


def build_language_tasks(*, seed: int, task_count: int, max_support: int, query_examples: int) -> list[LanguageTask]:
    rng = random.Random(seed)
    tasks: list[LanguageTask] = []
    builders = (_build_order_task, _build_agreement_task)
    for index in range(task_count):
        builder = builders[index % len(builders)]
        tasks.append(builder(rng, max_support, query_examples))
    rng.shuffle(tasks)
    return tasks


class LanguageTaskDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, tasks: Sequence[LanguageTask], *, max_support: int, query_examples: int) -> None:
        self.tasks = list(tasks)
        self.max_support = max_support
        self.query_examples = query_examples
        self.max_sequence_length = max(
            max(len(sentence) for sentence in task.support_sentences)
            for task in self.tasks
        )
        self.max_sequence_length = max(
            self.max_sequence_length,
            max(len(sentence) for task in self.tasks for sentence in task.query_sentences),
        )

    def __len__(self) -> int:
        return len(self.tasks)

    def _encode_sentence(self, sentence: Sequence[int]) -> tuple[list[int], list[int], list[int]]:
        padded = [PAD_ID] * self.max_sequence_length
        padded[: len(sentence)] = sentence
        input_ids = padded[:-1]
        targets = padded[1:]
        mask = [1 if token != PAD_ID else 0 for token in targets]
        return input_ids, targets, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        task = self.tasks[index]
        support_ids = torch.zeros((self.max_support, self.max_sequence_length - 1), dtype=torch.long)
        support_targets = torch.zeros_like(support_ids)
        support_mask = torch.zeros_like(support_ids, dtype=torch.float32)
        support_present = torch.zeros(self.max_support, dtype=torch.float32)
        for slot, sentence in enumerate(task.support_sentences[: self.max_support]):
            input_ids, targets, token_mask = self._encode_sentence(sentence)
            support_ids[slot] = torch.tensor(input_ids, dtype=torch.long)
            support_targets[slot] = torch.tensor(targets, dtype=torch.long)
            support_mask[slot] = torch.tensor(token_mask, dtype=torch.float32)
            support_present[slot] = 1.0

        query_ids = torch.zeros((self.query_examples, self.max_sequence_length - 1), dtype=torch.long)
        query_targets = torch.zeros_like(query_ids)
        query_mask = torch.zeros_like(query_ids, dtype=torch.float32)
        for slot, sentence in enumerate(task.query_sentences[: self.query_examples]):
            input_ids, targets, token_mask = self._encode_sentence(sentence)
            query_ids[slot] = torch.tensor(input_ids, dtype=torch.long)
            query_targets[slot] = torch.tensor(targets, dtype=torch.long)
            query_mask[slot] = torch.tensor(token_mask, dtype=torch.float32)
        family_id = 0 if task.family == "order_map" else 1
        return {
            "support_ids": support_ids,
            "support_targets": support_targets,
            "support_mask": support_mask,
            "support_present": support_present,
            "query_ids": query_ids,
            "query_targets": query_targets,
            "query_mask": query_mask,
            "family_id": torch.tensor(family_id, dtype=torch.long),
        }


class TaskConditionedGRULM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        memory_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.task_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.memory_scale = nn.Parameter(torch.tensor(6.0))
        self.output = nn.Linear(hidden_dim, vocab_size)

    def _encode_sentences(self, input_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        outputs, _ = self.encoder(embeddings)
        lengths = token_mask.sum(dim=1).long().clamp_min(1)
        gather_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, outputs.size(-1))
        return outputs.gather(1, gather_index).squeeze(1)

    def encode_support(
        self,
        support_ids: torch.Tensor,
        support_targets: torch.Tensor,
        support_mask: torch.Tensor,
        support_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, support_size, sequence_length = support_ids.shape
        embeddings = self.embedding(support_ids.view(batch_size * support_size, sequence_length))
        flat_sequence_states, _ = self.encoder(embeddings)
        sequence_states = flat_sequence_states.view(batch_size, support_size, sequence_length, -1)
        flat_states = self._encode_sentences(
            support_ids.reshape(batch_size * support_size, sequence_length),
            support_mask.reshape(batch_size * support_size, sequence_length),
        ).reshape(batch_size, support_size, -1)
        weights = support_present.unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (flat_states * weights).sum(dim=1) / denom
        no_support = (support_present.sum(dim=1, keepdim=True) == 0).float()
        pooled = pooled * (1.0 - no_support)
        memory_mask = support_present.unsqueeze(-1) * support_mask
        return sequence_states, support_targets, memory_mask, torch.tanh(self.task_proj(pooled))

    def forward(
        self,
        *,
        support_ids: torch.Tensor,
        support_targets: torch.Tensor | None = None,
        support_mask: torch.Tensor,
        support_present: torch.Tensor,
        query_ids: torch.Tensor,
    ) -> torch.Tensor:
        if support_targets is None:
            raise ValueError("TaskConditionedGRULM requires support_targets for associative retrieval.")
        support_states, support_targets, support_memory_mask, task_vector = self.encode_support(
            support_ids,
            support_targets,
            support_mask,
            support_present,
        )
        batch_size, query_count, sequence_length = query_ids.shape
        query_embeddings = self.embedding(query_ids.reshape(batch_size * query_count, sequence_length))
        query_states, _ = self.encoder(query_embeddings)
        query_states = query_states.reshape(batch_size, query_count, sequence_length, -1)
        query_states = query_states + task_vector.unsqueeze(1).unsqueeze(1)
        flat_query_states = query_states.reshape(batch_size, query_count * sequence_length, -1)
        flat_query_keys = self.query_proj(flat_query_states)
        flat_support_keys = self.key_proj(support_states.reshape(batch_size, -1, support_states.size(-1)))
        flat_support_targets = support_targets.reshape(batch_size, -1)
        flat_memory_mask = support_memory_mask.reshape(batch_size, -1)
        scores = torch.matmul(flat_query_keys, flat_support_keys.transpose(1, 2)) / math.sqrt(flat_query_keys.size(-1))
        memory_mask = flat_memory_mask.unsqueeze(1).expand(-1, flat_query_keys.size(1), -1)
        scores = scores.masked_fill(memory_mask == 0, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * memory_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        memory_votes = torch.zeros(
            batch_size,
            flat_query_keys.size(1),
            self.output.out_features,
            dtype=attention.dtype,
            device=attention.device,
        )
        target_index = flat_support_targets.unsqueeze(1).expand(-1, flat_query_keys.size(1), -1)
        memory_votes.scatter_add_(dim=2, index=target_index, src=attention)
        memory_votes = memory_votes.reshape(batch_size * query_count, sequence_length, -1)
        query_states = self.dropout(query_states.reshape(batch_size * query_count, sequence_length, -1))
        base_logits = self.output(query_states)
        gate = torch.sigmoid(self.gate(query_states))
        logits = base_logits + gate * self.memory_scale * memory_votes
        return logits.view(batch_size, query_count, sequence_length, -1)


class GPT2Block(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        states: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_input = self.ln_1(states)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        states = states + self.dropout(attn_output)
        states = states + self.mlp(self.ln_2(states))
        return states


class GPT2LikeLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        layers: int,
        ff_dim: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.blocks = nn.ModuleList(
            GPT2Block(
                d_model=d_model,
                n_heads=n_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.output.weight = self.embedding.weight

    def _run_blocks(self, states: torch.Tensor, *, token_mask: torch.Tensor) -> torch.Tensor:
        causal_mask = torch.triu(
            torch.ones((states.size(1), states.size(1)), dtype=torch.bool, device=states.device),
            diagonal=1,
        )
        key_padding_mask = token_mask == 0
        for block in self.blocks:
            states = block(states, causal_mask=causal_mask, key_padding_mask=key_padding_mask)
        return self.final_norm(states)

    def _build_prefix(
        self,
        *,
        support_ids: torch.Tensor,
        support_targets: torch.Tensor,
        support_mask: torch.Tensor,
        support_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, support_size, sequence_length = support_ids.shape
        sentence_length = sequence_length + 1
        full_support_tokens = torch.full(
            (batch_size, support_size, sentence_length),
            PAD_ID,
            dtype=support_ids.dtype,
            device=support_ids.device,
        )
        full_support_mask = torch.zeros(
            (batch_size, support_size, sentence_length),
            dtype=torch.bool,
            device=support_ids.device,
        )
        full_support_tokens[:, :, :sequence_length] = support_ids
        full_support_mask[:, :, :sequence_length] = support_mask.bool()
        lengths = support_mask.sum(dim=-1).long().clamp_min(1)
        terminal = support_targets.gather(-1, (lengths - 1).unsqueeze(-1)).squeeze(-1)
        full_support_tokens.scatter_(2, lengths.unsqueeze(-1), terminal.unsqueeze(-1))
        full_support_mask.scatter_(2, lengths.unsqueeze(-1), support_present.bool().unsqueeze(-1))
        active_mask = support_present.bool().unsqueeze(-1)
        full_support_tokens = full_support_tokens * active_mask.long()
        full_support_mask = full_support_mask & active_mask
        prefix_tokens = full_support_tokens.reshape(batch_size, support_size * sentence_length)
        prefix_mask = full_support_mask.reshape(batch_size, support_size * sentence_length)
        return prefix_tokens, prefix_mask

    def forward(
        self,
        *,
        support_ids: torch.Tensor,
        support_targets: torch.Tensor | None = None,
        support_mask: torch.Tensor,
        support_present: torch.Tensor,
        query_ids: torch.Tensor,
    ) -> torch.Tensor:
        if support_targets is None:
            raise ValueError("GPT2LikeLM requires support_targets for in-context prefix reconstruction.")
        batch_size, query_count, sequence_length = query_ids.shape
        prefix_tokens, prefix_mask = self._build_prefix(
            support_ids=support_ids,
            support_targets=support_targets,
            support_mask=support_mask,
            support_present=support_present,
        )
        prefix_length = prefix_tokens.size(1)
        expanded_prefix_tokens = prefix_tokens.unsqueeze(1).expand(-1, query_count, -1)
        expanded_prefix_mask = prefix_mask.unsqueeze(1).expand(-1, query_count, -1)
        query_mask = query_ids != PAD_ID
        combined_ids = torch.cat([expanded_prefix_tokens, query_ids], dim=-1)
        combined_mask = torch.cat([expanded_prefix_mask, query_mask], dim=-1)
        flat_ids = combined_ids.reshape(batch_size * query_count, prefix_length + sequence_length)
        flat_mask = combined_mask.reshape(batch_size * query_count, prefix_length + sequence_length)
        positions = torch.arange(flat_ids.size(1), device=query_ids.device).unsqueeze(0).expand_as(flat_ids)
        states = self.embedding(flat_ids) + self.position_embedding(positions)
        encoded = self.dropout(self._run_blocks(states, token_mask=flat_mask.float()))
        logits = self.output(encoded[:, prefix_length:, :])
        return logits.view(batch_size, query_count, sequence_length, -1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_models(config: BenchmarkConfig, *, max_length: int) -> dict[str, nn.Module]:
    memory_dim = config.gru_memory_dim or min(config.gru_embedding_dim, config.gru_hidden_dim)
    context_length = config.max_support * (max_length + 1) + max_length
    return {
        "fast_gru": TaskConditionedGRULM(
            vocab_size=VOCAB_SIZE,
            embedding_dim=config.gru_embedding_dim,
            hidden_dim=config.gru_hidden_dim,
            memory_dim=memory_dim,
            dropout=config.dropout,
        ),
        "gpt2_like": GPT2LikeLM(
            vocab_size=VOCAB_SIZE,
            d_model=config.gpt2_d_model,
            n_heads=config.gpt2_heads,
            layers=config.gpt2_layers,
            ff_dim=config.gpt2_ff_dim,
            dropout=config.dropout,
            max_length=context_length,
        ),
    }


def _task_support_present(batch_size: int, max_support: int, *, min_support: int, device: torch.device) -> torch.Tensor:
    shots = torch.randint(min_support, max_support + 1, (batch_size,), device=device)
    slots = torch.arange(max_support, device=device).unsqueeze(0).expand(batch_size, -1)
    return (slots < shots.unsqueeze(1)).float()


def _batch_loss_and_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, float]:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    losses = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    loss = (losses * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
    predictions = flat_logits.argmax(dim=-1)
    accuracy = ((predictions == flat_targets).float() * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
    return loss, float(accuracy.item())


def _masked_loss_totals(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    losses = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    loss_sum = (losses * flat_mask).sum()
    hit_sum = ((flat_logits.argmax(dim=-1) == flat_targets).float() * flat_mask).sum()
    token_total = flat_mask.sum().clamp_min(1.0)
    return loss_sum, float(hit_sum.item()), float(token_total.item())


def _sequence_focus_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    token_nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    scaled = token_nll / max(temperature, 1e-4)
    scaled = scaled.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
    weights = torch.softmax(scaled, dim=-1)
    weights = weights * mask
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    focused = (weights * token_nll).sum(dim=-1)
    active = (mask.sum(dim=-1) > 0).float()
    return (focused * active).sum() / active.sum().clamp_min(1.0)


def _sequence_exact_hits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_hits = (logits.argmax(dim=-1) == targets).float() * mask
    return (token_hits.sum(dim=-1) == mask.sum(dim=-1)).float()


def _autoregressive_rollout(
    model: nn.Module,
    *,
    support_ids: torch.Tensor,
    support_targets: torch.Tensor,
    support_mask: torch.Tensor,
    support_present: torch.Tensor,
    query_ids: torch.Tensor,
    teacher_force_next_ids: torch.Tensor | None = None,
    teacher_force_prob: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    rollout_ids = torch.zeros_like(query_ids)
    rollout_ids[:, :, 0] = query_ids[:, :, 0]
    input_mask = query_ids != PAD_ID
    step_logits: list[torch.Tensor] = []
    for position in range(query_ids.size(-1)):
        logits = model(
            support_ids=support_ids,
            support_targets=support_targets,
            support_mask=support_mask,
            support_present=support_present,
            query_ids=rollout_ids,
        )
        current_logits = logits[:, :, position, :]
        step_logits.append(current_logits)
        if position + 1 >= query_ids.size(-1):
            continue
        next_tokens = current_logits.argmax(dim=-1)
        next_valid = input_mask[:, :, position + 1]
        if teacher_force_next_ids is not None and teacher_force_prob > 0.0:
            teacher_choice = teacher_force_next_ids[:, :, position + 1]
            teacher_mask = torch.rand(
                next_tokens.shape,
                device=next_tokens.device,
            ) < teacher_force_prob
            next_tokens = torch.where(teacher_mask & next_valid, teacher_choice, next_tokens)
        next_rollout_ids = rollout_ids.clone()
        next_rollout_ids[:, :, position + 1] = torch.where(
            next_valid,
            next_tokens,
            torch.full_like(next_tokens, PAD_ID),
        )
        rollout_ids = next_rollout_ids
    return torch.stack(step_logits, dim=2), rollout_ids


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _scheduled_rollout_weight(config: BenchmarkConfig, epoch_index: int) -> float:
    if config.rollout_loss_weight <= 0.0:
        return 0.0
    if config.rollout_weight_ramp_epochs <= 0:
        return config.rollout_loss_weight
    progress = min((epoch_index + 1) / max(config.rollout_weight_ramp_epochs, 1), 1.0)
    return config.rollout_loss_weight * progress


def _scheduled_teacher_force_prob(config: BenchmarkConfig, epoch_index: int) -> float:
    if config.rollout_teacher_forcing_prob <= 0.0:
        return 0.0
    if config.rollout_teacher_forcing_decay_epochs <= 0:
        return config.rollout_teacher_forcing_prob
    progress = min(epoch_index / max(config.rollout_teacher_forcing_decay_epochs, 1), 1.0)
    return config.rollout_teacher_forcing_prob * (1.0 - progress)


def evaluate_model(
    model: nn.Module,
    dataset: LanguageTaskDataset,
    *,
    support_shots: Iterable[int],
    batch_size: int,
    device: str,
) -> dict[str, dict[str, float]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device_obj = torch.device(device)
    use_amp = device_obj.type == "cuda"
    model.eval()
    metrics: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for shots in support_shots:
            loss_sum = 0.0
            token_hit_total = 0.0
            token_total = 0.0
            sequence_hit_total = 0.0
            sequence_total = 0.0
            autoregressive_token_hit_total = 0.0
            autoregressive_token_total = 0.0
            autoregressive_sequence_hit_total = 0.0
            autoregressive_sequence_total = 0.0
            family_hits = [0, 0]
            family_totals = [0, 0]
            autoregressive_family_hits = [0, 0]
            autoregressive_family_totals = [0, 0]
            for batch in loader:
                batch = _move_batch(batch, device_obj)
                support_present = torch.zeros_like(batch["support_present"])
                if shots > 0:
                    support_present[:, : min(shots, support_present.size(1))] = 1.0
                with torch.autocast(device_type=device_obj.type, enabled=use_amp):
                    logits = model(
                        support_ids=batch["support_ids"],
                        support_targets=batch["support_targets"],
                        support_mask=batch["support_mask"],
                        support_present=support_present,
                        query_ids=batch["query_ids"],
                    )
                batch_loss_sum, batch_hit_sum, batch_token_total = _masked_loss_totals(
                    logits,
                    batch["query_targets"],
                    batch["query_mask"],
                )
                loss_sum += float(batch_loss_sum.item())
                token_hit_total += batch_hit_sum
                token_total += batch_token_total
                sequence_hits = _sequence_exact_hits(logits, batch["query_targets"], batch["query_mask"])
                sequence_hit_total += float(sequence_hits.sum().item())
                sequence_total += float(sequence_hits.numel())
                with torch.autocast(device_type=device_obj.type, enabled=use_amp):
                    autoregressive_logits, _rollout_ids = _autoregressive_rollout(
                        model,
                        support_ids=batch["support_ids"],
                        support_targets=batch["support_targets"],
                        support_mask=batch["support_mask"],
                        support_present=support_present,
                        query_ids=batch["query_ids"],
                    )
                autoregressive_loss_sum, autoregressive_hit_sum, autoregressive_tokens = _masked_loss_totals(
                    autoregressive_logits,
                    batch["query_targets"],
                    batch["query_mask"],
                )
                autoregressive_token_hit_total += autoregressive_hit_sum
                autoregressive_token_total += autoregressive_tokens
                autoregressive_sequence_hits = _sequence_exact_hits(
                    autoregressive_logits,
                    batch["query_targets"],
                    batch["query_mask"],
                )
                autoregressive_sequence_hit_total += float(autoregressive_sequence_hits.sum().item())
                autoregressive_sequence_total += float(autoregressive_sequence_hits.numel())
                family_ids = batch["family_id"].tolist()
                for row_index, family_id in enumerate(family_ids):
                    family_hits[family_id] += int(sequence_hits[row_index].sum().item())
                    family_totals[family_id] += sequence_hits.size(1)
                    autoregressive_family_hits[family_id] += int(autoregressive_sequence_hits[row_index].sum().item())
                    autoregressive_family_totals[family_id] += autoregressive_sequence_hits.size(1)
            metrics[str(shots)] = {
                "loss": loss_sum / max(token_total, 1.0),
                "token_accuracy": token_hit_total / max(token_total, 1.0),
                "sequence_accuracy": sequence_hit_total / max(sequence_total, 1.0),
                "order_map_sequence_accuracy": family_hits[0] / max(family_totals[0], 1),
                "agreement_marker_sequence_accuracy": family_hits[1] / max(family_totals[1], 1),
                "autoregressive_token_accuracy": autoregressive_token_hit_total / max(autoregressive_token_total, 1.0),
                "autoregressive_sequence_accuracy": autoregressive_sequence_hit_total / max(autoregressive_sequence_total, 1.0),
                "order_map_autoregressive_sequence_accuracy": autoregressive_family_hits[0] / max(autoregressive_family_totals[0], 1),
                "agreement_marker_autoregressive_sequence_accuracy": autoregressive_family_hits[1] / max(autoregressive_family_totals[1], 1),
            }
    return metrics


def train_model(
    model: nn.Module,
    train_dataset: LanguageTaskDataset,
    val_dataset: LanguageTaskDataset,
    *,
    config: BenchmarkConfig,
) -> tuple[nn.Module, float, list[dict[str, float]]]:
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(device="cuda", enabled=config.use_amp and device.type == "cuda")
    use_amp = config.use_amp and device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_val_rollout_sequence = float("-inf")
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        rollout_weight = _scheduled_rollout_weight(config, epoch)
        rollout_teacher_force_prob = _scheduled_teacher_force_prob(config, epoch)
        model.train()
        train_loss_sum = 0.0
        train_hit_total = 0.0
        train_token_total = 0.0
        train_sequence_focus_total = 0.0
        train_sequence_focus_batches = 0
        train_rollout_loss_total = 0.0
        train_rollout_sequence_hit_total = 0.0
        train_rollout_sequence_total = 0.0
        train_rollout_batches = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            support_present = _task_support_present(
                batch["support_present"].size(0),
                batch["support_present"].size(1),
                min_support=config.train_support_min,
                device=device,
            )
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(
                    support_ids=batch["support_ids"],
                    support_targets=batch["support_targets"],
                    support_mask=batch["support_mask"],
                    support_present=support_present,
                    query_ids=batch["query_ids"],
                )
                loss, _accuracy = _batch_loss_and_accuracy(logits, batch["query_targets"], batch["query_mask"])
                sequence_focus = _sequence_focus_loss(
                    logits,
                    batch["query_targets"],
                    batch["query_mask"],
                    temperature=config.sequence_focus_temperature,
                )
                if config.sequence_focus_weight > 0.0:
                    loss = loss + config.sequence_focus_weight * sequence_focus
                rollout_loss_value = torch.zeros((), device=device)
                rollout_sequence_focus = torch.zeros((), device=device)
                rollout_sequence_hits = torch.zeros(
                    (batch["query_ids"].size(0), batch["query_ids"].size(1)),
                    device=device,
                )
                if rollout_weight > 0.0:
                    rollout_logits, _rollout_ids = _autoregressive_rollout(
                        model,
                        support_ids=batch["support_ids"],
                        support_targets=batch["support_targets"],
                        support_mask=batch["support_mask"],
                        support_present=support_present,
                        query_ids=batch["query_ids"],
                        teacher_force_next_ids=batch["query_ids"],
                        teacher_force_prob=rollout_teacher_force_prob,
                    )
                    rollout_loss_value, _rollout_accuracy = _batch_loss_and_accuracy(
                        rollout_logits,
                        batch["query_targets"],
                        batch["query_mask"],
                    )
                    rollout_sequence_focus = _sequence_focus_loss(
                        rollout_logits,
                        batch["query_targets"],
                        batch["query_mask"],
                        temperature=config.sequence_focus_temperature,
                    )
                    rollout_sequence_hits = _sequence_exact_hits(
                        rollout_logits,
                        batch["query_targets"],
                        batch["query_mask"],
                    )
                    rollout_objective = rollout_loss_value
                    if config.sequence_focus_weight > 0.0:
                        rollout_objective = rollout_objective + config.sequence_focus_weight * rollout_sequence_focus
                    loss = loss + rollout_weight * rollout_objective
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_loss_sum, batch_hit_sum, batch_token_total = _masked_loss_totals(
                logits.detach(),
                batch["query_targets"],
                batch["query_mask"],
            )
            train_loss_sum += float(batch_loss_sum.item())
            train_hit_total += batch_hit_sum
            train_token_total += batch_token_total
            train_sequence_focus_total += float(sequence_focus.item())
            train_sequence_focus_batches += 1
            if rollout_weight > 0.0:
                train_rollout_loss_total += float(rollout_loss_value.item())
                train_rollout_sequence_hit_total += float(rollout_sequence_hits.sum().item())
                train_rollout_sequence_total += float(rollout_sequence_hits.numel())
                train_rollout_batches += 1
        model.eval()
        val_loss_sum = 0.0
        val_hit_total = 0.0
        val_token_total = 0.0
        val_sequence_focus_total = 0.0
        val_sequence_focus_batches = 0
        val_rollout_loss_total = 0.0
        val_rollout_batches = 0
        val_rollout_sequence_hit_total = 0.0
        val_rollout_sequence_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                support_present = batch["support_present"]
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(
                        support_ids=batch["support_ids"],
                        support_targets=batch["support_targets"],
                        support_mask=batch["support_mask"],
                        support_present=support_present,
                        query_ids=batch["query_ids"],
                    )
                    _loss, _accuracy = _batch_loss_and_accuracy(logits, batch["query_targets"], batch["query_mask"])
                    sequence_focus = _sequence_focus_loss(
                        logits,
                        batch["query_targets"],
                        batch["query_mask"],
                        temperature=config.sequence_focus_temperature,
                    )
                    rollout_loss_value = torch.zeros((), device=device)
                    rollout_sequence_hits = torch.zeros(
                        (batch["query_ids"].size(0), batch["query_ids"].size(1)),
                        device=device,
                    )
                    if rollout_weight > 0.0:
                        rollout_logits, _rollout_ids = _autoregressive_rollout(
                            model,
                            support_ids=batch["support_ids"],
                            support_targets=batch["support_targets"],
                            support_mask=batch["support_mask"],
                            support_present=support_present,
                            query_ids=batch["query_ids"],
                        )
                        rollout_loss_value, _rollout_accuracy = _batch_loss_and_accuracy(
                            rollout_logits,
                            batch["query_targets"],
                            batch["query_mask"],
                        )
                        rollout_sequence_hits = _sequence_exact_hits(
                            rollout_logits,
                            batch["query_targets"],
                            batch["query_mask"],
                        )
                batch_loss_sum, batch_hit_sum, batch_token_total = _masked_loss_totals(
                    logits,
                    batch["query_targets"],
                    batch["query_mask"],
                )
                val_loss_sum += float(batch_loss_sum.item())
                val_hit_total += batch_hit_sum
                val_token_total += batch_token_total
                val_sequence_focus_total += float(sequence_focus.item())
                val_sequence_focus_batches += 1
                if rollout_weight > 0.0:
                    val_rollout_loss_total += float(rollout_loss_value.item())
                    val_rollout_sequence_hit_total += float(rollout_sequence_hits.sum().item())
                    val_rollout_sequence_total += float(rollout_sequence_hits.numel())
                    val_rollout_batches += 1
        train_loss = train_loss_sum / max(train_token_total, 1.0)
        train_accuracy = train_hit_total / max(train_token_total, 1.0)
        val_loss = val_loss_sum / max(val_token_total, 1.0)
        val_accuracy = val_hit_total / max(val_token_total, 1.0)
        train_rollout_sequence_accuracy = train_rollout_sequence_hit_total / max(train_rollout_sequence_total, 1.0)
        val_rollout_sequence_accuracy = val_rollout_sequence_hit_total / max(val_rollout_sequence_total, 1.0)
        history.append(
            {
                "epoch": float(epoch + 1),
                "rollout_weight": rollout_weight,
                "rollout_teacher_forcing_prob": rollout_teacher_force_prob,
                "train_loss": train_loss,
                "train_token_accuracy": train_accuracy,
                "train_sequence_focus_loss": train_sequence_focus_total / max(train_sequence_focus_batches, 1),
                "train_rollout_loss": train_rollout_loss_total / max(train_rollout_batches, 1),
                "train_rollout_sequence_accuracy": train_rollout_sequence_accuracy,
                "val_loss": val_loss,
                "val_token_accuracy": val_accuracy,
                "val_sequence_focus_loss": val_sequence_focus_total / max(val_sequence_focus_batches, 1),
                "val_rollout_loss": val_rollout_loss_total / max(val_rollout_batches, 1),
                "val_rollout_sequence_accuracy": val_rollout_sequence_accuracy,
            }
        )
        should_update = False
        if rollout_weight > 0.0 or config.rollout_loss_weight > 0.0:
            if val_rollout_sequence_accuracy > best_val_rollout_sequence + 1e-9:
                should_update = True
            elif math.isclose(val_rollout_sequence_accuracy, best_val_rollout_sequence, rel_tol=0.0, abs_tol=1e-9) and val_loss < best_val:
                should_update = True
        elif val_loss < best_val:
            should_update = True
        if should_update:
            best_val = val_loss
            best_val_rollout_sequence = val_rollout_sequence_accuracy
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return model, best_val, history


def adaptation_auc(metrics: dict[str, dict[str, float]], support_shots: Sequence[int]) -> float:
    if len(support_shots) < 2:
        return metrics[str(support_shots[0])]["token_accuracy"]
    area = 0.0
    shots = list(support_shots)
    for left, right in zip(shots, shots[1:]):
        left_acc = metrics[str(left)]["token_accuracy"]
        right_acc = metrics[str(right)]["token_accuracy"]
        area += (right - left) * (left_acc + right_acc) / 2.0
    max_support = max(shots[-1] - shots[0], 1)
    return area / max_support


def run_single_seed(config: BenchmarkConfig) -> dict[str, ModelReport]:
    set_global_seed(config.seed)
    train_tasks = build_language_tasks(
        seed=config.seed,
        task_count=config.train_tasks,
        max_support=config.max_support,
        query_examples=config.query_examples,
    )
    val_tasks = build_language_tasks(
        seed=config.seed + 1_000,
        task_count=config.val_tasks,
        max_support=config.max_support,
        query_examples=config.query_examples,
    )
    test_tasks = build_language_tasks(
        seed=config.seed + 2_000,
        task_count=config.test_tasks,
        max_support=config.max_support,
        query_examples=config.query_examples,
    )
    train_dataset = LanguageTaskDataset(train_tasks, max_support=config.max_support, query_examples=config.query_examples)
    val_dataset = LanguageTaskDataset(val_tasks, max_support=config.max_support, query_examples=config.query_examples)
    test_dataset = LanguageTaskDataset(test_tasks, max_support=config.max_support, query_examples=config.query_examples)
    max_length = max(
        train_dataset.max_sequence_length,
        val_dataset.max_sequence_length,
        test_dataset.max_sequence_length,
    ) - 1

    reports: dict[str, ModelReport] = {}
    for name, model in build_models(config, max_length=max_length).items():
        model, best_val, history = train_model(model, train_dataset, val_dataset, config=config)
        metrics = evaluate_model(
            model,
            test_dataset,
            support_shots=config.support_shots,
            batch_size=config.eval_batch_size,
            device=config.device,
        )
        reports[name] = ModelReport(
            name=name,
            parameter_count=count_parameters(model),
            best_val_loss=best_val,
            adaptation_auc=adaptation_auc(metrics, config.support_shots),
            autoregressive_adaptation_auc=adaptation_auc(
                {
                    shot: {
                        "token_accuracy": values["autoregressive_token_accuracy"],
                    }
                    for shot, values in metrics.items()
                },
                config.support_shots,
            ),
            eval_by_shot=metrics,
            history=history,
        )
    return reports


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def aggregate_seed_reports(seed_reports: list[dict[str, ModelReport]]) -> dict[str, dict[str, object]]:
    model_names = seed_reports[0].keys()
    payload: dict[str, dict[str, object]] = {}
    for name in model_names:
        parameter_counts = [report[name].parameter_count for report in seed_reports]
        best_val_losses = [report[name].best_val_loss for report in seed_reports]
        adaptation_aucs = [report[name].adaptation_auc for report in seed_reports]
        autoregressive_adaptation_aucs = [report[name].autoregressive_adaptation_auc for report in seed_reports]
        shot_keys = seed_reports[0][name].eval_by_shot.keys()
        eval_by_shot: dict[str, dict[str, float]] = {}
        for shot in shot_keys:
            metric_names = seed_reports[0][name].eval_by_shot[shot].keys()
            eval_by_shot[shot] = {
                metric_name: _mean([report[name].eval_by_shot[shot][metric_name] for report in seed_reports])
                for metric_name in metric_names
            }
        histories = [report[name].history for report in seed_reports]
        epoch_count = min(len(history) for history in histories)
        payload[name] = {
            "parameter_count_mean": _mean(parameter_counts),
            "parameter_count_values": parameter_counts,
            "best_val_loss_mean": _mean(best_val_losses),
            "adaptation_auc_mean": _mean(adaptation_aucs),
            "autoregressive_adaptation_auc_mean": _mean(autoregressive_adaptation_aucs),
            "eval_by_shot": eval_by_shot,
            "history": [
                {
                    metric_name: _mean([seed_history[epoch_index][metric_name] for seed_history in histories])
                    for metric_name in histories[0][epoch_index].keys()
                }
                for epoch_index in range(epoch_count)
            ],
        }
    return payload


def fairness_summary(model_reports: dict[str, dict[str, object]]) -> dict[str, object]:
    gru_params = float(model_reports["fast_gru"]["parameter_count_mean"])
    gpt2_params = float(model_reports["gpt2_like"]["parameter_count_mean"])
    gap = abs(gru_params - gpt2_params) / max(gru_params, gpt2_params, 1.0)
    return {
        "fast_gru_parameter_count": int(round(gru_params)),
        "gpt2_like_parameter_count": int(round(gpt2_params)),
        "relative_parameter_gap": gap,
        "parameter_gap_ok": gap <= 0.15,
        "same_vocab": True,
        "same_task_distribution": True,
        "same_optimizer": "AdamW",
        "same_schedule": True,
        "same_support_query_protocol": True,
    }


def run_benchmark(*, seeds: Sequence[int], config: BenchmarkConfig) -> dict[str, object]:
    start = time.perf_counter()
    seed_reports = []
    for seed in seeds:
        run_config = BenchmarkConfig(**{**asdict(config), "seed": seed})
        seed_reports.append(run_single_seed(run_config))
    aggregate = aggregate_seed_reports(seed_reports)
    runtime_seconds = time.perf_counter() - start
    return {
        "benchmark": "language_fastlearn_compare",
        "device": config.device,
        "seeds": list(seeds),
        "config": asdict(config),
        "fairness": fairness_summary(aggregate),
        "models": aggregate,
        "winner_by_adaptation_auc": max(
            aggregate.items(),
            key=lambda item: item[1]["adaptation_auc_mean"],
        )[0],
        "winner_by_autoregressive_adaptation_auc": max(
            aggregate.items(),
            key=lambda item: item[1]["autoregressive_adaptation_auc_mean"],
        )[0],
        "runtime_seconds": runtime_seconds,
}


SCALING_PRESETS: dict[str, dict[str, int]] = {
    "small": {
        "gru_embedding_dim": 32,
        "gru_hidden_dim": 56,
        "gru_memory_dim": 32,
        "gpt2_d_model": 32,
        "gpt2_heads": 4,
        "gpt2_layers": 2,
        "gpt2_ff_dim": 96,
    },
    "medium": {
        "gru_embedding_dim": 48,
        "gru_hidden_dim": 80,
        "gru_memory_dim": 48,
        "gpt2_d_model": 48,
        "gpt2_heads": 4,
        "gpt2_layers": 2,
        "gpt2_ff_dim": 160,
    },
    "large": {
        "gru_embedding_dim": 64,
        "gru_hidden_dim": 144,
        "gru_memory_dim": 80,
        "gpt2_d_model": 64,
        "gpt2_heads": 4,
        "gpt2_layers": 3,
        "gpt2_ff_dim": 224,
    },
}


def run_scaling_sweep(*, seeds: Sequence[int], config: BenchmarkConfig, scales: Sequence[str]) -> dict[str, object]:
    sweep: dict[str, object] = {}
    max_shot_key = str(max(config.support_shots))
    for scale in scales:
        if scale not in SCALING_PRESETS:
            raise ValueError(f"Unknown scale preset: {scale}")
        run_config = BenchmarkConfig(**{**asdict(config), **SCALING_PRESETS[scale]})
        result = run_benchmark(seeds=seeds, config=run_config)
        sweep[scale] = {
            "fairness": result["fairness"],
            "winner_by_adaptation_auc": result["winner_by_adaptation_auc"],
            "models": {
                    name: {
                        "parameter_count_mean": payload["parameter_count_mean"],
                        "adaptation_auc_mean": payload["adaptation_auc_mean"],
                        "autoregressive_adaptation_auc_mean": payload["autoregressive_adaptation_auc_mean"],
                        "shot8_token_accuracy": payload["eval_by_shot"][max_shot_key]["token_accuracy"],
                        "shot8_sequence_accuracy": payload["eval_by_shot"][max_shot_key]["sequence_accuracy"],
                        "shot8_autoregressive_token_accuracy": payload["eval_by_shot"][max_shot_key]["autoregressive_token_accuracy"],
                        "shot8_autoregressive_sequence_accuracy": payload["eval_by_shot"][max_shot_key]["autoregressive_sequence_accuracy"],
                    }
                for name, payload in result["models"].items()
            },
        }
    return {
        "benchmark": "language_fastlearn_scaling",
        "device": config.device,
        "seeds": list(seeds),
        "base_config": asdict(config),
        "scales": list(scales),
        "results": sweep,
    }


def parse_seed_list(raw: str) -> list[int]:
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair small-scale language fast-learning benchmark.")
    parser.add_argument("--mode", type=str, choices=("compare", "scaling"), default="compare")
    parser.add_argument("--seeds", type=str, default="13,29")
    parser.add_argument("--scales", type=str, default="small,medium,large")
    parser.add_argument("--train-tasks", type=int, default=512)
    parser.add_argument("--val-tasks", type=int, default=128)
    parser.add_argument("--test-tasks", type=int, default=128)
    parser.add_argument("--max-support", type=int, default=8)
    parser.add_argument("--query-examples", type=int, default=12)
    parser.add_argument("--train-support-min", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sequence-focus-weight", type=float, default=0.0)
    parser.add_argument("--sequence-focus-temperature", type=float, default=0.5)
    parser.add_argument("--rollout-loss-weight", type=float, default=0.0)
    parser.add_argument("--rollout-teacher-forcing-prob", type=float, default=0.0)
    parser.add_argument("--rollout-weight-ramp-epochs", type=int, default=0)
    parser.add_argument("--rollout-teacher-forcing-decay-epochs", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = BenchmarkConfig(
        train_tasks=args.train_tasks,
        val_tasks=args.val_tasks,
        test_tasks=args.test_tasks,
        max_support=args.max_support,
        query_examples=args.query_examples,
        train_support_min=args.train_support_min,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        sequence_focus_weight=args.sequence_focus_weight,
        sequence_focus_temperature=args.sequence_focus_temperature,
        rollout_loss_weight=args.rollout_loss_weight,
        rollout_teacher_forcing_prob=args.rollout_teacher_forcing_prob,
        rollout_weight_ramp_epochs=args.rollout_weight_ramp_epochs,
        rollout_teacher_forcing_decay_epochs=args.rollout_teacher_forcing_decay_epochs,
        device=args.device,
    )
    seeds = parse_seed_list(args.seeds)
    if args.mode == "scaling":
        payload = run_scaling_sweep(seeds=seeds, config=config, scales=[piece.strip() for piece in args.scales.split(",") if piece.strip()])
    else:
        payload = run_benchmark(seeds=seeds, config=config)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
