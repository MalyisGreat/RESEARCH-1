from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from arc_tactic3.language_fastlearn_benchmark import count_parameters, set_global_seed
from arc_tactic3.language_partial_untied_cluster import PartialUntiedClusterConfig, ensure_fineweb_cache
from arc_tactic3.language_realtext_microbench import (
    RealTextConfig,
    TokenBlockDataset,
    _build_optimizer,
    _build_train_batch_schedule,
    _dataset_tensors,
    _loss_and_tokens,
    _scheduled_batch_from_tensors,
    evaluate_loss,
)
from arc_tactic3.language_recurrent_nano_tricks import (
    FastPartialUntiedAssociativeLM,
    FactorizedUntiedHeadAssociativeLM,
    LowRankUntiedDeltaAssociativeLM,
    PartialUntiedAssociativeLM,
    _top_token_ids,
)


_DEFAULT_VARIANTS = ("partial_untied", "fast_partial_untied", "windowed_replay128")
_CAUSAL_CONV_ANCHOR_PREFIX = "causal_conv_mixer_sampled_vocab_anchor"


@dataclass(frozen=True, slots=True)
class LongSeqReplayProbeConfig:
    output_dir: Path = Path("artifacts/benchmark_runs/language/longseq_replay_probe")
    cache_path: Path | None = None
    validation_cache_path: Path | None = None
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    split: str = "train"
    text_column: str = "text"
    tokenizer_name: str = "gpt2"
    train_blocks: int = 512
    val_blocks: int = 64
    sequence_length: int = 635
    batch_size: int = 8
    eval_batch_size: int = 8
    train_steps: int = 24
    eval_interval: int = 12
    initial_eval: bool = True
    eval_loss_mode: str = "full"
    max_step_seconds: float | None = None
    max_eval_seconds: float | None = None
    reuse_variant_results: bool = False
    max_gpu_used_mb: float | None = None
    variant_checkpoint_interval: int = 0
    milestone_checkpoint_interval: int = 0
    resume_variant_checkpoints: bool = False
    resume_fresh_cache: bool = False
    train_log_interval: int = 1
    seed: int = 13
    train_token_offset: int = 0
    learning_rate: float = 2e-3
    lr_schedule: str = "constant"
    warmup_steps: int = 0
    min_learning_rate: float = 0.0
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = torch.cuda.is_available()
    amp_dtype: str = "fp16"
    pin_memory: bool = torch.cuda.is_available()
    use_fused_adamw: bool = torch.cuda.is_available()
    cache_dataset_on_device: bool = torch.cuda.is_available()
    local_files_only: bool = False
    torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    tokenization_batch_size: int = 512
    embedding_dim: int = 320
    hidden_dim: int = 640
    memory_dim: int = 320
    factorized_embedding_dim: int = 280
    factorized_hidden_dim: int = 560
    factorized_memory_dim: int = 280
    no_replay_embedding_dim: int = 260
    no_replay_hidden_dim: int = 520
    no_replay_rank: int = 96
    conv_embedding_dim: int = 282
    conv_layers: int = 1
    conv_rank: int = 96
    conv_kernel_size: int = 7
    partial_token_count: int = 1024
    dropout: float = 0.1
    window_size: int = 128
    untied_rank: int = 64
    sampled_vocab_size: int = 4096
    full_loss_interval: int = 4
    full_loss_token_stride: int = 8
    full_eval_token_chunk_size: int = 1024
    train_loss_token_chunk_size: int = 0
    hierarchical_class_count: int = 256
    variants: tuple[str, ...] = _DEFAULT_VARIANTS


_DEFAULT_CONFIG = LongSeqReplayProbeConfig()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _replace_with_retry(tmp, path)


def _replace_with_retry(source: Path, target: Path, *, attempts: int = 20) -> None:
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _run_slug(config: LongSeqReplayProbeConfig) -> str:
    initial = "init1" if config.initial_eval else "init0"
    eval_mode = "" if config.eval_loss_mode == "full" else f"_eval{config.eval_loss_mode}"
    return (
        f"seq{config.sequence_length}_steps{config.train_steps}_val{config.val_blocks}_"
        f"batch{config.batch_size}_eval{config.eval_interval}_{initial}{eval_mode}_seed{config.seed}"
    )


_REUSABLE_CONFIG_KEYS = (
    "dataset_name",
    "split",
    "text_column",
    "tokenizer_name",
    "train_blocks",
    "val_blocks",
    "sequence_length",
    "batch_size",
    "eval_batch_size",
    "train_steps",
    "eval_interval",
    "initial_eval",
    "eval_loss_mode",
    "seed",
    "learning_rate",
    "lr_schedule",
    "warmup_steps",
    "min_learning_rate",
    "weight_decay",
    "device",
    "use_amp",
    "amp_dtype",
    "pin_memory",
    "use_fused_adamw",
    "cache_dataset_on_device",
    "torch_compile",
    "torch_compile_mode",
    "embedding_dim",
    "hidden_dim",
    "memory_dim",
    "factorized_embedding_dim",
    "factorized_hidden_dim",
    "factorized_memory_dim",
    "no_replay_embedding_dim",
    "no_replay_hidden_dim",
    "no_replay_rank",
    "conv_embedding_dim",
    "conv_layers",
    "conv_rank",
    "conv_kernel_size",
    "partial_token_count",
    "dropout",
    "window_size",
    "untied_rank",
    "sampled_vocab_size",
    "full_loss_interval",
    "full_loss_token_stride",
    "hierarchical_class_count",
)

_PARTIAL_UNTIED_UNUSED_REUSABLE_KEYS = frozenset(
    {
        "factorized_embedding_dim",
        "factorized_hidden_dim",
        "factorized_memory_dim",
        "no_replay_embedding_dim",
        "no_replay_hidden_dim",
        "no_replay_rank",
        "conv_embedding_dim",
        "conv_layers",
        "conv_rank",
        "conv_kernel_size",
        "window_size",
        "untied_rank",
        "sampled_vocab_size",
        "full_loss_interval",
        "full_loss_token_stride",
        "hierarchical_class_count",
    }
)


def _load_reusable_variant_report(
    variant: str,
    config: LongSeqReplayProbeConfig,
    variant_artifact_dir: Path,
) -> dict[str, Any] | None:
    result_path = variant_artifact_dir / f"{variant}.json"
    if not result_path.exists():
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("benchmark") != "language_longseq_replay_probe_variant":
        raise ValueError(f"Reusable result for {variant} has unexpected benchmark: {result_path}")
    if payload.get("variant") != variant:
        raise ValueError(f"Reusable result for {variant} has mismatched variant: {result_path}")
    saved_config = payload.get("config", {})
    current_config = asdict(config)
    mismatches = [
        key
        for key in _reusable_config_keys_for_variant(variant)
        if _reusable_saved_config_value(saved_config, key) != current_config.get(key)
    ]
    if mismatches:
        joined = ", ".join(mismatches[:8])
        if len(mismatches) > 8:
            joined += ", ..."
        raise ValueError(f"Reusable result for {variant} does not match current config keys: {joined}")
    report = payload.get("report")
    if not isinstance(report, dict):
        raise ValueError(f"Reusable result for {variant} is missing report: {result_path}")
    return report


def _reusable_config_keys_for_variant(variant: str) -> tuple[str, ...]:
    if variant == "partial_untied":
        return tuple(key for key in _REUSABLE_CONFIG_KEYS if key not in _PARTIAL_UNTIED_UNUSED_REUSABLE_KEYS)
    return _REUSABLE_CONFIG_KEYS


def _reusable_saved_config_value(saved_config: dict[str, Any], key: str) -> Any:
    if key == "eval_loss_mode" and key not in saved_config:
        return "full"
    if key == "lr_schedule" and key not in saved_config:
        return "constant"
    if key == "warmup_steps" and key not in saved_config:
        return 0
    if key == "min_learning_rate" and key not in saved_config:
        return 0.0
    return saved_config.get(key)


def _scheduled_learning_rate(config: LongSeqReplayProbeConfig, step: int) -> float:
    if config.learning_rate < 0.0:
        raise ValueError("learning_rate must be non-negative.")
    if config.min_learning_rate < 0.0:
        raise ValueError("min_learning_rate must be non-negative.")
    if config.min_learning_rate > config.learning_rate:
        raise ValueError("min_learning_rate must be <= learning_rate.")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if config.lr_schedule == "constant":
        return config.learning_rate
    if config.lr_schedule not in {"linear", "cosine"}:
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")
    if step < 1:
        return 0.0 if config.warmup_steps > 0 else config.learning_rate
    if config.warmup_steps > 0 and step <= config.warmup_steps:
        return config.learning_rate * (step / config.warmup_steps)
    decay_steps = max(config.train_steps - config.warmup_steps, 1)
    decay_step = min(max(step - config.warmup_steps, 0), decay_steps)
    progress = decay_step / decay_steps
    if config.lr_schedule == "linear":
        decay_factor = 1.0 - progress
    else:
        decay_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * decay_factor


def _set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _full_loss_token_stride_for_variant(variant: str, config: LongSeqReplayProbeConfig) -> int:
    if not variant.startswith(_CAUSAL_CONV_ANCHOR_PREFIX):
        return config.full_loss_token_stride
    suffix = variant.removeprefix(_CAUSAL_CONV_ANCHOR_PREFIX)
    if not suffix:
        return config.full_loss_token_stride
    if not suffix.isdigit():
        raise ValueError(f"Invalid anchor stride variant suffix: {variant}")
    stride = int(suffix)
    if stride < 1:
        raise ValueError(f"Invalid anchor stride variant suffix: {variant}")
    return stride


def _comparison_vs_baseline(report: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    baseline_pure_tps = float(baseline["pure_train_tok_per_sec"])
    report_pure_tps = float(report["pure_train_tok_per_sec"])
    baseline_median_ms = float(baseline["step_time_median_ms"])
    report_median_ms = float(report["step_time_median_ms"])
    return {
        "loss_delta_vs_baseline": float(report["final_val_loss"]) - float(baseline["final_val_loss"]),
        "pure_speed_ratio_vs_baseline": report_pure_tps / max(baseline_pure_tps, 1e-9),
        "median_step_speed_ratio_vs_baseline": baseline_median_ms / max(report_median_ms, 1e-9),
        "baseline_step_time_median_ms": baseline_median_ms,
        "candidate_step_time_median_ms": report_median_ms,
    }


def _current_gpu_used_mb(device_name: str) -> float | None:
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        return None
    device = torch.device(device_name)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return (total_bytes - free_bytes) / (1024.0 * 1024.0)


def _enforce_gpu_preflight(config: LongSeqReplayProbeConfig) -> None:
    if config.max_gpu_used_mb is None:
        return
    used_mb = _current_gpu_used_mb(config.device)
    if used_mb is None:
        return
    if used_mb > config.max_gpu_used_mb:
        raise RuntimeError(
            f"GPU preflight rejected run: {used_mb:.1f}MB already used exceeds "
            f"max_gpu_used_mb={config.max_gpu_used_mb:.1f}MB."
        )


def _checkpoint_config_payload(config: LongSeqReplayProbeConfig) -> dict[str, Any]:
    config_payload = asdict(config)
    return {key: config_payload.get(key) for key in _REUSABLE_CONFIG_KEYS}


def _save_variant_checkpoint(
    path: Path,
    *,
    variant: str,
    config: LongSeqReplayProbeConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    tokens_seen: int,
    initial_val_loss: float,
    history: list[dict[str, float]],
    step_times: list[float],
    sampled_candidate_sizes: list[int],
    sampled_eval_candidate_sizes: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "language_longseq_replay_probe_variant_checkpoint",
        "variant": variant,
        "config": _checkpoint_config_payload(config),
        "step": step,
        "tokens_seen": tokens_seen,
        "initial_val_loss": initial_val_loss,
        "history": history,
        "step_times": step_times,
        "sampled_candidate_sizes": sampled_candidate_sizes,
        "sampled_eval_candidate_sizes": sampled_eval_candidate_sizes,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    _replace_with_retry(tmp, path)


def _load_variant_checkpoint(path: Path, *, variant: str, config: LongSeqReplayProbeConfig, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("benchmark") != "language_longseq_replay_probe_variant_checkpoint":
        raise ValueError(f"Unexpected checkpoint benchmark: {path}")
    if payload.get("variant") != variant:
        raise ValueError(f"Checkpoint variant mismatch for {variant}: {path}")
    saved_config = payload.get("config", {})
    current_config = _checkpoint_config_payload(config)
    extendable_keys = {"train_blocks", "train_steps"}
    runtime_placement_keys = {"cache_dataset_on_device", "pin_memory"}
    mismatches = []
    for key in _REUSABLE_CONFIG_KEYS:
        if _reusable_saved_config_value(saved_config, key) == current_config.get(key):
            continue
        if key == "train_blocks" and config.resume_fresh_cache:
            continue
        if key in extendable_keys and int(current_config.get(key, 0)) >= int(saved_config.get(key, 0)):
            continue
        if key in runtime_placement_keys:
            continue
        mismatches.append(key)
    if mismatches:
        joined = ", ".join(mismatches[:8])
        if len(mismatches) > 8:
            joined += ", ..."
        raise ValueError(f"Checkpoint for {variant} does not match current config keys: {joined}")
    return payload


class WindowedReplayPartialUntiedLM(PartialUntiedAssociativeLM):
    def __init__(self, *, window_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if window_size < 1:
            raise ValueError("window_size must be positive.")
        self.window_size = window_size

    def _window_indices(self, sequence_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(sequence_length, device=device)
        offsets = torch.arange(self.window_size, 0, -1, device=device)
        indices = positions[:, None] - offsets[None, :]
        valid = indices >= 0
        return indices.clamp_min(0), valid

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        partial_logits = self.partial_head(base_features)
        partial_index = self.untied_token_ids.view(1, 1, -1).expand(input_ids.size(0), input_ids.size(1), -1)
        base_logits.scatter_add_(2, partial_index, partial_logits.to(base_logits.dtype))

        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        indices, valid = self._window_indices(input_ids.size(1), input_ids.device)
        flat_indices = indices.reshape(-1)
        window_keys = memory_keys[:, flat_indices, :].view(
            input_ids.size(0),
            input_ids.size(1),
            self.window_size,
            memory_keys.size(-1),
        )
        scores = (query_keys.unsqueeze(2) * window_keys).sum(dim=-1) / math.sqrt(query_keys.size(-1))
        scores = scores.masked_fill(~valid.unsqueeze(0), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * valid.unsqueeze(0)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        window_token_ids = input_ids[:, flat_indices].view(input_ids.size(0), input_ids.size(1), self.window_size)
        gate = torch.sigmoid(self.gate(states))
        gated_attention = (attention * (gate * self.memory_scale)).to(base_logits.dtype)
        base_logits.scatter_add_(2, window_token_ids, gated_attention)
        return base_logits


class DetachedReplayPartialUntiedLM(PartialUntiedAssociativeLM):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        partial_logits = self.partial_head(base_features)
        partial_index = self.untied_token_ids.view(1, 1, -1).expand(input_ids.size(0), input_ids.size(1), -1)
        base_logits.scatter_add_(2, partial_index, partial_logits.to(base_logits.dtype))

        with torch.no_grad():
            replay_states = states.detach()
            query_keys = self.query_proj(replay_states)
            memory_keys = self.key_proj(replay_states)
            scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / math.sqrt(query_keys.size(-1))
            causal_mask = self._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
            attention = torch.softmax(scores, dim=-1)
            attention = attention * causal_mask
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            value_index = input_ids.unsqueeze(1).expand(-1, input_ids.size(1), -1)
            gate = torch.sigmoid(self.gate(replay_states))
            gated_attention = (attention * (gate * self.memory_scale)).to(base_logits.dtype)
        base_logits.scatter_add_(2, value_index, gated_attention)
        return base_logits


class FactorizedNoReplayLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        untied_rank: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.factor_down = nn.Linear(embedding_dim, untied_rank, bias=False)
        self.factor_up = nn.Linear(untied_rank, vocab_size, bias=True)

    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        head_features = F.relu(self.head_fc(states)).square()
        return self.head_proj(head_features)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.factor_up(self.factor_down(self.features(input_ids)))


class CausalConvMixerBlock(nn.Module):
    def __init__(self, *, dim: int, expansion: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive.")
        self.left_padding = (kernel_size - 1) * dilation
        self.conv_norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=dim,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, expansion * dim)
        self.ffn_out = nn.Linear(expansion * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.conv_norm(x).transpose(1, 2)
        conv_output = self.depthwise(F.pad(conv_input, (self.left_padding, 0))).transpose(1, 2)
        x = x + self.dropout(F.relu(conv_output).square())
        ffn_hidden = F.relu(self.ffn_in(self.ffn_norm(x))).square()
        return x + self.dropout(self.ffn_out(ffn_hidden))


class CausalConvFactorizedLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        layers: int,
        dropout: float,
        untied_rank: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        if layers < 0:
            raise ValueError("layers must be non-negative.")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.blocks = nn.ModuleList(
            CausalConvMixerBlock(
                dim=embedding_dim,
                expansion=2,
                kernel_size=kernel_size,
                dilation=2 ** (layer_index % 6),
                dropout=dropout,
            )
            for layer_index in range(layers)
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.head_fc = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.factor_down = nn.Linear(embedding_dim, untied_rank, bias=False)
        self.factor_up = nn.Linear(untied_rank, vocab_size, bias=True)

    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        states = self.embedding(input_ids)
        for block in self.blocks:
            states = block(states)
        head_features = F.relu(self.head_fc(self.final_norm(states))).square()
        return self.head_proj(head_features)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.factor_up(self.factor_down(self.features(input_ids)))


class HierarchicalSoftmaxAssociativeLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        class_count: int,
    ) -> None:
        super().__init__()
        if class_count < 2:
            raise ValueError("class_count must be >= 2.")
        self.vocab_size = vocab_size
        self.class_count = min(class_count, vocab_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.class_head = nn.Linear(embedding_dim, self.class_count)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        token_ids = torch.arange(vocab_size, dtype=torch.long)
        class_size = math.ceil(vocab_size / self.class_count)
        token_to_class = torch.div(token_ids, class_size, rounding_mode="floor").clamp_max(self.class_count - 1)
        token_to_local = token_ids - token_to_class * class_size
        class_token_ids = []
        max_class_size = 0
        for class_index in range(self.class_count):
            ids = token_ids[token_to_class == class_index]
            class_token_ids.append(ids)
            max_class_size = max(max_class_size, int(ids.numel()))
        padded_ids = torch.zeros((self.class_count, max_class_size), dtype=torch.long)
        class_masks = torch.zeros((self.class_count, max_class_size), dtype=torch.bool)
        for class_index, ids in enumerate(class_token_ids):
            padded_ids[class_index, : ids.numel()] = ids
            class_masks[class_index, : ids.numel()] = True
        self.register_buffer("token_to_class", token_to_class, persistent=False)
        self.register_buffer("token_to_local", token_to_local, persistent=False)
        self.register_buffer("class_token_ids", padded_ids, persistent=False)
        self.register_buffer("class_token_mask", class_masks, persistent=False)

    def _features(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        head_features = F.relu(self.head_fc(states)).square()
        return self.head_proj(head_features)

    def loss(self, input_ids: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, int]:
        features = self._features(input_ids)
        flat_features = features.reshape(-1, features.size(-1))
        flat_targets = targets.reshape(-1)
        if flat_targets.dtype != torch.long:
            flat_targets = flat_targets.long()
        target_classes = self.token_to_class[flat_targets]
        target_locals = self.token_to_local[flat_targets]
        class_loss = F.cross_entropy(self.class_head(flat_features), target_classes, reduction="sum")
        token_loss = flat_features.new_zeros(())
        for class_index in torch.unique(target_classes).tolist():
            class_mask = target_classes == int(class_index)
            class_features = flat_features[class_mask]
            class_ids = self.class_token_ids[int(class_index)][self.class_token_mask[int(class_index)]]
            class_logits = F.linear(
                class_features,
                self.embedding.weight.index_select(0, class_ids),
                self.output_bias.index_select(0, class_ids),
            )
            token_loss = token_loss + F.cross_entropy(class_logits, target_locals[class_mask], reduction="sum")
        token_count = int(flat_targets.numel())
        return (class_loss + token_loss) / max(token_count, 1), token_count


class SlowEvalAbort(RuntimeError):
    def __init__(self, *, elapsed_seconds: float, max_seconds: float) -> None:
        super().__init__(f"evaluation exceeded max_eval_seconds={max_seconds:.3f} after {elapsed_seconds:.3f}s")
        self.elapsed_seconds = elapsed_seconds
        self.max_seconds = max_seconds


def _configure_repo_local_hf_cache(output_dir: Path) -> None:
    hf_home = output_dir / "hf_home"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def _cache_config(config: LongSeqReplayProbeConfig) -> PartialUntiedClusterConfig:
    block_size = config.sequence_length + 1
    train_tokens = config.train_blocks * block_size
    val_tokens = config.val_blocks * block_size
    return PartialUntiedClusterConfig(
        output_dir=config.output_dir,
        cache_path=config.cache_path,
        validation_cache_path=config.validation_cache_path,
        dataset_name=config.dataset_name,
        split=config.split,
        text_column=config.text_column,
        tokenizer_name=config.tokenizer_name,
        total_tokens=train_tokens + val_tokens,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        train_token_offset=config.train_token_offset,
        sequence_length=config.sequence_length,
        seed=config.seed,
        device=config.device,
        local_files_only=config.local_files_only,
        use_amp=config.use_amp,
        amp_dtype=config.amp_dtype,
        pin_memory=config.pin_memory,
        use_fused_adamw=config.use_fused_adamw,
        cache_dataset_on_device=config.cache_dataset_on_device,
        tokenization_batch_size=config.tokenization_batch_size,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        eval_interval=config.eval_interval,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        memory_dim=config.memory_dim,
        partial_token_count=config.partial_token_count,
        dropout=config.dropout,
        save_best_checkpoint=False,
        save_latest_checkpoint=False,
        save_final_checkpoint=False,
    )


def _realtext_config(config: LongSeqReplayProbeConfig) -> RealTextConfig:
    return RealTextConfig(
        seed=config.seed,
        sequence_length=config.sequence_length,
        train_steps=config.train_steps,
        eval_interval=config.eval_interval,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        use_amp=config.use_amp,
        pin_memory=config.pin_memory,
        use_fused_adamw=config.use_fused_adamw,
        tensor_batching=True,
        cache_dataset_on_device=config.cache_dataset_on_device,
        paired_train_batches=True,
        reseed_per_model=True,
        initial_eval=config.initial_eval,
    )


def _autocast_kwargs(config: LongSeqReplayProbeConfig, device: torch.device) -> dict[str, Any]:
    if not config.use_amp or device.type != "cuda":
        return {"device_type": device.type, "enabled": False}
    dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
    return {"device_type": "cuda", "dtype": dtype, "enabled": True}


def _peak_vram_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)


def _build_model(
    variant: str,
    config: LongSeqReplayProbeConfig,
    *,
    vocab_size: int,
    partial_token_ids: torch.Tensor,
) -> nn.Module:
    common = {
        "vocab_size": vocab_size,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "memory_dim": config.memory_dim,
        "dropout": config.dropout,
        "max_length": config.sequence_length,
        "untied_token_ids": partial_token_ids,
    }
    if variant in {"partial_untied", "sampled_vocab4096", "sampled_vocab4096_full4"}:
        return PartialUntiedAssociativeLM(**common)
    if variant == "fast_partial_untied":
        return FastPartialUntiedAssociativeLM(**common)
    if variant == "windowed_replay128":
        return WindowedReplayPartialUntiedLM(window_size=config.window_size, **common)
    if variant == "detached_replay":
        return DetachedReplayPartialUntiedLM(**common)
    factorized_common = {
        key: value for key, value in common.items() if key != "untied_token_ids"
    }
    if variant == "factorized_untied":
        return FactorizedUntiedHeadAssociativeLM(
            untied_rank=config.untied_rank,
            **factorized_common,
        )
    if variant == "low_rank_untied":
        return LowRankUntiedDeltaAssociativeLM(
            untied_rank=config.untied_rank,
            **factorized_common,
        )
    factorized_20m_common = {
        "vocab_size": vocab_size,
        "embedding_dim": config.factorized_embedding_dim,
        "hidden_dim": config.factorized_hidden_dim,
        "memory_dim": config.factorized_memory_dim,
        "dropout": config.dropout,
        "max_length": config.sequence_length,
    }
    if variant == "factorized_untied_20m":
        return FactorizedUntiedHeadAssociativeLM(
            untied_rank=config.untied_rank,
            **factorized_20m_common,
        )
    if variant == "low_rank_untied_20m":
        return LowRankUntiedDeltaAssociativeLM(
            untied_rank=config.untied_rank,
            **factorized_20m_common,
        )
    if variant == "factorized_no_replay_20m":
        return FactorizedNoReplayLM(
            vocab_size=vocab_size,
            embedding_dim=config.no_replay_embedding_dim,
            hidden_dim=config.no_replay_hidden_dim,
            dropout=config.dropout,
            untied_rank=config.no_replay_rank,
        )
    if variant in {
        "causal_conv_mixer_20m",
        "causal_conv_mixer_sampled_vocab",
        "causal_conv_mixer_sampled_vocab_full4",
        "causal_conv_mixer_sampled_vocab_anchor",
    } or variant.startswith(_CAUSAL_CONV_ANCHOR_PREFIX):
        return CausalConvFactorizedLM(
            vocab_size=vocab_size,
            embedding_dim=config.conv_embedding_dim,
            layers=config.conv_layers,
            dropout=config.dropout,
            untied_rank=config.conv_rank,
            kernel_size=config.conv_kernel_size,
        )
    if variant == "hierarchical_softmax256":
        return HierarchicalSoftmaxAssociativeLM(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            class_count=config.hierarchical_class_count,
        )
    raise ValueError(f"Unknown variant: {variant}")


def _sampled_vocab_loss(
    model: PartialUntiedAssociativeLM,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    fixed_candidate_ids: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, int, int]:
    candidate_ids = torch.unique(torch.cat((fixed_candidate_ids.to(input_ids.device), targets.reshape(-1))))
    candidate_map = torch.full((vocab_size,), -1, dtype=torch.long, device=input_ids.device)
    candidate_map[candidate_ids] = torch.arange(candidate_ids.numel(), dtype=torch.long, device=input_ids.device)
    reduced_targets = candidate_map[targets.reshape(-1)]
    if bool((reduced_targets < 0).any()):
        raise RuntimeError("Sampled-vocab candidate set missed a target token.")

    embeddings = model.embedding(input_ids)
    states, _ = model.encoder(embeddings)
    states = model.dropout(states)
    head_features = F.relu(model.head_fc(states)).square()
    base_features = model.head_proj(head_features)
    candidate_weight = model.embedding.weight.index_select(0, candidate_ids)
    sampled_logits = F.linear(base_features, candidate_weight, model.output_bias.index_select(0, candidate_ids))

    partial_logits = model.partial_head(base_features)
    partial_map = torch.full((vocab_size,), -1, dtype=torch.long, device=input_ids.device)
    untied_ids = model.untied_token_ids.to(input_ids.device)
    partial_map[untied_ids] = torch.arange(untied_ids.numel(), dtype=torch.long, device=input_ids.device)
    partial_positions = partial_map[candidate_ids]
    partial_valid = partial_positions >= 0
    if bool(partial_valid.any()):
        sampled_logits[:, :, partial_valid] += partial_logits.index_select(2, partial_positions[partial_valid]).to(
            sampled_logits.dtype
        )

    query_keys = model.query_proj(states)
    memory_keys = model.key_proj(states)
    scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / math.sqrt(query_keys.size(-1))
    causal_mask = model._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
    scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
    attention = torch.softmax(scores, dim=-1)
    attention = attention * causal_mask
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    replay_indices = candidate_map[input_ids]
    replay_valid = replay_indices >= 0
    replay_scatter = replay_indices.clamp_min(0).unsqueeze(1).expand(-1, input_ids.size(1), -1)
    gate = torch.sigmoid(model.gate(states))
    gated_attention = (attention * (gate * model.memory_scale)).masked_fill(~replay_valid.unsqueeze(1), 0.0)
    sampled_logits.scatter_add_(2, replay_scatter, gated_attention.to(sampled_logits.dtype))
    loss = F.cross_entropy(sampled_logits.reshape(-1, sampled_logits.size(-1)), reduced_targets)
    return loss, int(targets.numel()), int(candidate_ids.numel())


def _factorized_sampled_vocab_loss(
    model: FactorizedNoReplayLM | CausalConvFactorizedLM,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    fixed_candidate_ids: torch.Tensor,
    vocab_size: int,
    token_chunk_size: int = 0,
) -> tuple[torch.Tensor, int, int]:
    candidate_ids = _candidate_ids_with_batch_targets(fixed_candidate_ids, targets)
    candidate_map = torch.full((vocab_size,), -1, dtype=torch.long, device=input_ids.device)
    candidate_map[candidate_ids] = torch.arange(candidate_ids.numel(), dtype=torch.long, device=input_ids.device)
    reduced_targets = candidate_map[targets.reshape(-1)]
    if bool((reduced_targets < 0).any()):
        raise RuntimeError("Sampled-vocab candidate set missed a target token.")

    base_features = model.features(input_ids)
    hidden = model.factor_down(base_features)
    candidate_weight = model.factor_up.weight.index_select(0, candidate_ids)
    candidate_bias = None
    if model.factor_up.bias is not None:
        candidate_bias = model.factor_up.bias.index_select(0, candidate_ids)
    loss_sum = _linear_cross_entropy_sum_chunked(
        hidden,
        reduced_targets.view_as(targets),
        candidate_weight,
        candidate_bias,
        token_chunk_size=token_chunk_size,
    )
    loss = loss_sum / targets.numel()
    return loss, int(targets.numel()), int(candidate_ids.numel())


def _candidate_ids_with_batch_targets(fixed_candidate_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_ids = targets.detach().reshape(-1).to("cpu").unique(sorted=True)
    fixed_ids = fixed_candidate_ids.detach().to("cpu")
    candidate_ids = torch.cat((fixed_ids, target_ids)).unique(sorted=True)
    return candidate_ids.to(targets.device, non_blocking=True)


def _linear_cross_entropy_sum_chunked(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    token_chunk_size: int,
) -> torch.Tensor:
    flat_hidden = hidden.reshape(-1, hidden.size(-1))
    flat_targets = targets.reshape(-1)
    if flat_targets.dtype != torch.long:
        flat_targets = flat_targets.long()
    if token_chunk_size <= 0 or flat_hidden.size(0) <= token_chunk_size:
        logits = F.linear(flat_hidden, weight, bias)
        return F.cross_entropy(logits, flat_targets, reduction="sum")

    loss_sum = None
    for start in range(0, flat_hidden.size(0), token_chunk_size):
        end = min(start + token_chunk_size, flat_hidden.size(0))
        logits = F.linear(flat_hidden[start:end], weight, bias)
        chunk_loss = F.cross_entropy(logits, flat_targets[start:end], reduction="sum")
        loss_sum = chunk_loss if loss_sum is None else loss_sum + chunk_loss
    if loss_sum is None:
        raise RuntimeError("Cannot compute chunked cross entropy over an empty token batch.")
    return loss_sum


def _factorized_sampled_vocab_anchor_loss(
    model: FactorizedNoReplayLM | CausalConvFactorizedLM,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    fixed_candidate_ids: torch.Tensor,
    vocab_size: int,
    token_stride: int,
    token_chunk_size: int = 0,
) -> tuple[torch.Tensor, int, int]:
    if token_stride < 1:
        raise ValueError("token_stride must be positive.")
    candidate_ids = _candidate_ids_with_batch_targets(fixed_candidate_ids, targets)
    candidate_map = torch.full((vocab_size,), -1, dtype=torch.long, device=targets.device)
    candidate_map[candidate_ids] = torch.arange(candidate_ids.numel(), device=targets.device)
    reduced_targets = candidate_map[targets.reshape(-1)]
    if bool((reduced_targets < 0).any()):
        raise RuntimeError("Sampled-vocab candidate set missed a target token.")

    base_features = model.features(input_ids)
    hidden = model.factor_down(base_features)
    candidate_weight = model.factor_up.weight.index_select(0, candidate_ids)
    candidate_bias = None
    if model.factor_up.bias is not None:
        candidate_bias = model.factor_up.bias.index_select(0, candidate_ids)
    sampled_loss_sum = _linear_cross_entropy_sum_chunked(
        hidden,
        reduced_targets.view_as(targets),
        candidate_weight,
        candidate_bias,
        token_chunk_size=token_chunk_size,
    )
    sampled_loss = sampled_loss_sum / targets.numel()

    anchor_hidden = hidden[:, ::token_stride, :]
    anchor_targets = targets[:, ::token_stride]
    anchor_loss_sum = _linear_cross_entropy_sum_chunked(
        anchor_hidden,
        anchor_targets,
        model.factor_up.weight,
        model.factor_up.bias,
        token_chunk_size=token_chunk_size,
    )
    anchor_loss = anchor_loss_sum / anchor_targets.numel()
    loss = 0.5 * (sampled_loss + anchor_loss)
    return loss, int(targets.numel()), int(candidate_ids.numel())


def _supports_factorized_full_loss(model: nn.Module) -> bool:
    return all(hasattr(model, name) for name in ("features", "factor_down", "factor_up"))


def _factorized_full_loss_chunked(
    model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    token_chunk_size: int,
) -> tuple[torch.Tensor, int]:
    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be positive.")
    features = model.features(input_ids)
    hidden = model.factor_down(features)
    total_loss = hidden.new_tensor(0.0)
    total_tokens = 0
    for start in range(0, hidden.size(1), token_chunk_size):
        end = min(start + token_chunk_size, hidden.size(1))
        logits = model.factor_up(hidden[:, start:end, :])
        chunk_targets = targets[:, start:end]
        if chunk_targets.dtype != torch.long:
            chunk_targets = chunk_targets.long()
        token_count = int(chunk_targets.numel())
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), chunk_targets.reshape(-1))
        total_loss = total_loss + loss * token_count
        total_tokens += token_count
    return total_loss / max(total_tokens, 1), total_tokens


def _evaluate_hierarchical_loss(
    model: HierarchicalSoftmaxAssociativeLM,
    val_source: tuple[torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
    autocast_kwargs: dict[str, Any],
    batch_size: int,
) -> float:
    model.eval()
    input_ids, targets = val_source
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for start in range(0, input_ids.size(0), batch_size):
            batch_input = input_ids[start : start + batch_size].to(device, non_blocking=device.type == "cuda")
            batch_targets = targets[start : start + batch_size].to(device, non_blocking=device.type == "cuda")
            if batch_targets.dtype != torch.long:
                batch_targets = batch_targets.long()
            with torch.autocast(**autocast_kwargs):
                loss, token_count = model.loss(batch_input, batch_targets)
            total_loss += float(loss.item()) * token_count
            total_tokens += token_count
    return total_loss / max(total_tokens, 1)


def _evaluate_standard_loss(
    model: nn.Module,
    val_source: tuple[torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
    autocast_kwargs: dict[str, Any],
    batch_size: int,
    token_chunk_size: int,
    max_seconds: float | None,
) -> tuple[float, float]:
    model.eval()
    input_ids, targets = val_source
    total_loss = 0.0
    total_tokens = 0
    start_time = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, input_ids.size(0), batch_size):
            batch_input = input_ids[start : start + batch_size].to(device, non_blocking=device.type == "cuda")
            batch_targets = targets[start : start + batch_size].to(device, non_blocking=device.type == "cuda")
            if batch_targets.dtype != torch.long:
                batch_targets = batch_targets.long()
            with torch.autocast(**autocast_kwargs):
                if _supports_factorized_full_loss(model):
                    loss, token_count = _factorized_full_loss_chunked(
                        model,
                        batch_input,
                        batch_targets,
                        token_chunk_size=token_chunk_size,
                    )
                else:
                    logits = model(batch_input)
                    loss, token_count = _loss_and_tokens(logits, batch_targets)
            total_loss += float(loss.item()) * token_count
            total_tokens += token_count
            elapsed = time.perf_counter() - start_time
            if max_seconds is not None and elapsed > max_seconds:
                raise SlowEvalAbort(elapsed_seconds=elapsed, max_seconds=max_seconds)
    return total_loss / max(total_tokens, 1), time.perf_counter() - start_time


def _evaluate_sampled_vocab_loss(
    model: nn.Module,
    val_source: tuple[torch.Tensor, torch.Tensor],
    *,
    fixed_candidate_ids: torch.Tensor,
    vocab_size: int,
    device: torch.device,
    autocast_kwargs: dict[str, Any],
    batch_size: int,
    max_seconds: float | None,
) -> tuple[float, float, float]:
    model.eval()
    input_ids, targets = val_source
    total_loss = 0.0
    total_tokens = 0
    candidate_sizes: list[int] = []
    start_time = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, input_ids.size(0), batch_size):
            batch_input = input_ids[start : start + batch_size].to(device, non_blocking=device.type == "cuda")
            batch_targets = targets[start : start + batch_size].to(device, non_blocking=device.type == "cuda")
            with torch.autocast(**autocast_kwargs):
                if isinstance(model, (FactorizedNoReplayLM, CausalConvFactorizedLM)):
                    loss, token_count, candidate_size = _factorized_sampled_vocab_loss(
                        model,
                        batch_input,
                        batch_targets,
                        fixed_candidate_ids=fixed_candidate_ids,
                        vocab_size=vocab_size,
                    )
                elif isinstance(model, PartialUntiedAssociativeLM):
                    loss, token_count, candidate_size = _sampled_vocab_loss(
                        model,
                        batch_input,
                        batch_targets,
                        fixed_candidate_ids=fixed_candidate_ids,
                        vocab_size=vocab_size,
                    )
                else:
                    raise TypeError(f"sampled_vocab eval does not support {type(model).__name__}.")
            total_loss += float(loss.item()) * token_count
            total_tokens += token_count
            candidate_sizes.append(candidate_size)
            elapsed = time.perf_counter() - start_time
            if max_seconds is not None and elapsed > max_seconds:
                raise SlowEvalAbort(elapsed_seconds=elapsed, max_seconds=max_seconds)
    candidate_mean = statistics.fmean(candidate_sizes) if candidate_sizes else 0.0
    return total_loss / max(total_tokens, 1), time.perf_counter() - start_time, candidate_mean


def _train_variant(
    variant: str,
    train_dataset: TokenBlockDataset,
    val_dataset: TokenBlockDataset,
    *,
    vocab_size: int,
    partial_token_ids: torch.Tensor,
    sampled_candidate_ids: torch.Tensor,
    batch_schedule: list[torch.Tensor],
    config: LongSeqReplayProbeConfig,
    variant_artifact_dir: Path | None = None,
) -> dict[str, Any]:
    set_global_seed(config.seed)
    device = torch.device(config.device)
    model = _build_model(variant, config, vocab_size=vocab_size, partial_token_ids=partial_token_ids).to(device)
    parameter_count = count_parameters(model)
    is_hierarchical = isinstance(model, HierarchicalSoftmaxAssociativeLM)
    is_factorized = isinstance(model, (FactorizedNoReplayLM, CausalConvFactorizedLM))
    if is_hierarchical and config.eval_loss_mode != "full":
        raise ValueError("hierarchical_softmax256 only supports full validation.")
    if config.torch_compile:
        if is_hierarchical:
            raise ValueError("torch_compile is not supported for hierarchical_softmax256.")
        model = torch.compile(model, mode=config.torch_compile_mode)
    real_config = _realtext_config(config)
    optimizer = _build_optimizer(model, real_config, model_name=variant)
    scaler = torch.amp.GradScaler(device="cuda", enabled=config.use_amp and device.type == "cuda" and config.amp_dtype == "fp16")
    parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]
    train_source = _dataset_tensors(
        train_dataset,
        device=device,
        cache_on_device=config.cache_dataset_on_device,
        pin_memory=config.pin_memory,
    )
    val_source = _dataset_tensors(
        val_dataset,
        device=device,
        cache_on_device=config.cache_dataset_on_device,
        pin_memory=config.pin_memory,
    )
    autocast_kwargs = _autocast_kwargs(config, device)
    state_path = variant_artifact_dir / f"{variant}.state.json" if variant_artifact_dir is not None else None
    result_path = variant_artifact_dir / f"{variant}.json" if variant_artifact_dir is not None else None
    checkpoint_path = variant_artifact_dir / f"{variant}.checkpoint.pt" if variant_artifact_dir is not None else None
    if state_path is not None:
        _write_json_atomic(
            state_path,
            {
                "status": "starting",
                "variant": variant,
                "step": 0,
                "train_steps": config.train_steps,
                "tokens_seen": 0,
                "parameter_count": parameter_count,
                "sequence_length": config.sequence_length,
                "val_blocks": config.val_blocks,
                "validation_loss_mode": config.eval_loss_mode,
                "learning_rate": config.learning_rate,
                "lr_schedule": config.lr_schedule,
                "warmup_steps": config.warmup_steps,
                "min_learning_rate": config.min_learning_rate,
            },
        )
        print(f"starting {variant}: params={parameter_count:,} seq={config.sequence_length} steps={config.train_steps}", flush=True)
    initial_val_loss = float("nan")
    history: list[dict[str, float]] = []
    step_times: list[float] = []
    sampled_candidate_sizes: list[int] = []
    sampled_eval_candidate_sizes: list[float] = []
    tokens_seen = 0
    start_step = 1
    if config.resume_variant_checkpoints and checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = _load_variant_checkpoint(checkpoint_path, variant=variant, config=config, device=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
        if cuda_rng_state_all is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state_all])
        initial_val_loss = float(checkpoint.get("initial_val_loss", float("nan")))
        history = list(checkpoint.get("history", []))
        step_times = list(checkpoint.get("step_times", []))
        sampled_candidate_sizes = list(checkpoint.get("sampled_candidate_sizes", []))
        sampled_eval_candidate_sizes = list(checkpoint.get("sampled_eval_candidate_sizes", []))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        start_step = int(checkpoint.get("step", 0)) + 1
        if state_path is not None:
            _write_json_atomic(
                state_path,
                {
                    "status": "resumed",
                    "variant": variant,
                    "step": start_step - 1,
                    "train_steps": config.train_steps,
                    "tokens_seen": tokens_seen,
                    "checkpoint_path": str(checkpoint_path),
                    "validation_loss_mode": config.eval_loss_mode,
                    "latest_learning_rate": float(_scheduled_learning_rate(config, max(start_step - 1, 0))),
                    "lr_schedule": config.lr_schedule,
                },
            )
        print(f"resumed {variant}: step={start_step - 1}/{config.train_steps}", flush=True)
    elif config.initial_eval and is_hierarchical:
        eval_start = time.perf_counter()
        initial_val_loss = _evaluate_hierarchical_loss(
            model,
            val_source,
            device=device,
            autocast_kwargs=autocast_kwargs,
            batch_size=config.eval_batch_size,
        )
        eval_duration = time.perf_counter() - eval_start
        if config.max_eval_seconds is not None and eval_duration > config.max_eval_seconds:
            if state_path is not None:
                _write_json_atomic(
                    state_path,
                    {
                        "status": "aborted_slow_eval",
                        "variant": variant,
                        "step": 0,
                        "train_steps": config.train_steps,
                        "eval_time_ms": eval_duration * 1000.0,
                        "max_eval_seconds": config.max_eval_seconds,
                        "peak_vram_mb": _peak_vram_mb(device),
                    },
                )
            raise RuntimeError(
                f"{variant} evaluation at step 0 took {eval_duration:.3f}s, "
                f"exceeding max_eval_seconds={config.max_eval_seconds:.3f}."
            )
    elif config.initial_eval:
        try:
            if config.eval_loss_mode == "full":
                initial_val_loss, eval_duration = _evaluate_standard_loss(
                    model,
                    val_source,
                    device=device,
                    autocast_kwargs=autocast_kwargs,
                    batch_size=config.eval_batch_size,
                    token_chunk_size=config.full_eval_token_chunk_size,
                    max_seconds=config.max_eval_seconds,
                )
            elif config.eval_loss_mode == "sampled_vocab":
                initial_val_loss, eval_duration, _ = _evaluate_sampled_vocab_loss(
                    model,
                    val_source,
                    fixed_candidate_ids=sampled_candidate_ids,
                    vocab_size=vocab_size,
                    device=device,
                    autocast_kwargs=autocast_kwargs,
                    batch_size=config.eval_batch_size,
                    max_seconds=config.max_eval_seconds,
                )
            else:
                raise ValueError(f"Unknown eval_loss_mode: {config.eval_loss_mode}")
        except SlowEvalAbort as error:
            if state_path is not None:
                _write_json_atomic(
                    state_path,
                    {
                        "status": "aborted_slow_eval",
                        "variant": variant,
                        "step": 0,
                        "train_steps": config.train_steps,
                        "eval_time_ms": error.elapsed_seconds * 1000.0,
                        "max_eval_seconds": config.max_eval_seconds,
                        "peak_vram_mb": _peak_vram_mb(device),
                    },
                )
            raise RuntimeError(
                f"{variant} evaluation at step 0 took {error.elapsed_seconds:.3f}s, "
                f"exceeding max_eval_seconds={config.max_eval_seconds:.3f}."
            ) from error
    if state_path is not None and config.initial_eval and start_step == 1:
        _write_json_atomic(
            state_path,
            {
                "status": "running",
                "variant": variant,
                "step": 0,
                "train_steps": config.train_steps,
                "tokens_seen": 0,
                "initial_val_loss": float(initial_val_loss),
                "latest_val_loss": float(initial_val_loss),
                "parameter_count": parameter_count,
                "peak_vram_mb": _peak_vram_mb(device),
                "validation_loss_mode": config.eval_loss_mode,
                "latest_learning_rate": 0.0,
                "lr_schedule": config.lr_schedule,
            },
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if config.initial_eval and not history:
        history.append(
            {
                "step": 0.0,
                "tokens_seen": 0.0,
                "train_loss": float("nan"),
                "val_loss": float(initial_val_loss),
                "learning_rate": 0.0,
            }
        )
    start_time = time.perf_counter()
    schedule_start = 0 if config.resume_fresh_cache and start_step > 1 else start_step - 1
    remaining_steps = max(config.train_steps - start_step + 1, 0)
    active_schedule = batch_schedule[schedule_start : schedule_start + remaining_steps]
    for step_offset, batch_indices in enumerate(active_schedule):
        step = start_step + step_offset
        current_lr = _scheduled_learning_rate(config, step)
        _set_optimizer_learning_rate(optimizer, current_lr)
        batch = _scheduled_batch_from_tensors(
            train_source[0],
            train_source[1],
            batch_indices,
            device=device,
            non_blocking=config.pin_memory and device.type == "cuda",
        )
        step_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(**autocast_kwargs):
            if is_hierarchical:
                loss, token_count = model.loss(batch["input_ids"], batch["targets"])
            else:
                use_sampled_loss = variant == "sampled_vocab4096" or (
                    variant == "sampled_vocab4096_full4" and step % config.full_loss_interval != 0
                )
                if use_sampled_loss:
                    loss, token_count, candidate_size = _sampled_vocab_loss(
                        model,
                        batch["input_ids"],
                        batch["targets"],
                        fixed_candidate_ids=sampled_candidate_ids,
                        vocab_size=vocab_size,
                    )
                    sampled_candidate_sizes.append(candidate_size)
                elif variant == "causal_conv_mixer_sampled_vocab" or (
                    variant == "causal_conv_mixer_sampled_vocab_full4" and step % config.full_loss_interval != 0
                ):
                    if not is_factorized:
                        raise TypeError(f"{variant} requires a factorized model.")
                    loss, token_count, candidate_size = _factorized_sampled_vocab_loss(
                        model,
                        batch["input_ids"],
                        batch["targets"],
                        fixed_candidate_ids=sampled_candidate_ids,
                        vocab_size=vocab_size,
                        token_chunk_size=config.train_loss_token_chunk_size,
                    )
                    sampled_candidate_sizes.append(candidate_size)
                elif variant.startswith(_CAUSAL_CONV_ANCHOR_PREFIX):
                    if not is_factorized:
                        raise TypeError(f"{variant} requires a factorized model.")
                    loss, token_count, candidate_size = _factorized_sampled_vocab_anchor_loss(
                        model,
                        batch["input_ids"],
                        batch["targets"],
                        fixed_candidate_ids=sampled_candidate_ids,
                        vocab_size=vocab_size,
                        token_stride=_full_loss_token_stride_for_variant(variant, config),
                        token_chunk_size=config.train_loss_token_chunk_size,
                    )
                    sampled_candidate_sizes.append(candidate_size)
                else:
                    logits = model(batch["input_ids"])
                    loss, token_count = _loss_and_tokens(logits, batch["targets"])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameter_list, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_duration = time.perf_counter() - step_start
        step_times.append(step_duration)
        tokens_seen += token_count
        if config.max_step_seconds is not None and step_duration > config.max_step_seconds:
            if checkpoint_path is not None and config.variant_checkpoint_interval > 0:
                _save_variant_checkpoint(
                    checkpoint_path,
                    variant=variant,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    tokens_seen=tokens_seen,
                    initial_val_loss=initial_val_loss,
                    history=history,
                    step_times=step_times,
                    sampled_candidate_sizes=sampled_candidate_sizes,
                    sampled_eval_candidate_sizes=sampled_eval_candidate_sizes,
                )
            if state_path is not None:
                _write_json_atomic(
                    state_path,
                    {
                        "status": "aborted_slow_step",
                        "variant": variant,
                        "step": step,
                        "train_steps": config.train_steps,
                        "tokens_seen": tokens_seen,
                        "latest_train_loss": float(loss.detach().item()),
                        "step_time_ms": step_duration * 1000.0,
                        "max_step_seconds": config.max_step_seconds,
                        "peak_vram_mb": _peak_vram_mb(device),
                        "latest_learning_rate": float(current_lr),
                        "lr_schedule": config.lr_schedule,
                    },
                )
            raise RuntimeError(
                f"{variant} step {step} took {step_duration:.3f}s, "
                f"exceeding max_step_seconds={config.max_step_seconds:.3f}."
            )
        if state_path is not None:
            pure_train_time = sum(step_times)
            _write_json_atomic(
                state_path,
                {
                    "status": "running",
                    "variant": variant,
                    "step": step,
                    "train_steps": config.train_steps,
                    "tokens_seen": tokens_seen,
                    "latest_train_loss": float(loss.detach().item()),
                    "latest_val_loss": float(history[-1]["val_loss"]) if history else float("nan"),
                    "latest_learning_rate": float(current_lr),
                    "lr_schedule": config.lr_schedule,
                    "pure_train_tok_per_sec": tokens_seen / max(pure_train_time, 1e-9),
                    "step_time_ms": step_duration * 1000.0,
                    "peak_vram_mb": _peak_vram_mb(device),
                },
            )
            if step == 1 or step == config.train_steps or step % max(config.train_log_interval, 1) == 0:
                print(
                    f"{variant} step={step}/{config.train_steps} "
                    f"train={float(loss.detach().item()):.4f} lr={current_lr:.3g} "
                    f"pure_tok/s={tokens_seen / max(pure_train_time, 1e-9):,.0f}",
                    flush=True,
                )
        if step % config.eval_interval == 0 or step == config.train_steps:
            if is_hierarchical:
                eval_start = time.perf_counter()
                val_loss = _evaluate_hierarchical_loss(
                    model,
                    val_source,
                    device=device,
                    autocast_kwargs=autocast_kwargs,
                    batch_size=config.eval_batch_size,
                )
            else:
                try:
                    if config.eval_loss_mode == "full":
                        val_loss, eval_duration = _evaluate_standard_loss(
                            model,
                            val_source,
                            device=device,
                            autocast_kwargs=autocast_kwargs,
                            batch_size=config.eval_batch_size,
                            token_chunk_size=config.full_eval_token_chunk_size,
                            max_seconds=config.max_eval_seconds,
                        )
                    elif config.eval_loss_mode == "sampled_vocab":
                        val_loss, eval_duration, eval_candidate_size = _evaluate_sampled_vocab_loss(
                            model,
                            val_source,
                            fixed_candidate_ids=sampled_candidate_ids,
                            vocab_size=vocab_size,
                            device=device,
                            autocast_kwargs=autocast_kwargs,
                            batch_size=config.eval_batch_size,
                            max_seconds=config.max_eval_seconds,
                        )
                        sampled_eval_candidate_sizes.append(eval_candidate_size)
                    else:
                        raise ValueError(f"Unknown eval_loss_mode: {config.eval_loss_mode}")
                except SlowEvalAbort as error:
                    if state_path is not None:
                        _write_json_atomic(
                            state_path,
                            {
                                "status": "aborted_slow_eval",
                                "variant": variant,
                                "step": step,
                                "train_steps": config.train_steps,
                                "tokens_seen": tokens_seen,
                                "latest_train_loss": float(loss.detach().item()),
                                "eval_time_ms": error.elapsed_seconds * 1000.0,
                                "max_eval_seconds": config.max_eval_seconds,
                                "peak_vram_mb": _peak_vram_mb(device),
                                "latest_learning_rate": float(current_lr),
                                "lr_schedule": config.lr_schedule,
                            },
                        )
                    raise RuntimeError(
                        f"{variant} evaluation at step {step} took {error.elapsed_seconds:.3f}s, "
                        f"exceeding max_eval_seconds={config.max_eval_seconds:.3f}."
                    ) from error
            history.append(
                {
                    "step": float(step),
                    "tokens_seen": float(tokens_seen),
                    "train_loss": float(loss.detach().item()),
                    "val_loss": float(val_loss),
                    "learning_rate": float(current_lr),
                }
            )
            if state_path is not None:
                _write_json_atomic(
                    state_path,
                    {
                        "status": "running",
                        "variant": variant,
                        "step": step,
                        "train_steps": config.train_steps,
                        "tokens_seen": tokens_seen,
                        "latest_train_loss": float(loss.detach().item()),
                        "latest_val_loss": float(val_loss),
                        "latest_learning_rate": float(current_lr),
                        "lr_schedule": config.lr_schedule,
                        "validation_loss_mode": config.eval_loss_mode,
                        "pure_train_tok_per_sec": tokens_seen / max(sum(step_times), 1e-9),
                        "peak_vram_mb": _peak_vram_mb(device),
                    },
                )
                print(f"{variant} eval step={step} val={float(val_loss):.4f} mode={config.eval_loss_mode}", flush=True)
        if checkpoint_path is not None and config.variant_checkpoint_interval > 0:
            should_save_checkpoint = (
                step % config.variant_checkpoint_interval == 0
                or (config.eval_interval > 0 and step % config.eval_interval == 0)
                or step == config.train_steps
            )
            if should_save_checkpoint:
                _save_variant_checkpoint(
                    checkpoint_path,
                    variant=variant,
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    tokens_seen=tokens_seen,
                    initial_val_loss=initial_val_loss,
                    history=history,
                    step_times=step_times,
                    sampled_candidate_sizes=sampled_candidate_sizes,
                    sampled_eval_candidate_sizes=sampled_eval_candidate_sizes,
                )
                if config.milestone_checkpoint_interval > 0 and step % config.milestone_checkpoint_interval == 0:
                    milestone_path = checkpoint_path.with_name(
                        f"{checkpoint_path.stem}.step{step}_tokens{tokens_seen}.pt"
                    )
                    _save_variant_checkpoint(
                        milestone_path,
                        variant=variant,
                        config=config,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=step,
                        tokens_seen=tokens_seen,
                        initial_val_loss=initial_val_loss,
                        history=history,
                        step_times=step_times,
                        sampled_candidate_sizes=sampled_candidate_sizes,
                        sampled_eval_candidate_sizes=sampled_eval_candidate_sizes,
                    )
    total_time = time.perf_counter() - start_time
    pure_train_time = sum(step_times)
    report = {
        "variant": variant,
        "parameter_count": parameter_count,
        "initial_val_loss": float(initial_val_loss),
        "final_val_loss": float(history[-1]["val_loss"]),
        "train_tokens_seen": tokens_seen,
        "train_tok_per_sec": tokens_seen / max(total_time, 1e-9),
        "pure_train_tok_per_sec": tokens_seen / max(pure_train_time, 1e-9),
        "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
        "step_time_median_ms": statistics.median(step_times) * 1000.0,
        "peak_vram_mb": _peak_vram_mb(device),
        "sampled_candidate_size_mean": statistics.fmean(sampled_candidate_sizes) if sampled_candidate_sizes else None,
        "sampled_eval_candidate_size_mean": statistics.fmean(sampled_eval_candidate_sizes) if sampled_eval_candidate_sizes else None,
        "validation_loss_mode": config.eval_loss_mode,
        "learning_rate": config.learning_rate,
        "lr_schedule": config.lr_schedule,
        "warmup_steps": config.warmup_steps,
        "min_learning_rate": config.min_learning_rate,
        "final_learning_rate": _scheduled_learning_rate(config, config.train_steps),
        "pure_train_time_seconds": pure_train_time,
        "total_training_time_seconds": total_time,
        "history": history,
    }
    if state_path is not None:
        _write_json_atomic(
            state_path,
            {
                "status": "completed",
                "variant": variant,
                "step": config.train_steps,
                "train_steps": config.train_steps,
                "tokens_seen": tokens_seen,
                "final_val_loss": report["final_val_loss"],
                "validation_loss_mode": report["validation_loss_mode"],
                "final_learning_rate": report["final_learning_rate"],
                "lr_schedule": report["lr_schedule"],
                "pure_train_tok_per_sec": report["pure_train_tok_per_sec"],
                "peak_vram_mb": report["peak_vram_mb"],
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None and checkpoint_path.exists() else None,
                "result_path": str(result_path) if result_path is not None else None,
            },
        )
    if result_path is not None:
        _write_json_atomic(
            result_path,
            {
                "benchmark": "language_longseq_replay_probe_variant",
                "variant": variant,
                "config": {
                    **asdict(config),
                    "output_dir": str(config.output_dir),
                    "cache_path": str(config.cache_path) if config.cache_path is not None else None,
                    "validation_cache_path": (
                        str(config.validation_cache_path) if config.validation_cache_path is not None else None
                    ),
                },
                "report": report,
            },
        )
    return report


def run_longseq_replay_probe(config: LongSeqReplayProbeConfig) -> dict[str, Any]:
    _configure_repo_local_hf_cache(config.output_dir)
    _enforce_gpu_preflight(config)
    if torch.cuda.is_available() and config.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    cache_config = _cache_config(config)
    train_dataset, val_dataset, vocab_size, resolved_cache_path = ensure_fineweb_cache(cache_config, print_progress=True)
    partial_token_ids = _top_token_ids(train_dataset, count=config.partial_token_count, vocab_size=vocab_size)
    sampled_candidate_ids = _top_token_ids(train_dataset, count=config.sampled_vocab_size, vocab_size=vocab_size)
    schedule = _build_train_batch_schedule(
        len(train_dataset),
        batch_size=config.batch_size,
        steps=config.train_steps,
        seed=config.seed,
        drop_last=True,
    )
    reports: dict[str, dict[str, Any]] = {}
    variant_artifact_dir = config.output_dir / "variant_results" / _run_slug(config)
    for variant in config.variants:
        if config.reuse_variant_results:
            reused_report = _load_reusable_variant_report(variant, config, variant_artifact_dir)
            if reused_report is not None:
                reports[variant] = reused_report
                print(f"reused {variant}: {variant_artifact_dir / f'{variant}.json'}", flush=True)
                continue
        reports[variant] = _train_variant(
            variant,
            train_dataset,
            val_dataset,
            vocab_size=vocab_size,
            partial_token_ids=partial_token_ids,
            sampled_candidate_ids=sampled_candidate_ids,
            batch_schedule=schedule,
            config=config,
            variant_artifact_dir=variant_artifact_dir,
        )
    comparisons = {}
    baseline = reports.get("partial_untied")
    if baseline is not None:
        for variant, report in reports.items():
            if variant == "partial_untied":
                continue
            comparisons[variant] = _comparison_vs_baseline(report, baseline)
    return {
        "benchmark": "language_longseq_replay_probe",
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "cache_path": str(config.cache_path) if config.cache_path is not None else None,
            "validation_cache_path": (
                str(config.validation_cache_path) if config.validation_cache_path is not None else None
            ),
            "resolved_cache_path": str(resolved_cache_path),
            "variant_artifact_dir": str(variant_artifact_dir),
        },
        "fairness": {
            "real_fineweb_edu_data": True,
            "same_cache": True,
            "same_batch_schedule": True,
            "same_token_budget": True,
            "same_model_size_target": "20m_parameter_partial_untied_family",
            "validation_loss_mode": config.eval_loss_mode,
            "reused_completed_variant_results": config.reuse_variant_results,
            "probe_only": True,
        },
        "results": reports,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe long-sequence partial_untied speed candidates on FineWeb-Edu.")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_CONFIG.output_dir)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--validation-cache-path", type=Path, default=None)
    parser.add_argument("--train-blocks", type=int, default=_DEFAULT_CONFIG.train_blocks)
    parser.add_argument("--val-blocks", type=int, default=_DEFAULT_CONFIG.val_blocks)
    parser.add_argument("--sequence-length", type=int, default=_DEFAULT_CONFIG.sequence_length)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_CONFIG.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=_DEFAULT_CONFIG.eval_batch_size)
    parser.add_argument("--train-steps", type=int, default=_DEFAULT_CONFIG.train_steps)
    parser.add_argument("--eval-interval", type=int, default=_DEFAULT_CONFIG.eval_interval)
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--eval-loss-mode", choices=("full", "sampled_vocab"), default=_DEFAULT_CONFIG.eval_loss_mode)
    parser.add_argument("--max-step-seconds", type=float, default=None)
    parser.add_argument("--max-eval-seconds", type=float, default=None)
    parser.add_argument("--reuse-variant-results", action="store_true")
    parser.add_argument("--max-gpu-used-mb", type=float, default=None)
    parser.add_argument("--variant-checkpoint-interval", type=int, default=_DEFAULT_CONFIG.variant_checkpoint_interval)
    parser.add_argument(
        "--milestone-checkpoint-interval",
        type=int,
        default=_DEFAULT_CONFIG.milestone_checkpoint_interval,
    )
    parser.add_argument("--resume-variant-checkpoints", action="store_true")
    parser.add_argument("--resume-fresh-cache", action="store_true")
    parser.add_argument("--train-log-interval", type=int, default=_DEFAULT_CONFIG.train_log_interval)
    parser.add_argument("--seed", type=int, default=_DEFAULT_CONFIG.seed)
    parser.add_argument("--train-token-offset", type=int, default=_DEFAULT_CONFIG.train_token_offset)
    parser.add_argument("--learning-rate", type=float, default=_DEFAULT_CONFIG.learning_rate)
    parser.add_argument("--lr-schedule", choices=("constant", "linear", "cosine"), default=_DEFAULT_CONFIG.lr_schedule)
    parser.add_argument("--warmup-steps", type=int, default=_DEFAULT_CONFIG.warmup_steps)
    parser.add_argument("--min-learning-rate", type=float, default=_DEFAULT_CONFIG.min_learning_rate)
    parser.add_argument("--weight-decay", type=float, default=_DEFAULT_CONFIG.weight_decay)
    parser.add_argument("--factorized-embedding-dim", type=int, default=_DEFAULT_CONFIG.factorized_embedding_dim)
    parser.add_argument("--factorized-hidden-dim", type=int, default=_DEFAULT_CONFIG.factorized_hidden_dim)
    parser.add_argument("--factorized-memory-dim", type=int, default=_DEFAULT_CONFIG.factorized_memory_dim)
    parser.add_argument("--no-replay-embedding-dim", type=int, default=_DEFAULT_CONFIG.no_replay_embedding_dim)
    parser.add_argument("--no-replay-hidden-dim", type=int, default=_DEFAULT_CONFIG.no_replay_hidden_dim)
    parser.add_argument("--no-replay-rank", type=int, default=_DEFAULT_CONFIG.no_replay_rank)
    parser.add_argument("--conv-embedding-dim", type=int, default=_DEFAULT_CONFIG.conv_embedding_dim)
    parser.add_argument("--conv-layers", type=int, default=_DEFAULT_CONFIG.conv_layers)
    parser.add_argument("--conv-rank", type=int, default=_DEFAULT_CONFIG.conv_rank)
    parser.add_argument("--conv-kernel-size", type=int, default=_DEFAULT_CONFIG.conv_kernel_size)
    parser.add_argument("--window-size", type=int, default=_DEFAULT_CONFIG.window_size)
    parser.add_argument("--untied-rank", type=int, default=_DEFAULT_CONFIG.untied_rank)
    parser.add_argument("--sampled-vocab-size", type=int, default=_DEFAULT_CONFIG.sampled_vocab_size)
    parser.add_argument("--full-loss-interval", type=int, default=_DEFAULT_CONFIG.full_loss_interval)
    parser.add_argument("--full-loss-token-stride", type=int, default=_DEFAULT_CONFIG.full_loss_token_stride)
    parser.add_argument("--full-eval-token-chunk-size", type=int, default=_DEFAULT_CONFIG.full_eval_token_chunk_size)
    parser.add_argument("--train-loss-token-chunk-size", type=int, default=_DEFAULT_CONFIG.train_loss_token_chunk_size)
    parser.add_argument("--hierarchical-class-count", type=int, default=_DEFAULT_CONFIG.hierarchical_class_count)
    parser.add_argument("--device", type=str, default=_DEFAULT_CONFIG.device)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-cache-dataset-on-device", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-compile-mode", type=str, default=_DEFAULT_CONFIG.torch_compile_mode)
    parser.add_argument("--tokenization-batch-size", type=int, default=_DEFAULT_CONFIG.tokenization_batch_size)
    parser.add_argument("--variants", nargs="+", default=list(_DEFAULT_VARIANTS))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = LongSeqReplayProbeConfig(
        output_dir=args.output_dir,
        cache_path=args.cache_path,
        validation_cache_path=args.validation_cache_path,
        train_blocks=args.train_blocks,
        val_blocks=args.val_blocks,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        initial_eval=not args.skip_initial_eval,
        eval_loss_mode=args.eval_loss_mode,
        max_step_seconds=args.max_step_seconds,
        max_eval_seconds=args.max_eval_seconds,
        reuse_variant_results=args.reuse_variant_results,
        max_gpu_used_mb=args.max_gpu_used_mb,
        variant_checkpoint_interval=args.variant_checkpoint_interval,
        milestone_checkpoint_interval=args.milestone_checkpoint_interval,
        resume_variant_checkpoints=args.resume_variant_checkpoints,
        resume_fresh_cache=args.resume_fresh_cache,
        train_log_interval=args.train_log_interval,
        seed=args.seed,
        train_token_offset=args.train_token_offset,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        factorized_embedding_dim=args.factorized_embedding_dim,
        factorized_hidden_dim=args.factorized_hidden_dim,
        factorized_memory_dim=args.factorized_memory_dim,
        no_replay_embedding_dim=args.no_replay_embedding_dim,
        no_replay_hidden_dim=args.no_replay_hidden_dim,
        no_replay_rank=args.no_replay_rank,
        conv_embedding_dim=args.conv_embedding_dim,
        conv_layers=args.conv_layers,
        conv_rank=args.conv_rank,
        conv_kernel_size=args.conv_kernel_size,
        window_size=args.window_size,
        untied_rank=args.untied_rank,
        sampled_vocab_size=args.sampled_vocab_size,
        full_loss_interval=args.full_loss_interval,
        full_loss_token_stride=args.full_loss_token_stride,
        full_eval_token_chunk_size=args.full_eval_token_chunk_size,
        train_loss_token_chunk_size=args.train_loss_token_chunk_size,
        hierarchical_class_count=args.hierarchical_class_count,
        device=args.device,
        local_files_only=args.local_files_only,
        pin_memory=not args.no_pin_memory,
        cache_dataset_on_device=not args.no_cache_dataset_on_device,
        torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
        tokenization_batch_size=args.tokenization_batch_size,
        variants=tuple(args.variants),
    )
    payload = run_longseq_replay_probe(config)
    text = json.dumps(payload, indent=2, sort_keys=True)
    output_path = args.output if args.output is not None else config.output_dir / "longseq_replay_probe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
