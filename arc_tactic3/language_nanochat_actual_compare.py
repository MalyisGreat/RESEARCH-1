from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from arc_tactic3.language_fastlearn_benchmark import count_parameters, set_global_seed
from arc_tactic3.language_realtext_microbench import (
    RealTextConfig,
    TokenBlockDataset,
    _build_optimizer,
    _build_scheduler,
    _build_train_batch_schedule,
    _dataset_tensors,
    _iter_tensor_batches,
    _loss_and_tokens,
    _move_batch,
    _scheduled_batch_from_tensors,
    AssociativeRecurrentLM,
)
from arc_tactic3.language_throughput_candidates import GRUOnlyLM, WindowedAssociativeLM


@dataclass(frozen=True, slots=True)
class NanochatActualCompareConfig:
    cache_path: Path
    tokenizer_name: str = "gpt2"
    train_blocks: int = 8192
    val_blocks: int = 512
    sequence_length: int = 127
    batch_size: int = 16
    eval_batch_size: int = 32
    train_steps: int = 384
    eval_interval: int = 96
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 13
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = torch.cuda.is_available()
    pin_memory: bool = torch.cuda.is_available()
    use_fused_adamw: bool = torch.cuda.is_available()
    tensor_batching: bool = False
    cache_dataset_on_device: bool = False
    compute_val_bpb: bool = True
    recurrent_embedding_dim: int = 144
    recurrent_hidden_dim: int = 288
    recurrent_memory_dim: int = 144
    dropout: float = 0.1
    window_size: int = 32
    paired_train_batches: bool = True
    reseed_per_model: bool = True
    train_schedule_seed: int | None = None
    optimizer_recipe: str = "default"
    warmup_steps: int = 0
    lr_schedule: str = "none"
    min_lr_scale: float = 1.0
    nano_n_layer: int = 4
    nano_n_head: int = 4
    nano_n_kv_head: int = 4
    nano_n_embd: int = 40
    nano_window_pattern: str = "SSSL"
    nano_softcap: float = 15.0
    nano_use_value_embeddings: bool = True
    nano_use_smear: bool = True
    nano_use_backout: bool = True


def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_dim = x.size(-1)
    half_dim = head_dim // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)


def _repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    batch_size, sequence_length, n_heads, head_dim = x.shape
    x = x.unsqueeze(3).expand(batch_size, sequence_length, n_heads, repeats, head_dim)
    return x.reshape(batch_size, sequence_length, n_heads * repeats, head_dim)


class _NanoLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)


def _nano_has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


class NanochatMiniAttention(nn.Module):
    def __init__(self, *, n_embd: int, n_head: int, n_kv_head: int, n_layer: int, layer_idx: int) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must divide evenly by n_head.")
        if n_head % n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head.")
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even to use rotary embeddings.")
        self.repeat_factor = n_head // n_kv_head
        self.c_q = _NanoLinear(n_embd, n_head * self.head_dim)
        self.c_k = _NanoLinear(n_embd, n_kv_head * self.head_dim)
        self.c_v = _NanoLinear(n_embd, n_kv_head * self.head_dim)
        self.c_proj = _NanoLinear(n_embd, n_embd)
        self.ve_gate_channels = min(12, n_embd)
        self.ve_gate = _NanoLinear(self.ve_gate_channels, n_kv_head) if _nano_has_ve(layer_idx, n_layer) else None

    def forward(
        self,
        x: torch.Tensor,
        *,
        ve: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        q = self.c_q(x).view(batch_size, sequence_length, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch_size, sequence_length, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(batch_size, sequence_length, self.n_kv_head, self.head_dim)

        if ve is not None and self.ve_gate is not None:
            ve = ve.view(batch_size, sequence_length, self.n_kv_head, self.head_dim)
            gate = 3.0 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)
        q = _rms_norm(q) * 1.2
        k = _rms_norm(k) * 1.2
        if self.repeat_factor != 1:
            k = _repeat_kv(k, self.repeat_factor)
            v = _repeat_kv(v, self.repeat_factor)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        y = torch.matmul(attention, v)
        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        return self.c_proj(y)


class NanochatMiniMLP(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.c_fc = _NanoLinear(n_embd, 4 * n_embd)
        self.c_proj = _NanoLinear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class NanochatMiniBlock(nn.Module):
    def __init__(self, *, n_embd: int, n_head: int, n_kv_head: int, n_layer: int, layer_idx: int) -> None:
        super().__init__()
        self.attn = NanochatMiniAttention(
            n_embd=n_embd,
            n_head=n_head,
            n_kv_head=n_kv_head,
            n_layer=n_layer,
            layer_idx=layer_idx,
        )
        self.mlp = NanochatMiniMLP(n_embd)

    def forward(
        self,
        x: torch.Tensor,
        *,
        ve: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(_rms_norm(x), ve=ve, cos=cos, sin=sin, attn_mask=attn_mask)
        x = x + self.mlp(_rms_norm(x))
        return x


class NanochatMiniLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        sequence_length: int,
        n_layer: int,
        n_head: int,
        n_kv_head: int,
        n_embd: int,
        window_pattern: str,
        softcap: float,
        use_value_embeddings: bool,
        use_smear: bool,
        use_backout: bool,
        pad_vocab_size_to: int = 64,
    ) -> None:
        super().__init__()
        padded_vocab_size = ((vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.sequence_length = sequence_length
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.window_sizes = self._compute_window_sizes(window_pattern, sequence_length, n_layer)
        self.softcap = softcap
        self.use_value_embeddings = use_value_embeddings
        self.use_smear = use_smear
        self.use_backout = use_backout

        self.wte = nn.Embedding(padded_vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            NanochatMiniBlock(
                n_embd=n_embd,
                n_head=n_head,
                n_kv_head=n_kv_head,
                n_layer=n_layer,
                layer_idx=layer_idx,
            )
            for layer_idx in range(n_layer)
        )
        self.lm_head = _NanoLinear(n_embd, padded_vocab_size)
        self.resid_lambdas = nn.Parameter(torch.ones(n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(n_layer))
        self.smear_gate_channels = min(24, n_embd)
        self.smear_gate = _NanoLinear(self.smear_gate_channels, 1)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        self.value_embeds = nn.ModuleDict(
            {
                str(layer_idx): nn.Embedding(padded_vocab_size, n_kv_head * self.head_dim)
                for layer_idx in range(n_layer)
                if _nano_has_ve(layer_idx, n_layer)
            }
        )

        cos, sin = self._precompute_rotary_embeddings(sequence_length, self.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._init_weights()

    def _compute_window_sizes(self, pattern: str, sequence_length: int, n_layer: int) -> list[tuple[int, int]]:
        pattern = pattern.upper()
        if not pattern or any(char not in "SL" for char in pattern):
            raise ValueError("window_pattern must use only 'S' and 'L'.")
        long_window = sequence_length
        short_window = max(16, math.ceil(long_window / 4))
        table = {"L": (long_window, 0), "S": (short_window, 0)}
        windows = [table[pattern[layer_idx % len(pattern)]] for layer_idx in range(n_layer)]
        windows[-1] = (long_window, 0)
        return windows

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int = 100_000) -> tuple[torch.Tensor, torch.Tensor]:
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]
        return cos, sin

    def _init_weights(self) -> None:
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        scale = math.sqrt(3.0) * self.n_embd ** -0.5
        for block in self.blocks:
            torch.nn.init.uniform_(block.attn.c_q.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_k.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_v.weight, -scale, scale)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -scale * 0.4, scale * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        for index in range(self.n_layer):
            self.resid_lambdas.data[index] = 1.15 - (0.10 * index / max(self.n_layer - 1, 1))
            self.x0_lambdas.data[index] = 0.20 - (0.15 * index / max(self.n_layer - 1, 1))
        for value_embed in self.value_embeds.values():
            torch.nn.init.uniform_(value_embed.weight, -scale, scale)

    def _attn_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        row = torch.arange(sequence_length, device=device)
        col = torch.arange(sequence_length, device=device)
        diff = row.view(1, 1, sequence_length, 1) - col.view(1, 1, 1, sequence_length)
        layer_masks: list[torch.Tensor] = []
        for window_left, _ in self.window_sizes:
            layer_mask = (diff >= 0) & (diff < max(window_left, 1))
            layer_masks.append(layer_mask)
        return torch.cat(layer_masks, dim=0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.sequence_length:
            raise ValueError(f"input length {sequence_length} exceeds configured sequence length {self.sequence_length}")
        x = _rms_norm(self.wte(input_ids))

        if self.use_smear and sequence_length > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, : self.smear_gate_channels]))
            x = torch.cat((x[:, :1], x[:, 1:] + gate * x[:, :-1]), dim=1)

        x0 = x
        x_backout = None
        cos = self.cos[:, :sequence_length].to(dtype=x.dtype, device=x.device)
        sin = self.sin[:, :sequence_length].to(dtype=x.dtype, device=x.device)
        layer_masks = self._attn_mask(sequence_length, x.device)
        backout_layer = self.n_layer // 2

        for layer_idx, block in enumerate(self.blocks):
            x = self.resid_lambdas[layer_idx].to(x.dtype) * x + self.x0_lambdas[layer_idx].to(x.dtype) * x0
            ve = None
            if self.use_value_embeddings and str(layer_idx) in self.value_embeds:
                ve = self.value_embeds[str(layer_idx)](input_ids).to(dtype=x.dtype)
            x = block(
                x,
                ve=ve,
                cos=cos,
                sin=sin,
                attn_mask=layer_masks[layer_idx : layer_idx + 1],
            )
            if self.use_backout and layer_idx == backout_layer:
                x_backout = x
        if self.use_backout and x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = _rms_norm(x)
        logits = self.lm_head(x)[..., : self.vocab_size].float()
        return self.softcap * torch.tanh(logits / self.softcap)


class ReLU2HeadAssociativeLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        memory_dim: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(hidden_dim, memory_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.memory_scale = nn.Parameter(torch.tensor(6.0))
        self.register_buffer(
            "_causal_mask",
            torch.tril(torch.ones((max_length, max_length), dtype=torch.bool), diagonal=-1).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)
        head_features = self.head_fc(states)
        head_features = F.relu(head_features).square()
        base_features = self.head_proj(head_features)
        base_logits = F.linear(base_features, self.embedding.weight, self.output_bias)
        query_keys = self.query_proj(states)
        memory_keys = self.key_proj(states)
        scores = torch.matmul(query_keys, memory_keys.transpose(1, 2)) / query_keys.size(-1) ** 0.5
        causal_mask = self._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        value_index = input_ids.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        gate = torch.sigmoid(self.gate(states))
        gated_attention = (attention * (gate * self.memory_scale)).to(base_logits.dtype)
        base_logits.scatter_add_(2, value_index, gated_attention)
        return base_logits


def _load_cached_datasets(config: NanochatActualCompareConfig) -> tuple[TokenBlockDataset, TokenBlockDataset, int]:
    payload = torch.load(config.cache_path, map_location="cpu", weights_only=False)
    block_size = config.sequence_length + 1
    train_blocks = payload["train_tokens"].long().view(-1, block_size)
    val_blocks = payload["val_tokens"].long().view(-1, block_size)
    train_dataset = TokenBlockDataset(
        train_blocks[: config.train_blocks, :-1].contiguous(),
        train_blocks[: config.train_blocks, 1:].contiguous(),
    )
    val_dataset = TokenBlockDataset(
        val_blocks[: config.val_blocks, :-1].contiguous(),
        val_blocks[: config.val_blocks, 1:].contiguous(),
    )
    return train_dataset, val_dataset, int(payload["vocab_size"])


def _shared_realtext_config(config: NanochatActualCompareConfig) -> RealTextConfig:
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
        tensor_batching=config.tensor_batching,
        cache_dataset_on_device=config.cache_dataset_on_device,
        paired_train_batches=config.paired_train_batches,
        reseed_per_model=config.reseed_per_model,
        train_schedule_seed=config.train_schedule_seed,
        optimizer_recipe=config.optimizer_recipe,
        warmup_steps=config.warmup_steps,
        lr_schedule=config.lr_schedule,
        min_lr_scale=config.min_lr_scale,
    )


def _build_models(config: NanochatActualCompareConfig, *, vocab_size: int) -> dict[str, nn.Module]:
    return {
        "recurrent_baseline": AssociativeRecurrentLM(
            vocab_size=vocab_size,
            embedding_dim=config.recurrent_embedding_dim,
            hidden_dim=config.recurrent_hidden_dim,
            memory_dim=config.recurrent_memory_dim,
            dropout=config.dropout,
            max_length=config.sequence_length,
        ),
        "recurrent_champion": ReLU2HeadAssociativeLM(
            vocab_size=vocab_size,
            embedding_dim=config.recurrent_embedding_dim,
            hidden_dim=config.recurrent_hidden_dim,
            memory_dim=config.recurrent_memory_dim,
            dropout=config.dropout,
            max_length=config.sequence_length,
        ),
        "gru_only": GRUOnlyLM(
            vocab_size=vocab_size,
            embedding_dim=config.recurrent_embedding_dim,
            hidden_dim=config.recurrent_hidden_dim,
            dropout=config.dropout,
        ),
        "windowed_32": WindowedAssociativeLM(
            vocab_size=vocab_size,
            embedding_dim=config.recurrent_embedding_dim,
            hidden_dim=config.recurrent_hidden_dim,
            memory_dim=config.recurrent_memory_dim,
            dropout=config.dropout,
            max_length=config.sequence_length,
            window_size=config.window_size,
        ),
        "nanochat_small": NanochatMiniLM(
            vocab_size=vocab_size,
            sequence_length=config.sequence_length,
            n_layer=config.nano_n_layer,
            n_head=config.nano_n_head,
            n_kv_head=config.nano_n_kv_head,
            n_embd=config.nano_n_embd,
            window_pattern=config.nano_window_pattern,
            softcap=config.nano_softcap,
            use_value_embeddings=config.nano_use_value_embeddings,
            use_smear=config.nano_use_smear,
            use_backout=config.nano_use_backout,
        ),
    }


def _estimate_target_bytes(dataset: TokenBlockDataset, *, tokenizer) -> int:
    total_bytes = 0
    for row in dataset.targets:
        text = tokenizer.decode(row.tolist(), clean_up_tokenization_spaces=False)
        total_bytes += len(text.encode("utf-8"))
    return max(total_bytes, 1)


def _peak_vram_mb(device: str) -> float | None:
    if device != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _evaluate_candidate(
    model: nn.Module,
    eval_source,
    *,
    device: torch.device,
    use_amp: bool,
    config: RealTextConfig,
) -> float:
    model.eval()
    loss_sum = 0.0
    token_total = 0
    batch_iterator = eval_source
    if config.tensor_batching:
        input_ids, targets = eval_source
        batch_iterator = _iter_tensor_batches(
            input_ids,
            targets,
            batch_size=config.eval_batch_size,
            shuffle=False,
            drop_last=False,
            device=device,
            non_blocking=config.pin_memory and device.type == "cuda",
        )
    with torch.inference_mode():
        for batch in batch_iterator:
            if not config.tensor_batching:
                batch = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch["input_ids"])
                loss, tokens = _loss_and_tokens(logits, batch["targets"])
            loss_sum += float(loss.item()) * tokens
            token_total += tokens
    return loss_sum / max(token_total, 1)


def _train_candidate(
    model: nn.Module,
    train_dataset: TokenBlockDataset,
    val_dataset: TokenBlockDataset,
    *,
    model_name: str,
    tokenizer,
    config: RealTextConfig,
    compute_val_bpb: bool,
    batch_schedule: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    device = torch.device(config.device)
    model.to(device)
    optimizer = _build_optimizer(model, config, model_name=model_name)
    scheduler = _build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler(device="cuda", enabled=config.use_amp and device.type == "cuda")
    use_amp = config.use_amp and device.type == "cuda"
    parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]
    train_source = _dataset_tensors(
        train_dataset,
        device=device,
        cache_on_device=config.cache_dataset_on_device,
        pin_memory=config.pin_memory,
    )
    if config.tensor_batching:
        val_source = _dataset_tensors(
            val_dataset,
            device=device,
            cache_on_device=config.cache_dataset_on_device,
            pin_memory=config.pin_memory,
        )
    else:
        pin_memory = config.pin_memory and device.type == "cuda"
        val_source = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=0,
            persistent_workers=False,
        )

    if batch_schedule is None:
        batch_schedule = _build_train_batch_schedule(
            len(train_dataset),
            batch_size=config.batch_size,
            steps=config.train_steps,
            seed=config.seed if config.train_schedule_seed is None else config.train_schedule_seed,
            drop_last=True,
        )

    initial_val_loss = _evaluate_candidate(model, val_source, device=device, use_amp=use_amp, config=config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history: list[dict[str, float]] = [
        {
            "step": 0.0,
            "sequences_seen": 0.0,
            "tokens_seen": 0.0,
            "train_loss": float("nan"),
            "val_loss": float(initial_val_loss),
        }
    ]
    step_times: list[float] = []
    tokens_seen = 0
    sequences_seen = 0
    start = time.perf_counter()
    for step, batch_indices in enumerate(batch_schedule, start=1):
        batch = _scheduled_batch_from_tensors(
            train_source[0],
            train_source[1],
            batch_indices,
            device=device,
            non_blocking=config.pin_memory and device.type == "cuda",
        )
        step_start = time.perf_counter()
        model.train()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch["input_ids"])
            loss, token_count = _loss_and_tokens(logits, batch["targets"])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameter_list, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - step_start)
        tokens_seen += token_count
        sequences_seen += batch["input_ids"].size(0)
        if step % config.eval_interval == 0 or step == config.train_steps:
            val_loss = _evaluate_candidate(model, val_source, device=device, use_amp=use_amp, config=config)
            history.append(
                {
                    "step": float(step),
                    "sequences_seen": float(sequences_seen),
                    "tokens_seen": float(tokens_seen),
                    "train_loss": float(loss.item()),
                    "val_loss": float(val_loss),
                }
            )

    total_time = time.perf_counter() - start
    pure_train_time = sum(step_times)
    final_val_loss = float(history[-1]["val_loss"])
    target_bytes = _estimate_target_bytes(val_dataset, tokenizer=tokenizer) if compute_val_bpb and tokenizer is not None else None
    total_val_tokens = int(val_dataset.targets.numel())
    total_val_bits = final_val_loss * total_val_tokens / math.log(2.0)
    return {
        "parameter_count": count_parameters(model),
        "initial_val_loss": float(initial_val_loss),
        "final_val_loss": final_val_loss,
        "val_bits_per_token": final_val_loss / math.log(2.0),
        "val_bpb": (total_val_bits / target_bytes) if target_bytes is not None else None,
        "train_tokens_seen": tokens_seen,
        "train_tok_per_sec": tokens_seen / max(total_time, 1e-9),
        "pure_train_tok_per_sec": tokens_seen / max(pure_train_time, 1e-9),
        "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
        "step_time_median_ms": statistics.median(step_times) * 1000.0,
        "pure_train_time_seconds": pure_train_time,
        "eval_overhead_seconds": max(total_time - pure_train_time, 0.0),
        "peak_vram_mb": _peak_vram_mb(config.device),
        "paired_train_batches": bool(config.paired_train_batches),
        "reseed_per_model": bool(config.reseed_per_model),
        "history": history,
        "total_training_time_seconds": total_time,
    }


def run_nanochat_actual_compare(config: NanochatActualCompareConfig) -> dict[str, Any]:
    set_global_seed(config.seed)
    train_dataset, val_dataset, vocab_size = _load_cached_datasets(config)
    tokenizer = None
    if config.compute_val_bpb:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True, local_files_only=True)
    shared_config = _shared_realtext_config(config)
    schedule_seed = config.seed if config.train_schedule_seed is None else config.train_schedule_seed
    batch_schedule = _build_train_batch_schedule(
        len(train_dataset),
        batch_size=shared_config.batch_size,
        steps=shared_config.train_steps,
        seed=schedule_seed,
        drop_last=True,
    )
    reports: dict[str, dict[str, Any]] = {}
    for model_name, model in _build_models(config, vocab_size=vocab_size).items():
        if config.reseed_per_model:
            set_global_seed(config.seed)
        reports[model_name] = _train_candidate(
            model,
            train_dataset,
            val_dataset,
            model_name=model_name,
            tokenizer=tokenizer,
            config=shared_config,
            compute_val_bpb=config.compute_val_bpb,
            batch_schedule=batch_schedule,
        )
    return {
        "benchmark": "language_nanochat_actual_compare",
        "config": {
            **asdict(config),
            "cache_path": str(config.cache_path),
        },
        "fairness": {
            "same_dataset": True,
            "same_tokenizer": True,
            "paired_batch_schedule": True,
            "epoch_reshuffle": True,
            "reseed_per_model": bool(config.reseed_per_model),
        },
        "results": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare recurrent candidates against a scaled-down Nanochat architecture.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--train-blocks", type=int, default=8192)
    parser.add_argument("--val-blocks", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=384)
    parser.add_argument("--eval-interval", type=int, default=96)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--optimizer-recipe", type=str, default="default", choices=("default", "transformer_fair"))
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-schedule", type=str, default="none", choices=("none", "linear", "cosine"))
    parser.add_argument("--min-lr-scale", type=float, default=1.0)
    parser.add_argument("--nano-window-pattern", type=str, default="SSSL")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = NanochatActualCompareConfig(
        cache_path=args.cache_path,
        train_blocks=args.train_blocks,
        val_blocks=args.val_blocks,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        device=args.device,
        optimizer_recipe=args.optimizer_recipe,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        min_lr_scale=args.min_lr_scale,
        nano_window_pattern=args.nano_window_pattern,
    )
    payload = run_nanochat_actual_compare(config)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
