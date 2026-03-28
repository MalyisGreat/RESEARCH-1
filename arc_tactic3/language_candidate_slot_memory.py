from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from arc_tactic3.language_fastlearn_benchmark import set_global_seed
from arc_tactic3.language_nanochat_actual_compare import _load_cached_datasets, _shared_realtext_config, _train_candidate
from arc_tactic3.language_realtext_microbench import _build_train_batch_schedule
from arc_tactic3.language_recurrent_nano_tricks import PartialUntiedAssociativeLM, RecurrentNanoTricksConfig, _top_token_ids


@dataclass(frozen=True, slots=True)
class SlotMemoryCompareConfig:
    cache_path: Path
    train_blocks: int = 1024
    val_blocks: int = 64
    sequence_length: int = 127
    train_steps: int = 16
    eval_interval: int = 8
    seed: int = 13
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_val_bpb: bool = False
    partial_untied_tokens: int = 1024
    recurrent_embedding_dim: int = 144
    recurrent_hidden_dim: int = 288
    recurrent_memory_dim: int = 144
    slot_count: int = 8
    slot_dim: int = 144
    use_amp: bool = torch.cuda.is_available()
    pin_memory: bool = torch.cuda.is_available()
    use_fused_adamw: bool = torch.cuda.is_available()
    tensor_batching: bool = False
    cache_dataset_on_device: bool = False


class SlotMemoryRecurrentLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        slot_count: int,
        slot_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head_fc = nn.Linear(hidden_dim, 4 * embedding_dim)
        self.head_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

        self.slot_init = nn.Parameter(torch.randn(slot_count, slot_dim) * 0.02)
        self.read_query = nn.Linear(hidden_dim, slot_dim)
        self.write_query = nn.Linear(hidden_dim, slot_dim)
        self.write_value = nn.Linear(hidden_dim, slot_dim)
        self.write_gate = nn.Linear(hidden_dim, slot_count)
        self.read_mix = nn.Linear(hidden_dim, embedding_dim)
        self.slot_to_feature = nn.Linear(slot_dim, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        states, _ = self.encoder(embeddings)
        states = self.dropout(states)

        batch_size, seq_len, _ = states.shape
        slots = self.slot_init.unsqueeze(0).expand(batch_size, -1, -1).clone()
        retrieved_features: list[torch.Tensor] = []

        for step in range(seq_len):
            state_t = states[:, step]

            read_scores = torch.matmul(self.read_query(state_t).unsqueeze(1), slots.transpose(1, 2)).squeeze(1)
            read_attn = torch.softmax(read_scores, dim=-1)
            read_vec = torch.einsum("bs,bsd->bd", read_attn, slots)
            retrieved_features.append(self.slot_to_feature(read_vec))

            write_scores = torch.matmul(self.write_query(state_t).unsqueeze(1), slots.transpose(1, 2)).squeeze(1)
            write_attn = torch.softmax(write_scores, dim=-1)
            write_gate = torch.sigmoid(self.write_gate(state_t))
            write_strength = (write_attn * write_gate).unsqueeze(-1)
            candidate = torch.tanh(self.write_value(state_t)).unsqueeze(1)
            slots = slots * (1.0 - write_strength) + candidate * write_strength

        retrieved = torch.stack(retrieved_features, dim=1)
        head_features = F.relu(self.head_fc(states)).square()
        base_features = self.head_proj(head_features)
        mixed_features = base_features + torch.sigmoid(self.read_mix(states)) * retrieved
        return F.linear(mixed_features, self.embedding.weight, self.output_bias)


def _build_models(
    config: RecurrentNanoTricksConfig,
    *,
    vocab_size: int,
    partial_token_ids: torch.Tensor,
    slot_count: int,
    slot_dim: int,
) -> dict[str, nn.Module]:
    common = {
        "vocab_size": vocab_size,
        "embedding_dim": config.recurrent_embedding_dim,
        "hidden_dim": config.recurrent_hidden_dim,
        "dropout": config.dropout,
    }
    return {
        "partial_untied": PartialUntiedAssociativeLM(
            vocab_size=vocab_size,
            embedding_dim=config.recurrent_embedding_dim,
            hidden_dim=config.recurrent_hidden_dim,
            memory_dim=config.recurrent_memory_dim,
            dropout=config.dropout,
            max_length=config.sequence_length,
            untied_token_ids=partial_token_ids,
        ),
        "slot_memory_candidate": SlotMemoryRecurrentLM(
            **common,
            slot_count=slot_count,
            slot_dim=slot_dim,
        ),
    }


def run_slot_memory_compare(config: SlotMemoryCompareConfig) -> dict[str, Any]:
    base_cfg = RecurrentNanoTricksConfig(
        cache_path=config.cache_path,
        train_blocks=config.train_blocks,
        val_blocks=config.val_blocks,
        sequence_length=config.sequence_length,
        train_steps=config.train_steps,
        eval_interval=config.eval_interval,
        seed=config.seed,
        device=config.device,
        compute_val_bpb=config.compute_val_bpb,
        partial_untied_tokens=config.partial_untied_tokens,
        recurrent_embedding_dim=config.recurrent_embedding_dim,
        recurrent_hidden_dim=config.recurrent_hidden_dim,
        recurrent_memory_dim=config.recurrent_memory_dim,
        window_size=config.slot_count,
        use_amp=config.use_amp,
        pin_memory=config.pin_memory,
        use_fused_adamw=config.use_fused_adamw,
        tensor_batching=config.tensor_batching,
        cache_dataset_on_device=config.cache_dataset_on_device,
    )
    set_global_seed(config.seed)
    train_dataset, val_dataset, vocab_size = _load_cached_datasets(base_cfg)
    partial_token_ids = _top_token_ids(train_dataset, count=base_cfg.partial_untied_tokens, vocab_size=vocab_size)
    shared_cfg = _shared_realtext_config(base_cfg)
    batch_schedule = _build_train_batch_schedule(
        len(train_dataset),
        batch_size=shared_cfg.batch_size,
        steps=shared_cfg.train_steps,
        seed=shared_cfg.seed if shared_cfg.train_schedule_seed is None else shared_cfg.train_schedule_seed,
        drop_last=True,
    )
    models = _build_models(
        base_cfg,
        vocab_size=vocab_size,
        partial_token_ids=partial_token_ids,
        slot_count=config.slot_count,
        slot_dim=config.slot_dim,
    )
    results: dict[str, dict[str, Any]] = {}
    total = len(models)
    for index, (model_name, model) in enumerate(models.items(), start=1):
        set_global_seed(config.seed)
        report = _train_candidate(
            model,
            train_dataset,
            val_dataset,
            model_name=model_name,
            tokenizer=None,
            config=shared_cfg,
            compute_val_bpb=config.compute_val_bpb,
            batch_schedule=batch_schedule,
        )
        results[model_name] = report
        print(
            json.dumps(
                {
                    "progress": f"{index}/{total}",
                    "model": model_name,
                    "final_val_loss": report["final_val_loss"],
                    "train_tok_per_sec": report["train_tok_per_sec"],
                    "pure_train_tok_per_sec": report["pure_train_tok_per_sec"],
                    "peak_vram_mb": report["peak_vram_mb"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    baseline = results["partial_untied"]
    candidate = results["slot_memory_candidate"]
    summary = {
        "partial_untied": {
            "final_val_loss": baseline["final_val_loss"],
            "train_tok_per_sec": baseline["train_tok_per_sec"],
            "pure_train_tok_per_sec": baseline["pure_train_tok_per_sec"],
            "peak_vram_mb": baseline["peak_vram_mb"],
            "parameter_count": baseline["parameter_count"],
        },
        "slot_memory_candidate": {
            "final_val_loss": candidate["final_val_loss"],
            "train_tok_per_sec": candidate["train_tok_per_sec"],
            "pure_train_tok_per_sec": candidate["pure_train_tok_per_sec"],
            "peak_vram_mb": candidate["peak_vram_mb"],
            "parameter_count": candidate["parameter_count"],
        },
        "delta_vs_partial_untied": {
            "loss_improvement_percent": 100.0 * (baseline["final_val_loss"] - candidate["final_val_loss"]) / baseline["final_val_loss"],
            "train_tok_per_sec_improvement_percent": 100.0
            * (candidate["train_tok_per_sec"] - baseline["train_tok_per_sec"])
            / baseline["train_tok_per_sec"],
            "pure_train_tok_per_sec_improvement_percent": 100.0
            * (candidate["pure_train_tok_per_sec"] - baseline["pure_train_tok_per_sec"])
            / baseline["pure_train_tok_per_sec"],
        },
    }
    return {
        "benchmark": "language_candidate_slot_memory",
        "config": {
            "cache_path": str(config.cache_path),
            "train_blocks": config.train_blocks,
            "val_blocks": config.val_blocks,
            "sequence_length": config.sequence_length,
            "train_steps": config.train_steps,
            "eval_interval": config.eval_interval,
            "seed": config.seed,
            "device": config.device,
            "compute_val_bpb": config.compute_val_bpb,
            "partial_untied_tokens": config.partial_untied_tokens,
            "recurrent_embedding_dim": config.recurrent_embedding_dim,
            "recurrent_hidden_dim": config.recurrent_hidden_dim,
            "recurrent_memory_dim": config.recurrent_memory_dim,
            "slot_count": config.slot_count,
            "slot_dim": config.slot_dim,
            "use_amp": config.use_amp,
            "pin_memory": config.pin_memory,
            "use_fused_adamw": config.use_fused_adamw,
            "tensor_batching": config.tensor_batching,
            "cache_dataset_on_device": config.cache_dataset_on_device,
        },
        "results": results,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cheap compare runner for the slot-memory candidate.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--train-blocks", type=int, default=1024)
    parser.add_argument("--val-blocks", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=127)
    parser.add_argument("--train-steps", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recurrent-embedding-dim", type=int, default=144)
    parser.add_argument("--recurrent-hidden-dim", type=int, default=288)
    parser.add_argument("--recurrent-memory-dim", type=int, default=144)
    parser.add_argument("--partial-untied-tokens", type=int, default=1024)
    parser.add_argument("--slot-count", type=int, default=8)
    parser.add_argument("--slot-dim", type=int, default=144)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-fused-adamw", action="store_true")
    parser.add_argument("--tensor-batching", action="store_true")
    parser.add_argument("--cache-on-device", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = run_slot_memory_compare(
        SlotMemoryCompareConfig(
            cache_path=args.cache_path,
            train_blocks=args.train_blocks,
            val_blocks=args.val_blocks,
            sequence_length=args.sequence_length,
            train_steps=args.train_steps,
            eval_interval=args.eval_interval,
            seed=args.seed,
            device=args.device,
            recurrent_embedding_dim=args.recurrent_embedding_dim,
            recurrent_hidden_dim=args.recurrent_hidden_dim,
            recurrent_memory_dim=args.recurrent_memory_dim,
            partial_untied_tokens=args.partial_untied_tokens,
            slot_count=args.slot_count,
            slot_dim=args.slot_dim,
            use_amp=not args.no_amp,
            pin_memory=not args.no_pin_memory,
            use_fused_adamw=not args.no_fused_adamw,
            tensor_batching=args.tensor_batching,
            cache_dataset_on_device=args.cache_on_device,
        )
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
