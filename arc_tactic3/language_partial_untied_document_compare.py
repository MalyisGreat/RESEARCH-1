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
from transformers import AutoTokenizer

from arc_tactic3.language_fastlearn_benchmark import count_parameters
from arc_tactic3.language_nanochat_actual_compare import _peak_vram_mb
from arc_tactic3.language_partial_untied_streaming_compare import StreamingPartialUntiedAssociativeLM
from arc_tactic3.language_realtext_microbench import RealTextConfig, _build_optimizer, _build_scheduler, _loss_and_tokens, set_global_seed
from arc_tactic3.language_recurrent_nano_tricks import _top_token_ids


@dataclass(frozen=True, slots=True)
class DocumentCompareConfig:
    cache_path: Path
    output_path: Path
    target_tokens: int = 5_000_000
    cheap_target_tokens: int = 1_000_000
    seed: int = 13
    sequence_length: int = 127
    batch_size: int = 16
    eval_batch_size: int = 16
    eval_steps: int = 64
    eval_interval: int = 256
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = torch.cuda.is_available()
    use_fused_adamw: bool = torch.cuda.is_available()
    embedding_dim: int = 144
    hidden_dim: int = 288
    memory_dim: int = 144
    partial_token_count: int = 512
    dropout: float = 0.1
    tokenizer_name: str = "gpt2"
    sample_prompt: str | None = None
    sample_every_tokens: int = 10_000_000
    sample_generation_tokens: int = 40
    variant_mode: str = "both"
    run_cheap: bool = True
    run_hold: bool = True


class DocumentStreamBatcher:
    def __init__(
        self,
        *,
        tokens: torch.Tensor,
        doc_offsets: torch.Tensor,
        sequence_length: int,
        batch_size: int,
        steps: int,
        seed: int,
    ) -> None:
        self.tokens = tokens
        self.doc_offsets = doc_offsets.long()
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.steps = steps
        self.seed = seed
        self._eligible_docs = self._build_eligible_docs()
        if self._eligible_docs.numel() < batch_size:
            raise ValueError("Not enough eligible documents for the requested batch size.")
        self.reset()

    def _build_eligible_docs(self) -> torch.Tensor:
        lengths = self.doc_offsets[1:] - self.doc_offsets[:-1]
        eligible = torch.nonzero(lengths >= (self.sequence_length + 1), as_tuple=False).squeeze(-1)
        if eligible.numel() == 0:
            raise ValueError("No documents are long enough for the requested sequence length.")
        return eligible.long()

    def _new_order(self) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self._epoch)
        perm = torch.randperm(self._eligible_docs.numel(), generator=generator)
        return self._eligible_docs[perm]

    def reset(self) -> None:
        self._epoch = 0
        self._order = self._new_order()
        self._cursor = 0
        self._lane_docs = torch.full((self.batch_size,), fill_value=-1, dtype=torch.long)
        self._lane_positions = torch.zeros((self.batch_size,), dtype=torch.long)
        for lane in range(self.batch_size):
            self._assign_next_doc(lane)
        self._step = 0

    def _assign_next_doc(self, lane: int) -> None:
        if self._cursor >= self._order.numel():
            self._epoch += 1
            self._order = self._new_order()
            self._cursor = 0
        doc_index = int(self._order[self._cursor].item())
        self._cursor += 1
        self._lane_docs[lane] = doc_index
        self._lane_positions[lane] = int(self.doc_offsets[doc_index].item())

    def next_batch(self, *, device: torch.device) -> dict[str, torch.Tensor]:
        if self._step >= self.steps:
            raise StopIteration
        input_rows = []
        target_rows = []
        reset_mask = torch.zeros((self.batch_size,), dtype=torch.bool)
        doc_ids = []
        for lane in range(self.batch_size):
            doc_index = int(self._lane_docs[lane].item())
            doc_start = int(self.doc_offsets[doc_index].item())
            doc_end = int(self.doc_offsets[doc_index + 1].item())
            pos = int(self._lane_positions[lane].item())
            if doc_end - pos < self.sequence_length + 1:
                self._assign_next_doc(lane)
                reset_mask[lane] = True
                doc_index = int(self._lane_docs[lane].item())
                doc_start = int(self.doc_offsets[doc_index].item())
                doc_end = int(self.doc_offsets[doc_index + 1].item())
                pos = int(self._lane_positions[lane].item())
            segment = self.tokens[pos : pos + self.sequence_length + 1]
            if segment.numel() != self.sequence_length + 1:
                raise RuntimeError("DocumentStreamBatcher failed to provide a full segment within one document.")
            input_rows.append(segment[:-1])
            target_rows.append(segment[1:])
            doc_ids.append(doc_index)
            self._lane_positions[lane] = pos + self.sequence_length
        self._step += 1
        return {
            "input_ids": torch.stack(input_rows).to(device=device, non_blocking=True),
            "targets": torch.stack(target_rows).to(device=device, non_blocking=True),
            "reset_mask": reset_mask.to(device=device, non_blocking=True),
            "doc_ids": torch.tensor(doc_ids, dtype=torch.long, device=device),
        }


def _build_doc_reset_segment_starts(doc_offsets: torch.Tensor, *, sequence_length: int) -> torch.Tensor:
    starts: list[torch.Tensor] = []
    stride = sequence_length
    min_doc_tokens = sequence_length + 1
    for doc_start, doc_end in zip(doc_offsets[:-1].tolist(), doc_offsets[1:].tolist(), strict=True):
        doc_length = doc_end - doc_start
        if doc_length < min_doc_tokens:
            continue
        doc_starts = torch.arange(doc_start, doc_end - sequence_length, stride, dtype=torch.long)
        if doc_starts.numel() > 0:
            starts.append(doc_starts)
    if not starts:
        raise ValueError("No document reset segments are available for the requested sequence length.")
    return torch.cat(starts)


class DocumentResetBatcher:
    def __init__(
        self,
        *,
        tokens: torch.Tensor,
        doc_offsets: torch.Tensor,
        sequence_length: int,
        batch_size: int,
        steps: int,
        seed: int,
    ) -> None:
        self.tokens = tokens
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.steps = steps
        self.seed = seed
        self._segment_starts = _build_doc_reset_segment_starts(doc_offsets.long(), sequence_length=sequence_length)
        if self._segment_starts.numel() < batch_size:
            raise ValueError("Not enough reset segments for the requested batch size.")
        self._segment_offsets = torch.arange(sequence_length + 1, dtype=torch.long)
        self.reset()

    def _new_order(self) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self._epoch)
        perm = torch.randperm(self._segment_starts.numel(), generator=generator)
        return self._segment_starts[perm]

    def reset(self) -> None:
        self._epoch = 0
        self._order = self._new_order()
        self._cursor = 0
        self._step = 0

    def _take_starts(self) -> torch.Tensor:
        if self._cursor + self.batch_size <= self._order.numel():
            starts = self._order[self._cursor : self._cursor + self.batch_size]
            self._cursor += self.batch_size
            return starts
        tail = self._order[self._cursor :]
        self._epoch += 1
        self._order = self._new_order()
        self._cursor = 0
        remaining = self.batch_size - tail.numel()
        head = self._order[:remaining]
        self._cursor = remaining
        return torch.cat((tail, head))

    def next_batch(self, *, device: torch.device) -> dict[str, torch.Tensor]:
        if self._step >= self.steps:
            raise StopIteration
        starts = self._take_starts()
        gather_index = starts.unsqueeze(1) + self._segment_offsets.unsqueeze(0)
        segments = self.tokens[gather_index]
        self._step += 1
        return {
            "input_ids": segments[:, :-1].to(device=device, non_blocking=True),
            "targets": segments[:, 1:].to(device=device, non_blocking=True),
            "reset_mask": torch.zeros((self.batch_size,), dtype=torch.bool, device=device),
            "doc_ids": torch.full((self.batch_size,), fill_value=-1, dtype=torch.long, device=device),
        }


def _load_doc_cache(config: DocumentCompareConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    payload = torch.load(config.cache_path, map_location="cpu", weights_only=False)
    required = {"train_tokens", "val_tokens", "train_doc_offsets", "val_doc_offsets", "vocab_size"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise ValueError(f"Document cache is missing required keys: {missing}")
    return (
        payload["train_tokens"].long(),
        payload["train_doc_offsets"].long(),
        payload["val_tokens"].long(),
        payload["val_doc_offsets"].long(),
        int(payload["vocab_size"]),
    )


def _make_realtext_config(config: DocumentCompareConfig, *, train_steps: int) -> RealTextConfig:
    return RealTextConfig(
        seed=config.seed,
        sequence_length=config.sequence_length,
        train_steps=train_steps,
        eval_interval=config.eval_interval,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        use_amp=config.use_amp,
        pin_memory=torch.cuda.is_available(),
        use_fused_adamw=config.use_fused_adamw,
        tensor_batching=True,
        cache_dataset_on_device=True,
        paired_train_batches=True,
        reseed_per_model=True,
        train_schedule_seed=config.seed,
        dropout=config.dropout,
        initial_eval=False,
    )


def _apply_reset_to_hidden(hidden: torch.Tensor | None, reset_mask: torch.Tensor) -> torch.Tensor | None:
    if hidden is None or not torch.any(reset_mask):
        return hidden
    next_hidden = hidden.clone()
    next_hidden[:, reset_mask, :] = 0
    return next_hidden


def _evaluate_doc_loss(
    model: StreamingPartialUntiedAssociativeLM,
    *,
    batcher: DocumentStreamBatcher,
    device: torch.device,
    use_amp: bool,
    carry_hidden: bool,
) -> float:
    model.eval()
    batcher.reset()
    hidden: torch.Tensor | None = None
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(batcher.steps):
            batch = batcher.next_batch(device=device)
            hidden = _apply_reset_to_hidden(hidden, batch["reset_mask"]) if carry_hidden else None
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, next_hidden, _, _ = model(
                    batch["input_ids"],
                    prev_hidden=hidden,
                    carry_hidden=carry_hidden,
                    carry_memory=False,
                )
                loss, token_count = _loss_and_tokens(logits, batch["targets"])
            total_loss += float(loss.item()) * token_count
            total_tokens += token_count
            hidden = next_hidden
    return total_loss / max(total_tokens, 1)


def _greedy_generate_doc_sample(
    model: StreamingPartialUntiedAssociativeLM,
    *,
    prompt_ids: list[int],
    sequence_length: int,
    device: torch.device,
    total_steps: int,
    carry_hidden: bool,
) -> list[int]:
    sequence = prompt_ids[:]
    model.eval()
    hidden: torch.Tensor | None = None
    with torch.no_grad():
        if carry_hidden:
            warmup = torch.tensor(prompt_ids[-sequence_length:], dtype=torch.long, device=device).unsqueeze(0)
            logits, hidden, _, _ = model(
                warmup,
                prev_hidden=None,
                carry_hidden=True,
                carry_memory=False,
            )
            sequence.append(int(logits[0, -1].argmax().item()))
            for _ in range(max(total_steps - 1, 0)):
                step_input = torch.tensor([[sequence[-1]]], dtype=torch.long, device=device)
                logits, hidden, _, _ = model(
                    step_input,
                    prev_hidden=hidden,
                    carry_hidden=True,
                    carry_memory=False,
                )
                sequence.append(int(logits[0, -1].argmax().item()))
        else:
            for _ in range(total_steps):
                window = sequence[-sequence_length:]
                input_ids = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
                logits, _, _, _ = model(
                    input_ids,
                    prev_hidden=None,
                    carry_hidden=False,
                    carry_memory=False,
                )
                sequence.append(int(logits[0, -1].argmax().item()))
    return sequence


def _train_variant(
    *,
    variant_name: str,
    carry_hidden: bool,
    config: DocumentCompareConfig,
    train_tokens: torch.Tensor,
    train_doc_offsets: torch.Tensor,
    val_tokens: torch.Tensor,
    val_doc_offsets: torch.Tensor,
    vocab_size: int,
    partial_token_ids: torch.Tensor,
) -> dict[str, Any]:
    device = torch.device(config.device)
    steps = math.ceil(config.target_tokens / (config.sequence_length * config.batch_size))
    real_config = _make_realtext_config(config, train_steps=steps)
    model = StreamingPartialUntiedAssociativeLM(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        memory_dim=config.memory_dim,
        dropout=config.dropout,
        max_length=config.sequence_length,
        untied_token_ids=partial_token_ids,
        memory_segments=1,
    ).to(device)
    optimizer = _build_optimizer(model, real_config, model_name="partial_untied")
    scheduler = _build_scheduler(optimizer, real_config)
    scaler = torch.amp.GradScaler(device="cuda", enabled=real_config.use_amp and device.type == "cuda")
    use_amp = real_config.use_amp and device.type == "cuda"
    parameter_list = [parameter for parameter in model.parameters() if parameter.requires_grad]

    seed_offset = 17 if variant_name == "doc_reset" else 29
    batcher_cls = DocumentStreamBatcher if carry_hidden else DocumentResetBatcher
    train_batcher = batcher_cls(
        tokens=train_tokens,
        doc_offsets=train_doc_offsets,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        steps=steps,
        seed=config.seed + seed_offset,
    )
    val_batcher = batcher_cls(
        tokens=val_tokens,
        doc_offsets=val_doc_offsets,
        sequence_length=config.sequence_length,
        batch_size=config.eval_batch_size,
        steps=config.eval_steps,
        seed=config.seed + seed_offset + 1,
    )
    tokenizer = None
    prompt_ids: list[int] | None = None
    next_sample_tokens: int | None = None
    samples: list[dict[str, Any]] = []
    if config.sample_prompt:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
        tokenizer.model_max_length = int(1e9)
        prompt_ids = tokenizer(config.sample_prompt, add_special_tokens=False)["input_ids"]
        if not prompt_ids:
            raise ValueError("sample_prompt tokenized to an empty prompt.")
        next_sample_tokens = config.sample_every_tokens

    history: list[dict[str, float]] = []
    step_times: list[float] = []
    tokens_seen = 0
    hidden: torch.Tensor | None = None
    latest_val_loss = _evaluate_doc_loss(
        model,
        batcher=val_batcher,
        device=device,
        use_amp=use_amp,
        carry_hidden=carry_hidden,
    )
    history.append({"step": 0.0, "tokens_seen": 0.0, "train_loss": float("nan"), "val_loss": float(latest_val_loss)})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    actual_target_tokens = steps * config.sequence_length * config.batch_size

    for step in range(1, steps + 1):
        batch = train_batcher.next_batch(device=device)
        hidden = _apply_reset_to_hidden(hidden, batch["reset_mask"]) if carry_hidden else None
        step_start = time.perf_counter()
        model.train()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, next_hidden, _, _ = model(
                batch["input_ids"],
                prev_hidden=hidden,
                carry_hidden=carry_hidden,
                carry_memory=False,
            )
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
        hidden = next_hidden

        if step % config.eval_interval == 0 or step == steps:
            latest_val_loss = _evaluate_doc_loss(
                model,
                batcher=val_batcher,
                device=device,
                use_amp=use_amp,
                carry_hidden=carry_hidden,
            )
            history.append(
                {
                    "step": float(step),
                    "tokens_seen": float(tokens_seen),
                    "train_loss": float(loss.item()),
                    "val_loss": float(latest_val_loss),
                }
            )
            elapsed = time.perf_counter() - start
            train_tok_per_sec = tokens_seen / max(elapsed, 1e-9)
            eta_seconds = max(actual_target_tokens - tokens_seen, 0) / max(train_tok_per_sec, 1e-9)
            print(
                f"{variant_name} step={step:,}/{steps:,} tok={tokens_seen:,}/{actual_target_tokens:,} "
                f"train={loss.item():.4f} val={latest_val_loss:.4f} tok/s={train_tok_per_sec:,.0f} eta={eta_seconds/60:.1f}m",
                flush=True,
            )
            if prompt_ids is not None and tokenizer is not None and next_sample_tokens is not None and (
                tokens_seen >= next_sample_tokens or step == steps
            ):
                generated_ids = _greedy_generate_doc_sample(
                    model,
                    prompt_ids=prompt_ids,
                    sequence_length=config.sequence_length,
                    device=device,
                    total_steps=config.sample_generation_tokens,
                    carry_hidden=carry_hidden,
                )
                sample_text = tokenizer.decode(generated_ids)
                sample_entry = {
                    "step": float(step),
                    "tokens_seen": float(tokens_seen),
                    "target_sample_tokens": float(next_sample_tokens),
                    "prompt": config.sample_prompt,
                    "generated": sample_text,
                }
                samples.append(sample_entry)
                print(
                    f"[{variant_name}] sample tok={tokens_seen:,}\n"
                    f"PROMPT: {config.sample_prompt}\n"
                    f"GENERATED: {sample_text}",
                    flush=True,
                )
                next_sample_tokens += config.sample_every_tokens

    total_time = time.perf_counter() - start
    pure_train_time = sum(step_times)
    return {
        "parameter_count": count_parameters(model),
        "history": history,
        "final_val_loss": latest_val_loss,
        "train_tokens_seen": tokens_seen,
        "train_tok_per_sec": tokens_seen / max(total_time, 1e-9),
        "pure_train_tok_per_sec": tokens_seen / max(pure_train_time, 1e-9),
        "step_time_mean_ms": statistics.fmean(step_times) * 1000.0,
        "step_time_median_ms": statistics.median(step_times) * 1000.0,
        "peak_vram_mb": _peak_vram_mb(config.device),
        "total_training_time_seconds": total_time,
        "carry_hidden": carry_hidden,
        "samples": samples,
    }


def _run_suite(config: DocumentCompareConfig, *, target_tokens: int) -> dict[str, Any]:
    run_config = DocumentCompareConfig(**{**asdict(config), "target_tokens": target_tokens})
    set_global_seed(run_config.seed)
    train_tokens, train_doc_offsets, val_tokens, val_doc_offsets, vocab_size = _load_doc_cache(run_config)
    block_size = run_config.sequence_length + 1
    train_blocks = train_tokens[: (train_tokens.numel() // block_size) * block_size].view(-1, block_size)
    partial_targets = train_blocks[:, 1:].contiguous()
    partial_dataset = type("PartialDataset", (), {"targets": partial_targets})()
    partial_token_ids = _top_token_ids(partial_dataset, count=run_config.partial_token_count, vocab_size=vocab_size)

    variant_specs: list[tuple[str, bool]] = []
    if run_config.variant_mode in {"both", "doc_reset"}:
        variant_specs.append(("doc_reset", False))
    if run_config.variant_mode in {"both", "doc_carry_hidden"}:
        variant_specs.append(("doc_carry_hidden", True))
    results = {
        variant_name: _train_variant(
            variant_name=variant_name,
            carry_hidden=carry_hidden,
            config=run_config,
            train_tokens=train_tokens,
            train_doc_offsets=train_doc_offsets,
            val_tokens=val_tokens,
            val_doc_offsets=val_doc_offsets,
            vocab_size=vocab_size,
            partial_token_ids=partial_token_ids,
        )
        for variant_name, carry_hidden in variant_specs
    }
    return {
        "benchmark": "language_partial_untied_document_compare",
        "config": {
            **asdict(run_config),
            "cache_path": str(run_config.cache_path),
            "output_path": str(run_config.output_path),
        },
        "results": results,
    }


def run_document_compare(config: DocumentCompareConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if config.run_cheap:
        payload["cheap"] = _run_suite(config, target_tokens=config.cheap_target_tokens)
    if config.run_hold:
        payload["hold"] = _run_suite(config, target_tokens=config.target_tokens)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare document-reset vs document-carried hidden-state training.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=5_000_000)
    parser.add_argument("--cheap-target-tokens", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-prompt", type=str, default=None)
    parser.add_argument("--sample-every-tokens", type=int, default=10_000_000)
    parser.add_argument("--sample-generation-tokens", type=int, default=40)
    parser.add_argument("--variant", choices=("both", "doc_reset", "doc_carry_hidden"), default="both")
    parser.add_argument("--skip-cheap", action="store_true")
    parser.add_argument("--no-hold", action="store_true")
    args = parser.parse_args()

    config = DocumentCompareConfig(
        cache_path=args.cache_path,
        output_path=args.output_path,
        target_tokens=args.target_tokens,
        cheap_target_tokens=args.cheap_target_tokens,
        seed=args.seed,
        device=args.device,
        sample_prompt=args.sample_prompt,
        sample_every_tokens=args.sample_every_tokens,
        sample_generation_tokens=args.sample_generation_tokens,
        variant_mode=args.variant,
        run_cheap=not args.skip_cheap,
        run_hold=not args.no_hold,
    )
    payload = run_document_compare(config)
    text = json.dumps(payload, indent=2)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
