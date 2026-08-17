from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import nn


ROOT = Path(__file__).resolve().parent
PAD_ID = 0
BOQ_ID = 6
EOQ_ID = 7
EOA_ID = 11
CONDITION_IDS = {"direct": 8, "cot": 9, "noisy": 12, "synth": 13}


def lecun_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=1.0 / math.sqrt(module.in_features), a=-3.0, b=3.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=1.0 / math.sqrt(module.embedding_dim), a=-3.0, b=3.0)


@dataclass
class Config:
    variant: str
    output_dir: Path
    seed: int = 13
    steps: int = 500
    eval_interval: int = 100
    batch_size: int = 4
    max_length: int = 192
    dim: int = 128
    heads: int = 4
    expansion: int = 4
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 32
    bp_warmup_ratio: float = 0.2


def read_source_rows() -> list[dict[str, str]]:
    source = ROOT / "official_source_subset"
    rows = []
    for path in sorted(source.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    try:
        import pyarrow.parquet as pq

        for path in sorted(source.rglob("*.parquet")):
            rows.extend(pq.read_table(path).to_pylist())
    except ImportError:
        pass
    if not rows:
        raise RuntimeError("official source subset is missing; run download_source_files.py")
    return rows


def build_examples(max_length: int) -> tuple[list[dict[str, list[int] | int]], list[dict[str, list[int] | int]]]:
    cache = ROOT / f"prefixlm_examples_len{max_length}.pt"
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        return payload["train"], payload["validation"]
    tokenizer = Tokenizer.from_file(str(ROOT / "official_tokenizer" / "tokenizer.json"))
    examples = []
    dropped = 0
    for row in read_source_rows():
        condition = str(row["condition"]).split(",")[0]
        condition_id = CONDITION_IDS.get(condition, CONDITION_IDS["direct"])
        instruction_body = tokenizer.encode(str(row["instruction"]), add_special_tokens=False).ids
        response_body = tokenizer.encode(str(row["response"]), add_special_tokens=False).ids
        instruction = [BOQ_ID, condition_id, *instruction_body, EOQ_ID]
        response = [*response_body, EOA_ID]
        if len(response) < 2:
            dropped += 1
            continue
        allowed_instruction = max_length - len(response) + 1
        if allowed_instruction < 3:
            response = response[: max_length - 2]
            response[-1] = EOA_ID
            allowed_instruction = 3
        if len(instruction) > allowed_instruction:
            instruction = [instruction[0], instruction[1], *instruction[-(allowed_instruction - 2):]]
        inputs = instruction + response[:-1]
        labels = [-100] * (len(instruction) - 1) + response
        examples.append({"inputs": inputs, "labels": labels, "prefix_length": len(instruction)})
    generator = random.Random(20260816)
    generator.shuffle(examples)
    validation = examples[:512]
    train = examples[512:]
    torch.save({"train": train, "validation": validation, "dropped": dropped}, cache)
    return train, validation


def batch_examples(examples, indices: list[int], max_length: int, device: torch.device):
    batch = len(indices)
    inputs = torch.full((batch, max_length), PAD_ID, dtype=torch.long)
    labels = torch.full((batch, max_length), -100, dtype=torch.long)
    valid = torch.zeros((batch, max_length), dtype=torch.bool)
    prefixes = []
    for row_index, example_index in enumerate(indices):
        example = examples[example_index]
        length = len(example["inputs"])
        inputs[row_index, :length] = torch.tensor(example["inputs"])
        labels[row_index, :length] = torch.tensor(example["labels"])
        valid[row_index, :length] = True
        prefixes.append(int(example["prefix_length"]))
    positions = torch.arange(max_length)
    query = positions.view(1, max_length, 1)
    key = positions.view(1, 1, max_length)
    prefix = torch.tensor(prefixes).view(batch, 1, 1)
    prefix_query = query < prefix
    allowed = torch.where(prefix_query, key < prefix, key <= query)
    allowed &= valid.view(batch, 1, max_length)
    allowed &= valid.view(batch, max_length, 1)
    return inputs.to(device), labels.to(device), allowed.unsqueeze(1).to(device)


class GatedAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, expansion: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.attn_norm = nn.RMSNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_gate = nn.Linear(dim, heads, bias=True)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.mlp_norm = nn.RMSNorm(dim, eps=1e-6)
        self.mlp_in = nn.Linear(dim, 2 * expansion * dim, bias=False)
        self.mlp_out = nn.Linear(expansion * dim, dim, bias=False)
        nn.init.zeros_(self.attn_gate.weight)
        nn.init.zeros_(self.attn_gate.bias)

    def apply_rope(self, tensor: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tensor.size(-2), device=tensor.device, dtype=torch.float32)
        frequencies = torch.arange(0, self.head_dim, 2, device=tensor.device, dtype=torch.float32)
        frequencies = 1.0 / (10000.0 ** (frequencies / self.head_dim))
        angles = positions[:, None] * frequencies[None, :]
        cos = angles.cos().to(tensor.dtype).view(1, 1, tensor.size(-2), -1)
        sin = angles.sin().to(tensor.dtype).view(1, 1, tensor.size(-2), -1)
        even, odd = tensor[..., 0::2], tensor[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attn_norm(x)
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        shape = (x.size(0), x.size(1), self.heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        q = self.apply_rope(q)
        k = self.apply_rope(k)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        gate = torch.sigmoid(self.attn_gate(normalized)).transpose(1, 2).unsqueeze(-1)
        attended = (attended * gate).transpose(1, 2).contiguous().view_as(x)
        x = x + self.attn_out(attended)
        left, right = self.mlp_in(self.mlp_norm(x)).chunk(2, dim=-1)
        return x + self.mlp_out(F.silu(left) * right)


class PrefixTransformer(nn.Module):
    def __init__(self, config: Config, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.dim)
        self.blocks = nn.ModuleList([GatedAttentionBlock(config.dim, config.heads, config.expansion) for _ in range(2)])
        self.final_norm = nn.RMSNorm(config.dim, eps=1e-6)
        self.apply(lecun_init)
        for block in self.blocks:
            nn.init.zeros_(block.attn_gate.weight)
            nn.init.zeros_(block.attn_gate.bias)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, bp_steps: int = 5) -> torch.Tensor:
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x, mask)
        return F.linear(self.final_norm(x), self.embedding.weight)


class PrefixHRM(nn.Module):
    def __init__(self, config: Config, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.dim)
        self.high = GatedAttentionBlock(config.dim, config.heads, config.expansion)
        self.low = GatedAttentionBlock(config.dim, config.heads, config.expansion)
        self.register_buffer("low_initial", torch.empty(config.dim), persistent=True)
        self.final_norm = nn.RMSNorm(config.dim, eps=1e-6)
        self.apply(lecun_init)
        nn.init.zeros_(self.high.attn_gate.weight)
        nn.init.zeros_(self.high.attn_gate.bias)
        nn.init.zeros_(self.low.attn_gate.weight)
        nn.init.zeros_(self.low.attn_gate.bias)
        nn.init.trunc_normal_(self.low_initial, std=1.0, a=-3.0, b=3.0)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, bp_steps: int = 5) -> torch.Tensor:
        high = self.embedding(tokens)
        low = self.low_initial.to(high.dtype).view(1, 1, -1).expand_as(high)
        high_steps = min(2, bp_steps - 1)
        low_steps = bp_steps - high_steps
        low_index = 0
        for high_index in range(2):
            for _ in range(3):
                track = torch.is_grad_enabled() and low_index >= 6 - low_steps
                with torch.set_grad_enabled(track):
                    low = self.low(low + high, mask)
                low_index += 1
            track = torch.is_grad_enabled() and high_index >= 2 - high_steps
            with torch.set_grad_enabled(track):
                high = self.high(high + low, mask)
        return F.linear(self.final_norm(high), self.embedding.weight)


def learning_rate(config: Config, step: int) -> float:
    if step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


def bp_steps(config: Config, step: int) -> int:
    if config.variant != "hrm_warmup":
        return 5
    warmup = max(1, int(config.steps * config.bp_warmup_ratio))
    return 2 + int(min(1.0, step / warmup) * 3)


@torch.inference_mode()
def evaluate(model, examples, config: Config, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(examples), config.batch_size):
        indices = list(range(start, min(start + config.batch_size, len(examples))))
        inputs, labels, mask = batch_examples(examples, indices, config.max_length, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(inputs, mask, 5)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100, reduction="sum")
        tokens = int((labels != -100).sum())
        total_loss += float(loss)
        total_tokens += tokens
    model.train()
    return total_loss / total_tokens


def train(config: Config) -> None:
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda")
    train_examples, validation = build_examples(config.max_length)
    vocab_size = 65536
    model = PrefixTransformer(config, vocab_size) if config.variant == "transformer" else PrefixHRM(config, vocab_size)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator().manual_seed(config.seed)
    schedule = torch.randint(0, len(train_examples), (config.steps, config.batch_size), generator=generator)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    response_tokens = 0
    for step in range(1, config.steps + 1):
        inputs, labels, mask = batch_examples(train_examples, schedule[step - 1].tolist(), config.max_length, device)
        lr = learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(inputs, mask, bp_steps(config, step))
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        response_tokens += int((labels != -100).sum())
        if step == 1 or step % config.eval_interval == 0 or step == config.steps:
            val_loss = evaluate(model, validation, config, device)
            elapsed = time.perf_counter() - start_time
            record = {"step": step, "response_tokens": response_tokens, "train_loss": float(loss), "val_loss": val_loss, "lr": lr, "bp_steps": bp_steps(config, step), "response_tok_per_sec": response_tokens / elapsed}
            history.append(record)
            print("METRIC " + json.dumps(record), flush=True)
    report = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "train_examples": len(train_examples),
        "validation_examples": len(validation),
        "history": history,
    }
    (config.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "config": report["config"]}, config.output_dir / "checkpoint.pt")
    print("RESULT " + json.dumps(report), flush=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("transformer", "hrm_full", "hrm_warmup"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    return Config(variant=args.variant, output_dir=args.output_dir, steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    train(parse_args())
