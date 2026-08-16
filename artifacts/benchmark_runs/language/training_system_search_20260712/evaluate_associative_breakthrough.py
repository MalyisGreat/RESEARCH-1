from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F


os.environ.setdefault("PHRASE_ORDERS", "2,3")
os.environ.setdefault("PHRASE_HISTORY", "1")
os.environ.setdefault("SEMANTIC_TABLES", "2")
os.environ.setdefault("SEMANTIC_CANDIDATES", "3")

import phrase_semantic_induction_train as architecture


FREQUENCY_BUCKETS = (
    ("unseen_or_rare_0_10", 0, 10),
    ("frequency_11_100", 11, 100),
    ("frequency_101_1000", 101, 1_000),
    ("frequency_over_1000", 1_001, 2**63 - 1),
)


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = architecture.trainer.TrainConfig(**checkpoint["config"])
    model = architecture.PhraseSemanticInductionModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


@torch.inference_mode()
def evaluate(model, val_inputs, val_targets, frequencies, device, chunk_size: int = 256):
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    bucket_sums = {name: 0.0 for name, _, _ in FREQUENCY_BUCKETS}
    bucket_counts = {name: 0 for name, _, _ in FREQUENCY_BUCKETS}
    phrase_opportunities = {order: 0 for order in model.phrase_orders}
    phrase_correct = {order: 0 for order in model.phrase_orders}

    vocabulary = torch.arange(model.vocab_size, device=device)
    for row in range(val_inputs.size(0)):
        inputs = val_inputs[row : row + 1].to(device)
        targets = val_targets[row : row + 1].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            hidden = model.factor_down(model.features(inputs))
        for start in range(0, inputs.size(1), chunk_size):
            end = min(start + chunk_size, inputs.size(1))
            positions = torch.arange(start, end, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model.candidate_logits(
                    hidden=hidden[:, start:end],
                    input_ids=inputs,
                    candidate_ids=vocabulary,
                    weight=model.factor_up.weight,
                    bias=model.factor_up.bias,
                    positions=positions,
                )
            chunk_targets = targets[:, start:end]
            losses = F.cross_entropy(
                logits.float().flatten(0, 1), chunk_targets.flatten(), reduction="none"
            ).view_as(chunk_targets)
            predictions = logits.argmax(dim=-1)
            total_loss += float(losses.sum())
            total_tokens += chunk_targets.numel()
            total_correct += int(predictions.eq(chunk_targets).sum())

            target_frequency = frequencies[chunk_targets.cpu()].to(device)
            for name, minimum, maximum in FREQUENCY_BUCKETS:
                mask = (target_frequency >= minimum) & (target_frequency <= maximum)
                bucket_sums[name] += float(losses[mask].sum())
                bucket_counts[name] += int(mask.sum())

            for order in model.phrase_orders:
                retrieved = model._phrase_tokens[order][:, start:end]
                valid = model._phrase_valid[order][:, start:end]
                opportunity = valid & retrieved.eq(chunk_targets.unsqueeze(-1))
                opportunity = opportunity.any(dim=-1)
                phrase_opportunities[order] += int(opportunity.sum())
                phrase_correct[order] += int((predictions.eq(chunk_targets) & opportunity).sum())

    return {
        "full_vocab_loss": total_loss / total_tokens,
        "top1_accuracy": total_correct / total_tokens,
        "token_count": total_tokens,
        "frequency_buckets": {
            name: {
                "loss": bucket_sums[name] / max(bucket_counts[name], 1),
                "tokens": bucket_counts[name],
            }
            for name, _, _ in FREQUENCY_BUCKETS
        },
        "phrase_opportunities": {
            str(order): {
                "positions": phrase_opportunities[order],
                "top1_accuracy": phrase_correct[order] / max(phrase_opportunities[order], 1),
            }
            for order in model.phrase_orders
        },
    }


def learned_retrieval_state(model) -> dict[str, object]:
    return {
        "unigram_scale": float(F.softplus(model.induction_log_scale).detach()),
        "phrase_scales": {
            key: float(F.softplus(value).detach()) for key, value in model.phrase_log_scales.items()
        },
        "semantic_scale": float(F.softplus(model.semantic_log_scale).detach()),
        "semantic_temperature": float(model.semantic_log_temperature.exp().detach()),
        "unigram_gate_bias": float(model.induction_gate.bias.detach()),
        "phrase_gate_biases": {
            key: float(value.bias.detach()) for key, value in model.phrase_gates.items()
        },
        "semantic_gate_bias": float(model.semantic_gate.bias.detach()),
    }


def disable_parameter(parameter: torch.Tensor) -> None:
    parameter.fill_(-30.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-blocks", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    block_size = 10_161
    blocks = cache["val_tokens"][: args.val_blocks * block_size].view(args.val_blocks, block_size)
    val_inputs = blocks[:, :-1].contiguous()
    val_targets = blocks[:, 1:].contiguous()
    frequencies = torch.bincount(cache["train_tokens"].long(), minlength=50_257)
    model = load_model(args.checkpoint, device)

    original_unigram = model.induction_log_scale.detach().clone()
    original_semantic = model.semantic_log_scale.detach().clone()
    original_phrases = {
        key: value.detach().clone() for key, value in model.phrase_log_scales.items()
    }

    payload = {"learned_retrieval_state": learned_retrieval_state(model), "interventions": {}}
    payload["interventions"]["full_model"] = evaluate(
        model, val_inputs, val_targets, frequencies, device
    )
    with torch.no_grad():
        disable_parameter(model.semantic_log_scale)
    payload["interventions"]["semantic_disabled"] = evaluate(
        model, val_inputs, val_targets, frequencies, device
    )
    with torch.no_grad():
        model.semantic_log_scale.copy_(original_semantic)
        for parameter in model.phrase_log_scales.values():
            disable_parameter(parameter)
    payload["interventions"]["phrase_disabled"] = evaluate(
        model, val_inputs, val_targets, frequencies, device
    )
    with torch.no_grad():
        disable_parameter(model.semantic_log_scale)
    payload["interventions"]["semantic_and_phrase_disabled"] = evaluate(
        model, val_inputs, val_targets, frequencies, device
    )
    with torch.no_grad():
        disable_parameter(model.induction_log_scale)
    payload["interventions"]["all_induction_disabled"] = evaluate(
        model, val_inputs, val_targets, frequencies, device
    )

    with torch.no_grad():
        model.induction_log_scale.copy_(original_unigram)
        model.semantic_log_scale.copy_(original_semantic)
        for key, parameter in model.phrase_log_scales.items():
            parameter.copy_(original_phrases[key])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
