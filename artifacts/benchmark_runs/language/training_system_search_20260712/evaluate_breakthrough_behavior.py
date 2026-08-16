from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).parents[1]
TRAINER_PATH = ROOT / "token_recall_search_20260616" / "token_recall_train.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def induction_targets(input_ids: torch.Tensor, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch, length = input_ids.shape
    flat_tokens = input_ids.reshape(-1)
    batch_ids = torch.arange(batch, device=input_ids.device).repeat_interleave(length)
    keys = flat_tokens + batch_ids * vocab_size
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys.index_select(0, order)
    previous = torch.roll(order, 1)
    valid_sorted = torch.ones_like(sorted_keys, dtype=torch.bool)
    valid_sorted[0] = False
    valid_sorted[1:] = sorted_keys[1:] == sorted_keys[:-1]
    valid_sorted &= torch.remainder(previous, length) + 1 < length
    retrieved_sorted = flat_tokens.index_select(0, (previous + 1).clamp(max=flat_tokens.numel() - 1))
    retrieved = torch.zeros_like(flat_tokens).scatter_(0, order, retrieved_sorted)
    valid = torch.zeros_like(flat_tokens, dtype=torch.bool).scatter_(0, order, valid_sorted)
    return retrieved.view(batch, length), valid.view(batch, length)


@torch.inference_mode()
def evaluate_model(model, val_inputs, val_targets, device, chunk_size: int = 256) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    opportunity_count = 0
    opportunity_correct = 0
    valid_retrieval_count = 0
    for row in range(val_inputs.size(0)):
        inputs = val_inputs[row : row + 1].to(device)
        targets = val_targets[row : row + 1].to(device)
        retrieved, valid = induction_targets(inputs, model.vocab_size)
        opportunity = valid & retrieved.eq(targets)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            hidden = model.factor_down(model.features(inputs))
            for start in range(0, inputs.size(1), chunk_size):
                end = min(start + chunk_size, inputs.size(1))
                positions = torch.arange(start, end, device=device)
                logits = model.candidate_logits(
                    hidden=hidden[:, start:end],
                    input_ids=inputs,
                    candidate_ids=torch.arange(model.vocab_size, device=device),
                    weight=model.factor_up.weight,
                    bias=model.factor_up.bias,
                    positions=positions,
                )
                chunk_targets = targets[:, start:end]
                total_loss += float(F.cross_entropy(logits.flatten(0, 1), chunk_targets.flatten(), reduction="sum"))
                predictions = logits.argmax(dim=-1)
                total_correct += int(predictions.eq(chunk_targets).sum())
                total_count += chunk_targets.numel()
                chunk_opportunity = opportunity[:, start:end]
                opportunity_count += int(chunk_opportunity.sum())
                opportunity_correct += int((predictions.eq(chunk_targets) & chunk_opportunity).sum())
                valid_retrieval_count += int(valid[:, start:end].sum())
    return {
        "full_vocab_loss": total_loss / total_count,
        "top1_accuracy": total_correct / total_count,
        "valid_retrieval_positions": valid_retrieval_count,
        "exact_induction_opportunities": opportunity_count,
        "induction_opportunity_top1": opportunity_correct / max(opportunity_count, 1),
    }


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, device, seed: int) -> dict[str, object]:
    ids = torch.tensor([tokenizer.encode(prompt)], device=device, dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(seed)
    generated = []
    for _ in range(64):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            hidden = model.factor_down(model.features(ids))
            position = torch.tensor([ids.size(1) - 1], device=device)
            logits = model.candidate_logits(
                hidden=hidden[:, -1:],
                input_ids=ids,
                candidate_ids=torch.arange(model.vocab_size, device=device),
                weight=model.factor_up.weight,
                bias=model.factor_up.bias,
                positions=position,
            )[0, 0]
        top_values, top_indices = torch.topk(logits.float() / 0.8, 40)
        probabilities = torch.softmax(top_values, dim=-1)
        selected = torch.multinomial(probabilities, 1, generator=generator)
        next_id = top_indices[selected]
        ids = torch.cat((ids, next_id.view(1, 1)), dim=1)
        generated.append(int(next_id))
    distinct_1 = len(set(generated)) / len(generated)
    bigrams = list(zip(generated, generated[1:]))
    distinct_2 = len(set(bigrams)) / max(len(bigrams), 1)
    return {
        "prompt": prompt,
        "text": tokenizer.decode(ids[0].tolist()),
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "immediate_repeat_rate": sum(a == b for a, b in bigrams) / max(len(bigrams), 1),
    }


def load_checkpoint_model(module, checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = module.TrainConfig(**checkpoint["config"])
    model = module.CausalConvFactorizedLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    reference_module = load_module("behavior_reference_trainer", TRAINER_PATH)
    os.environ["WAVE10_HEAD_MULTIPLIER"] = "1"
    os.environ["WAVE10_IDENTITY_HEAD"] = "1"
    os.environ["WAVE10_RESIDUAL_HEAD"] = "0"
    import sorted_induction_train as candidate_wrapper

    candidate_module = candidate_wrapper.experiment.experiment.trainer
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    val_tokens = cache["val_tokens"]
    length = 10_161
    blocks = val_tokens[: 8 * length].view(8, length)
    val_inputs, val_targets = blocks[:, :-1], blocks[:, 1:]
    reference, _ = load_checkpoint_model(reference_module, args.reference, device)
    candidate, _ = load_checkpoint_model(candidate_module, args.candidate, device)
    payload = {
        "reference": evaluate_model(reference, val_inputs, val_targets, device),
        "candidate": evaluate_model(candidate, val_inputs, val_targets, device),
    }
    try:
        if args.skip_generation:
            raise RuntimeError("generation skipped by request")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        prompts = ["The most important thing", "In the year 2050", "Once upon a time", "The computer"]
        payload["generation"] = {
            "reference": [generate(reference, tokenizer, prompt, device, 100 + i) for i, prompt in enumerate(prompts)],
            "candidate": [generate(candidate, tokenizer, prompt, device, 100 + i) for i, prompt in enumerate(prompts)],
        }
    except Exception as exc:
        payload["generation_error"] = repr(exc)
    rendered = json.dumps(payload, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
