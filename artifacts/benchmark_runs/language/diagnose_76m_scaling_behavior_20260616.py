import argparse
import json
from pathlib import Path
from typing import Any

import torch

from standalone_longseq_anchor_train import CausalConvFactorizedLM, TrainConfig


PROMPTS = [
    "Question: What is machine learning? Answer:",
    "Python function to add two numbers:\n",
    "In plain English, validation loss measures",
    "A child asked why the sky is blue. The teacher said",
    "To debug a failing training run, first",
    "Write a polite email asking for more information:",
]


CONTINUATION_TESTS = [
    {
        "name": "machine_learning_definition",
        "prompt": "Question: What is machine learning? Answer:",
        "continuations": {
            "expected": " Machine learning is a method where computers learn patterns from data.",
            "observed_style": " The best way to learn a little about the next step is to have them practice.",
            "generic_web": " The following information is provided by the Department of Education.",
        },
    },
    {
        "name": "python_add_function",
        "prompt": "Python function to add two numbers:\n",
        "continuations": {
            "expected": "def add(a, b):\n    return a + b\n",
            "observed_style": "A variable in the first determines the number of functions of order variables.",
            "generic_web": "This section describes the procedure in the following table.",
        },
    },
    {
        "name": "validation_loss_definition",
        "prompt": "In plain English, validation loss measures",
        "continuations": {
            "expected": " how well the model predicts unseen validation data.",
            "observed_style": " are the result of reduced yields of new products.",
            "generic_web": " the data that are available in the report.",
        },
    },
    {
        "name": "sky_blue_answer",
        "prompt": "A child asked why the sky is blue. The teacher said",
        "continuations": {
            "expected": " sunlight is scattered by air molecules, and blue light scatters more.",
            "observed_style": " just as a child, you said. Now, you can't see what's going on.",
            "generic_web": " the following program was designed for preschool students.",
        },
    },
]


def load_config(raw: dict[str, Any], checkpoint_path: Path) -> TrainConfig:
    defaults = {
        "cache_path": checkpoint_path.parent / "unused_cache.pt",
        "output_dir": checkpoint_path.parent,
        "run_name": checkpoint_path.parent.name,
    }
    raw = {**defaults, **raw}
    names = set(TrainConfig.__dataclass_fields__)
    clean: dict[str, Any] = {}
    for name in names:
        if name not in raw:
            continue
        value = raw[name]
        if name in {"cache_path", "output_dir", "resume_checkpoint"} and value is not None:
            value = Path(value)
        clean[name] = value
    return TrainConfig(**clean)


@torch.inference_mode()
def generate(
    model: CausalConvFactorizedLM,
    tokenizer: Any,
    prompt: str,
    *,
    config: TrainConfig,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        window = generated[:, -int(config.sequence_length) :]
        hidden = model.factor_down(model.features(window))
        logits = model.factor_up(hidden[:, -1, :]).float().squeeze(0)
        if top_k == 1 or temperature <= 0:
            next_token = torch.argmax(logits).view(1)
        else:
            logits = logits / temperature
            if 0 < top_k < logits.numel():
                values, indices = torch.topk(logits, top_k)
                probs = torch.softmax(values, dim=-1)
                next_token = indices[torch.multinomial(probs, num_samples=1)]
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token.view(1, 1)], dim=1)
        if int(next_token.item()) == int(tokenizer.eos_token_id):
            break
    return tokenizer.decode(generated.squeeze(0).tolist(), skip_special_tokens=False)


@torch.inference_mode()
def continuation_nll(
    model: CausalConvFactorizedLM,
    tokenizer: Any,
    prompt: str,
    continuation: str,
    *,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
    ids = torch.tensor([prompt_ids + cont_ids], dtype=torch.long, device=device)
    if len(cont_ids) == 0:
        raise ValueError("empty continuation")
    if ids.size(1) > int(config.sequence_length):
        ids = ids[:, -int(config.sequence_length) :]
    input_ids = ids[:, :-1]
    targets = ids[:, 1:]
    hidden = model.factor_down(model.features(input_ids))
    logits = model.factor_up(hidden).float()
    losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    prompt_len = len(prompt_ids)
    start = max(prompt_len - 1, 0)
    cont_losses = losses[:, start:]
    return {
        "tokens": len(cont_ids),
        "mean_nll": float(cont_losses.mean().item()),
        "sum_nll": float(cont_losses.sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = load_config(dict(checkpoint.get("config", {})), args.checkpoint)
    model = CausalConvFactorizedLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, local_files_only=True)
    tokenizer.model_max_length = int(1e9)

    decode_settings = [
        {"name": "greedy", "temperature": 0.0, "top_k": 1},
        {"name": "low_temp_top20", "temperature": 0.35, "top_k": 20},
        {"name": "prior_sample_top64", "temperature": 0.8, "top_k": 64},
    ]
    generations: list[dict[str, Any]] = []
    for setting in decode_settings:
        torch.manual_seed(20260616)
        for prompt in PROMPTS:
            text = generate(
                model,
                tokenizer,
                prompt,
                config=config,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=float(setting["temperature"]),
                top_k=int(setting["top_k"]),
            )
            generations.append({"setting": setting["name"], "prompt": prompt, "text": text})

    continuation_scores: list[dict[str, Any]] = []
    for test in CONTINUATION_TESTS:
        scored = []
        for label, continuation in test["continuations"].items():
            score = continuation_nll(
                model,
                tokenizer,
                test["prompt"],
                continuation,
                config=config,
                device=device,
            )
            scored.append({"label": label, "continuation": continuation, **score})
        scored.sort(key=lambda row: row["mean_nll"])
        continuation_scores.append({"name": test["name"], "prompt": test["prompt"], "ranked": scored})

    result = {
        "checkpoint": str(args.checkpoint),
        "step": int(checkpoint.get("step", -1)),
        "tokens_seen": int(checkpoint.get("tokens_seen", -1)),
        "block_type": config.block_type,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "sequence_length": config.sequence_length,
        "conv_layers": config.conv_layers,
        "memory_rank": config.memory_rank,
        "memory_kernel_size": config.landmark_stride,
        "generations": generations,
        "continuation_scores": continuation_scores,
    }
    (args.output_dir / "diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output_dir / "diagnostics.md").open("w", encoding="utf-8") as handle:
        handle.write("# 76M Scaling Diagnostics\n\n")
        handle.write(f"checkpoint: `{args.checkpoint}`\n\n")
        handle.write("## Continuation Preference Tests\n\n")
        for test in continuation_scores:
            handle.write(f"### {test['name']}\n\n")
            handle.write(f"Prompt: `{test['prompt']}`\n\n")
            for row in test["ranked"]:
                handle.write(f"- `{row['label']}` mean_nll={row['mean_nll']:.4f}: {row['continuation']!r}\n")
            handle.write("\n")
        handle.write("## Decode Sweep\n\n")
        for row in generations:
            handle.write(f"### {row['setting']} | {row['prompt']!r}\n\n")
            handle.write(row["text"] + "\n\n")
    print("DIAGNOSTICS_RESULT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
