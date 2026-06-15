from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import tiktoken
from datasets import load_dataset


OUT = Path(os.environ.get(
    "REAL_CACHE_OUT",
    r"E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\neuron_search_20260605\manual_self\real_cache_finewebedu_sample_seq255_train768_val64_gpt2.pt",
))
META = OUT.with_suffix(".meta.json")
SEQUENCE_LENGTH = int(os.environ.get("REAL_CACHE_SEQUENCE_LENGTH", "255"))
BLOCK = SEQUENCE_LENGTH + 1
TRAIN_BLOCKS = int(os.environ.get("REAL_CACHE_TRAIN_BLOCKS", "768"))
VAL_BLOCKS = int(os.environ.get("REAL_CACHE_VAL_BLOCKS", "64"))
TRAIN_TOKENS = TRAIN_BLOCKS * BLOCK
VAL_TOKENS = VAL_BLOCKS * BLOCK
TOTAL_TOKENS = TRAIN_TOKENS + VAL_TOKENS


def stream_dataset():
    errors: list[str] = []
    for name in ("sample-10BT", None):
        try:
            if name is None:
                return load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
            return load_dataset("HuggingFaceFW/fineweb-edu", name=name, split="train", streaming=True)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"name={name!r}: {exc!r}")
    raise RuntimeError("could not load FineWebEdu stream: " + " | ".join(errors))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        print(f"EXISTS {OUT}")
        print(META.read_text(encoding="utf-8") if META.exists() else "{}")
        return
    encoding = tiktoken.get_encoding("gpt2")
    tokens: list[int] = []
    docs = 0
    chars = 0
    for row in stream_dataset():
        text = row.get("text") if isinstance(row, dict) else None
        if not text:
            continue
        docs += 1
        chars += len(text)
        tokens.extend(encoding.encode_ordinary(text))
        tokens.append(encoding.eot_token)
        if docs == 1 or docs % 100 == 0:
            print(f"cache_build docs={docs} chars={chars} tokens={len(tokens)}/{TOTAL_TOKENS}", flush=True)
        if len(tokens) >= TOTAL_TOKENS:
            break
    if len(tokens) < TOTAL_TOKENS:
        raise RuntimeError(f"only collected {len(tokens)} tokens, need {TOTAL_TOKENS}")
    all_tokens = torch.tensor(tokens[:TOTAL_TOKENS], dtype=torch.long)
    train_tokens = all_tokens[:TRAIN_TOKENS].contiguous()
    val_tokens = all_tokens[TRAIN_TOKENS:].contiguous()
    tmp = OUT.with_suffix(".tmp")
    torch.save({"train_tokens": train_tokens, "val_tokens": val_tokens, "vocab_size": 50257}, tmp)
    tmp.replace(OUT)
    meta = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config_attempted_first": "sample-10BT",
        "encoding": "gpt2",
        "sequence_length": SEQUENCE_LENGTH,
        "train_blocks": TRAIN_BLOCKS,
        "val_blocks": VAL_BLOCKS,
        "train_tokens": int(train_tokens.numel()),
        "val_tokens": int(val_tokens.numel()),
        "docs_read": docs,
        "chars_read": chars,
        "path": str(OUT),
    }
    META.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
