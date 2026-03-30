from __future__ import annotations

import json

import torch

from arc_tactic3.language_recurrent_screening_portfolio import ScreeningCandidate, run_screening_portfolio


def _make_doc_cache(path) -> None:
    docs = [
        torch.arange(0, 20, dtype=torch.int32),
        torch.arange(100, 122, dtype=torch.int32),
        torch.arange(200, 224, dtype=torch.int32),
        torch.arange(300, 326, dtype=torch.int32),
    ]
    train_docs = docs[:3]
    val_docs = docs[3:]
    train_tokens = torch.cat(train_docs)
    val_tokens = torch.cat(val_docs)
    train_offsets = [0]
    for doc in train_docs:
        train_offsets.append(train_offsets[-1] + doc.numel())
    val_offsets = [0]
    for doc in val_docs:
        val_offsets.append(val_offsets[-1] + doc.numel())
    torch.save(
        {
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "train_doc_offsets": torch.tensor(train_offsets, dtype=torch.int64),
            "val_doc_offsets": torch.tensor(val_offsets, dtype=torch.int64),
            "vocab_size": 1024,
        },
        path,
    )


def test_run_screening_portfolio_smoke(tmp_path) -> None:
    cache_path = tmp_path / "doc_cache.pt"
    _make_doc_cache(cache_path)
    output_dir = tmp_path / "screens"
    payload = run_screening_portfolio(
        cache_path=cache_path,
        output_dir=output_dir,
        seed=13,
        device="cpu",
        sample_prompt=None,
        sample_every_tokens=16,
        sample_generation_tokens=4,
        candidates=[
            ScreeningCandidate(
                label="tiny_factorized",
                target_tokens=112,
                sequence_length=7,
                partial_learning_rate=1e-3,
                partial_warmup_steps=2,
                partial_batch_size=2,
                partial_eval_batch_size=1,
                factorized_embedding_dim=8,
                factorized_hidden_dim=12,
                factorized_memory_dim=8,
                factorized_untied_rank=8,
            )
        ],
    )
    assert payload["status"] == "completed"
    assert "tiny_factorized" in payload["scoreboard"]
    summary_path = output_dir / "summary.json"
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["status"] == "completed"
