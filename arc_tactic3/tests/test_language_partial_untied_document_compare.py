from __future__ import annotations

import json

import torch

from arc_tactic3 import language_partial_untied_document_compare as module
from arc_tactic3.language_partial_untied_document_compare import (
    DocumentCompareConfig,
    DocumentResetBatcher,
    DocumentStreamBatcher,
    run_document_compare,
)


def _make_doc_cache(path) -> None:
    docs = [
        torch.arange(0, 12, dtype=torch.int32),
        torch.arange(100, 114, dtype=torch.int32),
        torch.arange(200, 214, dtype=torch.int32),
        torch.arange(300, 318, dtype=torch.int32),
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


def test_document_stream_batcher_resets_at_doc_boundary() -> None:
    docs = [
        torch.arange(0, 10, dtype=torch.long),
        torch.arange(100, 110, dtype=torch.long),
        torch.arange(200, 210, dtype=torch.long),
    ]
    tokens = torch.cat(docs)
    offsets = torch.tensor([0, 10, 20, 30], dtype=torch.long)
    batcher = DocumentStreamBatcher(tokens=tokens, doc_offsets=offsets, sequence_length=4, batch_size=2, steps=4, seed=13)
    seen_reset = False
    for _ in range(4):
        batch = batcher.next_batch(device=torch.device("cpu"))
        if bool(batch["reset_mask"].any().item()):
            seen_reset = True
            break
    assert seen_reset


def test_document_reset_batcher_stays_within_doc_segments() -> None:
    docs = [
        torch.arange(0, 10, dtype=torch.long),
        torch.arange(100, 110, dtype=torch.long),
        torch.arange(200, 214, dtype=torch.long),
    ]
    tokens = torch.cat(docs)
    offsets = torch.tensor([0, 10, 20, 34], dtype=torch.long)
    batcher = DocumentResetBatcher(tokens=tokens, doc_offsets=offsets, sequence_length=4, batch_size=2, steps=3, seed=13)
    seen = []
    for _ in range(3):
        batch = batcher.next_batch(device=torch.device("cpu"))
        assert not bool(batch["reset_mask"].any().item())
        for row in batch["input_ids"]:
            seen.append(tuple(int(token) for token in row.tolist()))
    valid = {
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (100, 101, 102, 103),
        (104, 105, 106, 107),
        (200, 201, 202, 203),
        (204, 205, 206, 207),
        (208, 209, 210, 211),
    }
    assert set(seen).issubset(valid)


def test_run_document_compare_smoke(tmp_path) -> None:
    cache_path = tmp_path / "doc_cache.pt"
    _make_doc_cache(cache_path)
    output_path = tmp_path / "doc_compare.json"
    config = DocumentCompareConfig(
        cache_path=cache_path,
        output_path=output_path,
        target_tokens=56,
        cheap_target_tokens=56,
        seed=13,
        sequence_length=7,
        batch_size=2,
        eval_batch_size=1,
        eval_steps=1,
        eval_interval=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        use_amp=False,
        use_fused_adamw=False,
        embedding_dim=8,
        hidden_dim=12,
        memory_dim=8,
        partial_token_count=8,
        dropout=0.0,
        run_hold=False,
    )
    payload = run_document_compare(config)
    assert "cheap" in payload
    assert set(payload["cheap"]["results"].keys()) == {"doc_reset", "doc_carry_hidden"}
    assert "doc_carry_hidden" in json.dumps(payload)


def test_run_document_compare_single_variant_hold_only(tmp_path) -> None:
    cache_path = tmp_path / "doc_cache.pt"
    _make_doc_cache(cache_path)
    output_path = tmp_path / "doc_compare_hold.json"
    config = DocumentCompareConfig(
        cache_path=cache_path,
        output_path=output_path,
        target_tokens=56,
        cheap_target_tokens=56,
        seed=13,
        sequence_length=7,
        batch_size=2,
        eval_batch_size=1,
        eval_steps=1,
        eval_interval=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        use_amp=False,
        use_fused_adamw=False,
        embedding_dim=8,
        hidden_dim=12,
        memory_dim=8,
        partial_token_count=8,
        dropout=0.0,
        variant_mode="doc_carry_hidden",
        run_cheap=False,
        run_hold=True,
    )
    payload = run_document_compare(config)
    assert "cheap" not in payload
    assert set(payload["hold"]["results"].keys()) == {"doc_carry_hidden"}


def test_run_document_compare_sampling_smoke(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / "doc_cache.pt"
    _make_doc_cache(cache_path)
    output_path = tmp_path / "doc_compare_sample.json"

    class _FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2, 3, 4]}

        def decode(self, token_ids):
            return " ".join(str(token) for token in token_ids)

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeTokenizer()

    monkeypatch.setattr(module, "AutoTokenizer", _FakeAutoTokenizer)

    config = DocumentCompareConfig(
        cache_path=cache_path,
        output_path=output_path,
        target_tokens=56,
        cheap_target_tokens=56,
        seed=13,
        sequence_length=7,
        batch_size=2,
        eval_batch_size=1,
        eval_steps=1,
        eval_interval=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        use_amp=False,
        use_fused_adamw=False,
        embedding_dim=8,
        hidden_dim=12,
        memory_dim=8,
        partial_token_count=8,
        dropout=0.0,
        sample_prompt="i went to the mall because",
        sample_every_tokens=16,
        sample_generation_tokens=4,
        variant_mode="doc_reset",
        run_cheap=False,
        run_hold=True,
    )
    payload = run_document_compare(config)
    samples = payload["hold"]["results"]["doc_reset"]["samples"]
    assert len(samples) >= 1
    assert samples[0]["prompt"] == "i went to the mall because"
