from __future__ import annotations

import json
from pathlib import Path

import torch

from arc_tactic3 import language_docaware_head_to_head as module
from arc_tactic3.language_docaware_head_to_head import (
    DocAwareHeadToHeadConfig,
    _build_factorized_model,
    _build_nanochat_model,
    _build_partial_model,
    _build_untied_model,
    run_docaware_head_to_head,
)
from arc_tactic3.language_fastlearn_benchmark import count_parameters


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


def test_default_model_sizes_are_near_20m() -> None:
    config = DocAwareHeadToHeadConfig(cache_path=Path("cache.pt"), output_dir=Path("runs"))
    partial = _build_partial_model(config, vocab_size=50_257, partial_token_ids=torch.arange(config.partial_token_count))
    untied = _build_untied_model(config, vocab_size=50_257)
    factorized = _build_factorized_model(config, vocab_size=50_257)
    nano = _build_nanochat_model(config, vocab_size=50_257)
    assert 19_000_000 <= count_parameters(partial) <= 21_000_000
    assert 19_000_000 <= count_parameters(untied) <= 21_000_000
    assert 19_000_000 <= count_parameters(factorized) <= 21_000_000
    assert 19_000_000 <= count_parameters(nano) <= 21_000_000


def test_run_docaware_head_to_head_smoke(tmp_path) -> None:
    cache_path = tmp_path / "doc_cache.pt"
    _make_doc_cache(cache_path)
    output_dir = tmp_path / "runs"
    config = DocAwareHeadToHeadConfig(
        cache_path=cache_path,
        output_dir=output_dir,
        target_tokens=112,
        seed=13,
        sequence_length=7,
        score_target_loss=100.0,
        model_filter="partial_untied_20m",
        eval_interval=2,
        log_interval=1,
        device="cpu",
        use_amp=False,
        use_fused_adamw=False,
        sample_prompt=None,
        partial_batch_size=2,
        partial_eval_batch_size=1,
        partial_learning_rate=1e-3,
        partial_weight_decay=0.0,
        partial_embedding_dim=8,
        partial_hidden_dim=12,
        partial_memory_dim=8,
        partial_token_count=8,
        nano_batch_size=2,
        nano_eval_batch_size=1,
        nano_learning_rate=1e-3,
        nano_weight_decay=0.0,
        nano_n_layer=2,
        nano_n_head=2,
        nano_n_kv_head=2,
        nano_n_embd=8,
        dropout=0.0,
    )
    payload = run_docaware_head_to_head(config)
    assert payload["status"] == "completed"
    assert "partial_untied_20m" in payload["models"]
    final_path = output_dir / "partial_untied_20m" / "final.json"
    assert final_path.exists()
    saved = json.loads(final_path.read_text(encoding="utf-8"))
    assert saved["model_name"] == "partial_untied_20m"


def test_run_docaware_head_to_head_sampling_smoke(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / "doc_cache.pt"
    _make_doc_cache(cache_path)
    output_dir = tmp_path / "runs_sample"

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

    config = DocAwareHeadToHeadConfig(
        cache_path=cache_path,
        output_dir=output_dir,
        target_tokens=112,
        seed=13,
        sequence_length=7,
        score_target_loss=100.0,
        model_filter="partial_untied_20m",
        eval_interval=2,
        log_interval=1,
        device="cpu",
        use_amp=False,
        use_fused_adamw=False,
        sample_prompt="i went to the mall because",
        sample_every_tokens=16,
        sample_generation_tokens=4,
        partial_batch_size=2,
        partial_eval_batch_size=1,
        partial_learning_rate=1e-3,
        partial_weight_decay=0.0,
        partial_embedding_dim=8,
        partial_hidden_dim=12,
        partial_memory_dim=8,
        partial_token_count=8,
        nano_batch_size=2,
        nano_eval_batch_size=1,
        nano_learning_rate=1e-3,
        nano_weight_decay=0.0,
        nano_n_layer=2,
        nano_n_head=2,
        nano_n_kv_head=2,
        nano_n_embd=8,
        dropout=0.0,
    )
    payload = run_docaware_head_to_head(config)
    samples = payload["models"]["partial_untied_20m"]["samples"]
    assert len(samples) >= 1
    assert samples[0]["prompt"] == "i went to the mall because"
