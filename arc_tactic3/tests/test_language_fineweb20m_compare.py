from __future__ import annotations

import pytest

from arc_tactic3.language_fineweb20m_compare import (
    FineWebCompareConfig,
    _config_payload,
    _buffer_to_dataset,
    _fill_token_buffer,
    _validate_token_budget,
)


class _FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) % 17 + 3 for char in text]


def test_validate_token_budget_requires_exact_block_alignment() -> None:
    _validate_token_budget(
        FineWebCompareConfig(
            total_tokens=20_000_000,
            train_tokens=18_000_000,
            val_tokens=2_000_000,
            sequence_length=127,
        )
    )
    with pytest.raises(ValueError):
        _validate_token_budget(
            FineWebCompareConfig(
                total_tokens=20_000_001,
                train_tokens=18_000_000,
                val_tokens=2_000_001,
                sequence_length=127,
            )
        )


def test_fill_token_buffer_caps_exact_token_count() -> None:
    tokenizer = _FakeTokenizer()
    texts = ["alpha", "beta", "gamma", "delta"]
    buffer = _fill_token_buffer(texts, tokenizer=tokenizer, total_tokens=10)
    assert buffer.numel() == 10
    assert str(buffer.dtype) == "torch.int32"


def test_buffer_to_dataset_builds_fixed_blocks() -> None:
    token_buffer = __import__("torch").arange(24, dtype=__import__("torch").int32)
    dataset = _buffer_to_dataset(token_buffer, sequence_length=5)
    assert len(dataset) == 4
    first = dataset[0]
    assert first["input_ids"].tolist() == [0, 1, 2, 3, 4]
    assert first["targets"].tolist() == [1, 2, 3, 4, 5]


def test_config_payload_serializes_cache_path() -> None:
    config = FineWebCompareConfig(cache_path=__import__("pathlib").Path("cache.pt"))
    payload = _config_payload(config)
    assert payload["cache_path"] == "cache.pt"
