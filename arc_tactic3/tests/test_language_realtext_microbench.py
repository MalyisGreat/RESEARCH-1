from __future__ import annotations

import numpy as np

from arc_tactic3 import language_realtext_microbench as realtext


def _dummy_dataset(sequence_count: int = 32, sequence_length: int = 8, vocab_size: int = 64) -> realtext.TokenBlockDataset:
    rows = []
    for row_index in range(sequence_count):
        start = 3 + row_index
        rows.append([(start + offset) % vocab_size for offset in range(sequence_length + 1)])
    blocks = realtext.torch.tensor(rows, dtype=realtext.torch.long)
    return realtext.TokenBlockDataset(blocks[:, :-1], blocks[:, 1:])


class _FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) % 17 for character in text]


class _BatchFakeTokenizer(_FakeTokenizer):
    def __init__(self) -> None:
        self.call_batch_sizes: list[int] = []

    def __call__(self, texts: list[str], add_special_tokens: bool = False) -> dict[str, list[list[int]]]:
        self.call_batch_sizes.append(len(texts))
        return {
            "input_ids": [self.encode(text, add_special_tokens=add_special_tokens) for text in texts],
        }


class _CountingTokenizer(_FakeTokenizer):
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.calls += 1
        return super().encode(text, add_special_tokens=add_special_tokens)


def test_build_models_are_fairly_matched() -> None:
    config = realtext.RealTextConfig(
        device="cpu",
        recurrent_embedding_dim=24,
        recurrent_hidden_dim=48,
        recurrent_memory_dim=24,
        gpt_d_model=24,
        gpt_heads=4,
        gpt_layers=2,
        gpt_ff_dim=96,
        sequence_length=16,
    )
    models = realtext.build_models(config, vocab_size=128)
    reports = {name: {"parameter_count": realtext.count_parameters(model)} for name, model in models.items()}
    fairness = realtext.fairness_summary(reports)
    assert fairness["parameter_gap_ok"] is True
    assert fairness["relative_parameter_gap"] <= 0.15


def test_stream_token_ids_respects_exact_cap() -> None:
    rows = [
        {"text": "abcdef"},
        {"text": "ghij"},
        {"text": "klmnop"},
    ]
    token_ids = realtext._stream_token_ids(
        rows,
        tokenizer=_FakeTokenizer(),
        text_column="text",
        token_cap=10,
    )
    assert isinstance(token_ids, np.ndarray)
    assert int(token_ids.size) == 10


def test_token_ids_to_blocks_discards_only_tail_remainder() -> None:
    token_ids = np.arange(20, dtype=np.uint32)
    dataset = realtext._token_ids_to_blocks(token_ids, sequence_length=4)
    assert len(dataset) == 4
    sample = dataset[0]
    assert sample["input_ids"].tolist() == [0, 1, 2, 3]
    assert sample["targets"].tolist() == [1, 2, 3, 4]


def test_texts_to_blocks_batch_tokenization_matches_encode_fallback() -> None:
    texts = ["alpha", "", "beta", " ", "gamma"]
    batch_tokenizer = _BatchFakeTokenizer()
    setattr(batch_tokenizer, "_codex_batch_size", 2)
    dataset_batched = realtext._texts_to_blocks(
        texts,
        tokenizer=batch_tokenizer,
        sequence_length=4,
        max_sequences=8,
    )
    dataset_fallback = realtext._texts_to_blocks(
        texts,
        tokenizer=_FakeTokenizer(),
        sequence_length=4,
        max_sequences=8,
    )
    assert realtext.torch.equal(dataset_batched.input_ids, dataset_fallback.input_ids)
    assert realtext.torch.equal(dataset_batched.targets, dataset_fallback.targets)


def test_texts_to_blocks_stops_early_once_block_budget_is_filled() -> None:
    texts = ["abcdefghij"] * 20
    tokenizer = _CountingTokenizer()
    dataset = realtext._texts_to_blocks(
        texts,
        tokenizer=tokenizer,
        sequence_length=4,
        max_sequences=2,
    )
    assert len(dataset) == 2
    assert tokenizer.calls < len(texts)


def test_effective_tokenization_batch_size_shrinks_small_runs() -> None:
    assert realtext._effective_tokenization_batch_size(requested_batch_size=256, max_sequences=64) == 32
    assert realtext._effective_tokenization_batch_size(requested_batch_size=256, max_sequences=512) == 64
    assert realtext._effective_tokenization_batch_size(requested_batch_size=256, max_sequences=2048) == 256


def test_texts_to_blocks_caps_batch_tokenization_work_for_small_runs() -> None:
    texts = ["abcdefghij"] * 100
    tokenizer = _BatchFakeTokenizer()
    setattr(tokenizer, "_codex_batch_size", 256)
    dataset = realtext._texts_to_blocks(
        texts,
        tokenizer=tokenizer,
        sequence_length=4,
        max_sequences=8,
    )
    assert len(dataset) == 8
    assert tokenizer.call_batch_sizes
    assert max(tokenizer.call_batch_sizes) <= 32


def test_load_realtext_datasets_uses_cache_without_touching_hub() -> None:
    train_dataset = _dummy_dataset(sequence_count=8, sequence_length=4, vocab_size=32)
    val_dataset = _dummy_dataset(sequence_count=4, sequence_length=4, vocab_size=32)
    payload = {
        "vocab_size": 32,
        "train_dataset": realtext._serialize_dataset(train_dataset),
        "val_dataset": realtext._serialize_dataset(val_dataset),
    }

    class _ExistingPath:
        def exists(self) -> bool:
            return True

    config = realtext.RealTextConfig(
        device="cpu",
        dataset_cache_path=_ExistingPath(),
    )

    original_tokenizer_loader = realtext.AutoTokenizer.from_pretrained
    original_dataset_loader = realtext.load_dataset
    original_torch_load = realtext.torch.load
    try:
        def _unexpected_tokenizer_loader(*args, **kwargs):
            raise AssertionError("Tokenizer loader should not be called when dataset cache is present.")

        def _unexpected_dataset_loader(*args, **kwargs):
            raise AssertionError("Dataset loader should not be called when dataset cache is present.")

        realtext.AutoTokenizer.from_pretrained = _unexpected_tokenizer_loader
        realtext.load_dataset = _unexpected_dataset_loader
        realtext.torch.load = lambda *args, **kwargs: payload
        loaded_train, loaded_val, vocab_size = realtext.load_realtext_datasets(config)
    finally:
        realtext.AutoTokenizer.from_pretrained = original_tokenizer_loader
        realtext.load_dataset = original_dataset_loader
        realtext.torch.load = original_torch_load

    assert vocab_size == 32
    assert realtext.torch.equal(loaded_train.input_ids, train_dataset.input_ids)
    assert realtext.torch.equal(loaded_val.targets, val_dataset.targets)


def test_resolved_dataset_cache_path_is_stable_for_same_config() -> None:
    config = realtext.RealTextConfig(
        device="cpu",
        local_files_only=True,
        sequence_length=16,
        max_train_sequences=32,
        max_eval_sequences=8,
    )
    path_one = realtext._resolved_dataset_cache_path(config)
    path_two = realtext._resolved_dataset_cache_path(config)
    assert path_one == path_two
    assert path_one is not None
    assert "dataset_cache" in str(path_one)


def test_load_realtext_datasets_iterates_rows_instead_of_materializing_text_column() -> None:
    class _FakeDataset(list):
        def __getitem__(self, item):
            if isinstance(item, str):
                raise AssertionError("Column materialization should not be used for small-run token packing.")
            return super().__getitem__(item)

    class _Tokenizer(_BatchFakeTokenizer):
        eos_token_id = 42
        vocab_size = 128

    original_tokenizer_loader = realtext.AutoTokenizer.from_pretrained
    original_dataset_loader = realtext.load_dataset
    try:
        realtext.AutoTokenizer.from_pretrained = lambda *args, **kwargs: _Tokenizer()
        realtext.load_dataset = lambda *args, split, **kwargs: _FakeDataset(
            [{"text": f"{split}-{index}"} for index in range(24)]
        )
        train_dataset, val_dataset, vocab_size = realtext.load_realtext_datasets(
            realtext.RealTextConfig(
                device="cpu",
                use_amp=False,
                sequence_length=4,
                max_train_sequences=4,
                max_eval_sequences=2,
                local_files_only=True,
            )
        )
    finally:
        realtext.AutoTokenizer.from_pretrained = original_tokenizer_loader
        realtext.load_dataset = original_dataset_loader

    assert vocab_size == 128
    assert len(train_dataset) == 4
    assert len(val_dataset) == 2


def test_train_microbenchmark_smoke_cpu() -> None:
    for tensor_batching in (False, True):
        config = realtext.RealTextConfig(
            device="cpu",
            use_amp=False,
            sequence_length=8,
            train_steps=2,
            eval_interval=1,
            batch_size=4,
            eval_batch_size=4,
            recurrent_embedding_dim=16,
            recurrent_hidden_dim=24,
            recurrent_memory_dim=16,
            gpt_d_model=16,
            gpt_heads=4,
            gpt_layers=1,
            gpt_ff_dim=80,
            tensor_batching=tensor_batching,
            cache_dataset_on_device=False,
            pin_memory=False,
        )
        train_dataset = _dummy_dataset(sequence_count=32, sequence_length=config.sequence_length, vocab_size=64)
        val_dataset = _dummy_dataset(sequence_count=16, sequence_length=config.sequence_length, vocab_size=64)
        models = realtext.build_models(config, vocab_size=64)
        for report in (
            realtext.train_microbenchmark(model, train_dataset, val_dataset, config=config)
            for model in models.values()
        ):
            assert report["parameter_count"] > 0
            assert report["initial_val_loss"] >= 0.0
            assert report["final_val_loss"] >= 0.0
            assert len(report["history"]) == 3
            assert report["history"][1]["tokens_seen"] > 0.0
            assert report["training_runtime_seconds"] >= 0.0


def test_train_batch_schedule_is_reproducible_and_reshuffles_epochs() -> None:
    schedule_one = realtext._build_train_batch_schedule(16, batch_size=4, steps=6, seed=11, drop_last=True)
    schedule_two = realtext._build_train_batch_schedule(16, batch_size=4, steps=6, seed=11, drop_last=True)
    schedule_three = realtext._build_train_batch_schedule(16, batch_size=4, steps=6, seed=12, drop_last=True)

    assert len(schedule_one) == 6
    assert all(batch.numel() == 4 for batch in schedule_one)
    assert all(realtext.torch.equal(left, right) for left, right in zip(schedule_one, schedule_two, strict=True))
    assert any(not realtext.torch.equal(left, right) for left, right in zip(schedule_one, schedule_three, strict=True))
    first_epoch = realtext.torch.cat(schedule_one[:4])
    second_epoch = realtext.torch.cat(schedule_one[4:6])
    assert first_epoch.numel() == 16
    assert len(realtext.torch.unique(first_epoch)) == 16
    assert not realtext.torch.equal(second_epoch, first_epoch[: second_epoch.numel()])


def test_train_batch_schedule_caching_returns_clones() -> None:
    params = dict(total_examples=16, batch_size=4, steps=4, seed=99, drop_last=True)
    schedule_a = realtext._build_train_batch_schedule(**params)
    schedule_b = realtext._build_train_batch_schedule(**params)
    assert schedule_a is not schedule_b
    assert schedule_a[0] is not schedule_b[0]
    assert realtext.torch.equal(schedule_a[0], schedule_b[0])
    schedule_a[0][0] = -1
    assert schedule_b[0][0] != -1


def test_associative_recurrent_scatter_add_matches_naive_votes() -> None:
    model = realtext.AssociativeRecurrentLM(
        vocab_size=32,
        embedding_dim=8,
        hidden_dim=12,
        memory_dim=8,
        dropout=0.0,
        max_length=8,
    )
    model.eval()
    input_ids = realtext.torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=realtext.torch.long)
    with realtext.torch.no_grad():
        embeddings = model.embedding(input_ids)
        states, _ = model.encoder(embeddings)
        base_logits = realtext.F.linear(model.hidden_to_embedding(states), model.embedding.weight, model.output_bias)
        query_keys = model.query_proj(states)
        memory_keys = model.key_proj(states)
        scores = realtext.torch.matmul(query_keys, memory_keys.transpose(1, 2)) / np.sqrt(query_keys.size(-1))
        causal_mask = model._causal_mask[:, : input_ids.size(1), : input_ids.size(1)]
        scores = scores.masked_fill(~causal_mask, realtext.torch.finfo(scores.dtype).min)
        attention = realtext.torch.softmax(scores, dim=-1)
        attention = attention * causal_mask
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        memory_votes = realtext.torch.zeros_like(base_logits)
        value_index = input_ids.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        memory_votes.scatter_add_(2, value_index, attention)
        gate = realtext.torch.sigmoid(model.gate(states))
        naive_logits = base_logits + gate * model.memory_scale * memory_votes
        fast_logits = model(input_ids)
    assert realtext.torch.allclose(fast_logits, naive_logits, atol=1e-6, rtol=1e-5)


def test_associative_recurrent_forward_runs_under_autocast_without_dtype_error() -> None:
    device = "cuda" if realtext.torch.cuda.is_available() else "cpu"
    autocast_enabled = realtext.torch.cuda.is_available()
    model = realtext.AssociativeRecurrentLM(
        vocab_size=64,
        embedding_dim=16,
        hidden_dim=24,
        memory_dim=16,
        dropout=0.0,
        max_length=8,
    ).to(device)
    input_ids = realtext.torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=realtext.torch.long, device=device)
    with realtext.torch.autocast(device_type=device, enabled=autocast_enabled):
        logits = model(input_ids)
    assert logits.shape == (1, 8, 64)


def test_gpt2_causal_lm_cached_masks_match_manual_forward() -> None:
    model = realtext.GPT2CausalLM(
        vocab_size=64,
        d_model=16,
        n_heads=4,
        layers=1,
        ff_dim=48,
        dropout=0.0,
        max_length=8,
    )
    model.eval()
    input_ids = realtext.torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=realtext.torch.long)
    with realtext.torch.no_grad():
        positions = realtext.torch.arange(input_ids.size(1), dtype=realtext.torch.long).unsqueeze(0).expand_as(input_ids)
        states = model.embedding(input_ids) + model.position_embedding(positions)
        causal_mask = realtext.torch.triu(
            realtext.torch.ones((input_ids.size(1), input_ids.size(1)), dtype=realtext.torch.bool),
            diagonal=1,
        )
        for block in model.blocks:
            states = block(states, causal_mask=causal_mask, key_padding_mask=None)
        manual_logits = model.output(model.final_norm(states))
        cached_logits = model(input_ids)
    assert realtext.torch.allclose(cached_logits, manual_logits, atol=1e-6, rtol=1e-5)


def test_load_realtext_datasets_uses_local_cache() -> None:
    train_dataset = _dummy_dataset(sequence_count=16, sequence_length=8, vocab_size=64)
    val_dataset = _dummy_dataset(sequence_count=8, sequence_length=8, vocab_size=64)
    payload = {
        "vocab_size": 64,
        "train_dataset": realtext._serialize_dataset(train_dataset),
        "val_dataset": realtext._serialize_dataset(val_dataset),
    }

    class _ExistingPath:
        def exists(self) -> bool:
            return True

    original_torch_load = realtext.torch.load
    try:
        realtext.torch.load = lambda *args, **kwargs: payload
        config = realtext.RealTextConfig(
            device="cpu",
            use_amp=False,
            sequence_length=8,
            dataset_cache_path=_ExistingPath(),
        )
        loaded_train, loaded_val, vocab_size = realtext.load_realtext_datasets(config)
        assert vocab_size == 64
        assert realtext.torch.equal(loaded_train.input_ids, train_dataset.input_ids)
        assert realtext.torch.equal(loaded_train.targets, train_dataset.targets)
        assert realtext.torch.equal(loaded_val.input_ids, val_dataset.input_ids)
        assert realtext.torch.equal(loaded_val.targets, val_dataset.targets)
    finally:
        realtext.torch.load = original_torch_load


def test_run_realtext_microbenchmark_payload_without_network() -> None:
    config = realtext.RealTextConfig(
        device="cpu",
        use_amp=False,
        sequence_length=8,
        train_steps=2,
        eval_interval=1,
        batch_size=4,
        eval_batch_size=4,
        recurrent_embedding_dim=16,
        recurrent_hidden_dim=24,
        recurrent_memory_dim=16,
        gpt_d_model=16,
        gpt_heads=4,
        gpt_layers=1,
        gpt_ff_dim=80,
    )

    original_loader = realtext.load_realtext_datasets
    try:
        def _fake_loader(_config: realtext.RealTextConfig):
            return (
                _dummy_dataset(sequence_count=32, sequence_length=_config.sequence_length, vocab_size=64),
                _dummy_dataset(sequence_count=16, sequence_length=_config.sequence_length, vocab_size=64),
                64,
            )

        realtext.load_realtext_datasets = _fake_loader
        payload = realtext.run_realtext_microbenchmark(config)
    finally:
        realtext.load_realtext_datasets = original_loader

    assert payload["benchmark"] == "language_realtext_microbench"
    assert payload["winner_by_final_val_loss"] in {"associative_recurrent", "gpt2_like"}
    assert payload["fairness"]["parameter_gap_ok"] is True
    assert payload["dataset"]["sequence_length"] == 8
    assert payload["dataset_loading_runtime_seconds"] >= 0.0
    for model_report in payload["models"].values():
        assert model_report["final_val_loss"] >= 0.0


def test_load_realtext_datasets_passes_local_files_only_to_datasets_loader() -> None:
    config = realtext.RealTextConfig(
        device="cpu",
        local_files_only=True,
        max_train_sequences=4,
        max_eval_sequences=2,
        sequence_length=4,
    )

    class _FakeTokenizer:
        eos_token_id = 0
        vocab_size = 32

        def __call__(self, texts, add_special_tokens=False):
            del add_special_tokens
            return {"input_ids": [[1, 2, 3] for _ in texts]}

        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return [1, 2, 3]

    captured_download_flags: list[bool] = []

    def _fake_dataset_loader(*args, **kwargs):
        del args
        download_config = kwargs.get("download_config")
        captured_download_flags.append(bool(download_config.local_files_only))
        return [{"text": "abc"}] * 8

    original_tokenizer_loader = realtext.AutoTokenizer.from_pretrained
    original_dataset_loader = realtext.load_dataset
    original_cache_resolver = realtext._resolved_dataset_cache_path
    try:
        realtext.AutoTokenizer.from_pretrained = lambda *args, **kwargs: _FakeTokenizer()
        realtext.load_dataset = _fake_dataset_loader
        realtext._resolved_dataset_cache_path = lambda _config: None
        realtext.load_realtext_datasets(config)
    finally:
        realtext.AutoTokenizer.from_pretrained = original_tokenizer_loader
        realtext.load_dataset = original_dataset_loader
        realtext._resolved_dataset_cache_path = original_cache_resolver

    assert captured_download_flags == [True, True]
