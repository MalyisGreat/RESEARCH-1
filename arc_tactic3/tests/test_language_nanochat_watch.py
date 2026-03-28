from __future__ import annotations

import json
from pathlib import Path

import torch

from arc_tactic3.language_nanochat_watch import NanochatWatchConfig, run_nanochat_watch, watch_nanochat_run


def test_nanochat_watch_cpu_smoke(tmp_path: Path) -> None:
    sequence_length = 7
    block_size = sequence_length + 1
    vocab_size = 32
    train_tokens = torch.randint(0, vocab_size, (block_size * 32,))
    val_tokens = torch.randint(0, vocab_size, (block_size * 8,))
    cache_path = tmp_path / "tiny_cache.pt"
    output_dir = tmp_path / "watch_run"
    torch.save(
        {
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "vocab_size": vocab_size,
        },
        cache_path,
    )
    config = NanochatWatchConfig(
        cache_path=cache_path,
        output_dir=output_dir,
        target_tokens=sequence_length * 4 * 2,
        seed=7,
        sequence_length=sequence_length,
        batch_size=4,
        eval_batch_size=4,
        eval_interval=1,
        log_interval=1,
        device="cpu",
        use_amp=False,
        pin_memory=False,
        use_fused_adamw=False,
        cache_dataset_on_device=False,
        val_blocks=4,
        nano_n_layer=2,
        nano_n_head=2,
        nano_n_kv_head=2,
        nano_n_embd=16,
        initial_eval=True,
    )
    payload = run_nanochat_watch(config, print_progress=False)
    assert payload["benchmark"] == "language_nanochat_watch"
    assert payload["report"]["train_tokens_seen"] >= config.target_tokens
    state = json.loads((output_dir / "state.json").read_text())
    assert state["status"] == "completed"
    final_payload = json.loads((output_dir / "final.json").read_text())
    assert final_payload["report"]["final_val_loss"] > 0.0
    assert (output_dir / "metrics.jsonl").exists()


def test_watch_nanochat_run_once(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "watch_dir"
    output_dir.mkdir()
    (output_dir / "state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "step": 12,
                "train_steps": 100,
                "progress": 0.12,
                "tokens_seen": 12345,
                "target_tokens": 100000,
                "train_tok_per_sec": 2222.2,
                "pure_train_tok_per_sec": 3333.3,
                "eta_seconds": 45.0,
                "latest_train_loss": 1.2345,
                "latest_val_loss": 2.3456,
                "peak_vram_mb": 987.6,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "metrics.jsonl").write_text(
        json.dumps({"kind": "eval", "step": 10, "tokens_seen": 10000, "val_loss": 2.5, "train_loss": 1.5}) + "\n",
        encoding="utf-8",
    )
    watch_nanochat_run(output_dir, once=True)
    output = capsys.readouterr().out
    assert "nanochat_50m" in output
    assert "12.0%" in output
    assert "latest eval:" in output
