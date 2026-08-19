from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
LANGUAGE_DIR = THIS_DIR.parent
H100_DIR = LANGUAGE_DIR / "h100_wave10_350m_fullvocab_20260616"
for path in (LANGUAGE_DIR, H100_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import standalone_longseq_anchor_train as base
import h100_wave10_fullvocab_train as h100
from deep_wave_delta import DeepWaveDeltaConfig, DeepWaveDeltaLM, config_from_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Wave-Delta H100 trainer with exact full-vocab validation")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--preset", choices=("10m", "100m", "350m"), default="350m")
    parser.add_argument("--target-tokens", type=int, default=100_000_000)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=2_048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=2_500)
    parser.add_argument("--milestone-checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--val-blocks", type=int, default=32)
    parser.add_argument("--sampled-vocab-size", type=int, default=32_768)
    parser.add_argument("--token-stride", type=int, default=4)
    parser.add_argument("--token-chunk-size", type=int, default=4_096)
    parser.add_argument("--full-eval-token-chunk-size", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--candidate-ids-path", type=Path, default=None)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--timing-warmup-steps", type=int, default=10)
    parser.add_argument("--loss-kernel", choices=("torch", "liger"), default="torch")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="max-autotune-no-cudagraphs")
    parser.add_argument("--cache-on-device", action="store_true")
    parser.add_argument("--cache-mmap", action="store_true")
    parser.add_argument("--skip-checkpoints", action="store_true")
    parser.add_argument("--save-final-checkpoint-only", action="store_true")
    parser.add_argument("--final-weights-only", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--recurrent-layers", type=int, default=None)
    parser.add_argument("--recurrent-heads", type=int, default=None)
    parser.add_argument("--key-dim", type=int, default=None)
    parser.add_argument("--value-dim", type=int, default=None)
    parser.add_argument("--local-memory-rank", type=int, default=None)
    parser.add_argument("--local-memory-kernel", type=int, default=128)
    parser.add_argument("--output-rank", type=int, default=None)
    return parser.parse_args()


def architecture_config(args: argparse.Namespace) -> DeepWaveDeltaConfig:
    overrides = {
        "vocab_size": 50_257,
        "dropout": args.dropout,
        "local_memory_kernel": args.local_memory_kernel,
        "use_fused_delta": True,
    }
    for argument, field in (
        ("dim", "dim"),
        ("depth", "depth"),
        ("recurrent_layers", "recurrent_layers"),
        ("recurrent_heads", "recurrent_heads"),
        ("key_dim", "key_dim"),
        ("value_dim", "value_dim"),
        ("local_memory_rank", "local_memory_rank"),
        ("output_rank", "output_rank"),
    ):
        value = getattr(args, argument)
        if value is not None:
            overrides[field] = value
    return config_from_preset(args.preset, **overrides)


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        # The shared H100 runner selects its device through this process-local probe.
        # This makes end-to-end CI possible without changing the production trainer.
        torch.cuda.is_available = lambda: False
    architecture = architecture_config(args)
    train_steps = args.train_steps or math.ceil(
        args.target_tokens / (args.batch_size * args.sequence_length)
    )
    config = base.TrainConfig(
        cache_path=args.cache_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        train_steps=train_steps,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        milestone_checkpoint_interval=args.milestone_checkpoint_interval,
        val_blocks=args.val_blocks,
        seed=args.seed,
        embedding_dim=architecture.dim,
        block_type="deep_wave_delta",
        conv_layers=architecture.depth,
        conv_rank=architecture.output_rank,
        memory_rank=architecture.local_memory_rank,
        landmark_stride=architecture.local_memory_kernel,
        sampled_vocab_size=args.sampled_vocab_size,
        token_stride=args.token_stride,
        token_chunk_size=args.token_chunk_size,
        full_eval_token_chunk_size=args.full_eval_token_chunk_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        amp_dtype=args.amp_dtype,
        resume_checkpoint=args.resume_checkpoint,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.write_json_atomic(
        args.output_dir / "deep_wave_delta_config.json",
        {
            "architecture": architecture.to_dict(),
            "parameter_report": DeepWaveDeltaLM(architecture).parameter_report(),
            "scientific_invariants": {
                "training_sequence_complexity": "O(sequence_length)",
                "incremental_inference_context_complexity": "O(1)",
                "full_sequence_attention": False,
                "local_fir_retained": True,
                "persistent_delta_state": True,
            },
        },
    )

    # Reuse the hardened data, loss, checkpoint, and exact-validation machinery.
    # The adapter is process-local and does not alter the existing Wave trainer.
    base.CausalConvFactorizedLM = lambda ignored: DeepWaveDeltaLM(architecture)
    h100.apply_architecture = lambda model, ignored: model
    h100.train(
        config,
        log_interval=args.log_interval,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        collapsed_conv=True,
        legacy_candidate_path=False,
        save_checkpoints=not args.skip_checkpoints,
        candidate_ids_path=args.candidate_ids_path,
        timing_warmup_steps=args.timing_warmup_steps,
        loss_kernel=args.loss_kernel,
        profile_steps=args.profile_steps,
        architecture="deep_wave_delta",
        cache_on_device=args.cache_on_device,
        cache_mmap=args.cache_mmap,
        save_final_checkpoint_only=args.save_final_checkpoint_only,
        final_weights_only=args.final_weights_only,
    )


if __name__ == "__main__":
    main()
