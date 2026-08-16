import math
import os

import standalone_longseq_anchor_train as trainer


def _cosine(step: int, *, warmup: int, target: int, peak: float, floor: float) -> float:
    if step <= 0:
        return floor
    if warmup > 0 and step <= warmup:
        return peak * (step / warmup)
    progress = min(max((step - warmup) / max(target - warmup, 1), 0.0), 1.0)
    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (peak - floor) * decay


def _linear_warm_cosine_local(
    local_step: int,
    *,
    warmup: int,
    decay_steps: int,
    start: float,
    peak: float,
    floor: float,
) -> float:
    local_step = max(0, local_step)
    if warmup > 0 and local_step <= warmup:
        return start + (peak - start) * (local_step / warmup)
    progress = min(max((local_step - warmup) / max(decay_steps - warmup, 1), 0.0), 1.0)
    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (peak - floor) * decay


def scheduled_learning_rate(config: trainer.TrainConfig, step: int) -> float:
    mode = os.environ.get("LR_PROBE_MODE", "control_smart_decay")
    resume_step = int(os.environ.get("LR_PROBE_RESUME_STEP", "196860"))
    additional_steps = int(os.environ.get("LR_PROBE_ADDITIONAL_STEPS", "9843"))
    original_target = int(os.environ.get("LR_PROBE_ORIGINAL_TARGET_STEPS", "492126"))
    local_step = max(0, step - resume_step)

    if mode == "control_smart_decay":
        return _cosine(step, warmup=2000, target=original_target, peak=3e-4, floor=1e-5)

    if mode == "high_floor_1e4":
        return _cosine(step, warmup=2000, target=original_target, peak=3e-4, floor=1e-4)

    if mode == "flat_2e4_slow_decay":
        return _linear_warm_cosine_local(
            local_step,
            warmup=500,
            decay_steps=additional_steps * 4,
            start=1e-4,
            peak=2e-4,
            floor=1.5e-4,
        )

    if mode == "aggressive_4e4_floor_1e4":
        return _linear_warm_cosine_local(
            local_step,
            warmup=500,
            decay_steps=additional_steps,
            start=1e-4,
            peak=4e-4,
            floor=1e-4,
        )

    raise ValueError(f"unknown LR_PROBE_MODE {mode!r}")


trainer.scheduled_learning_rate = scheduled_learning_rate


if __name__ == "__main__":
    trainer.train(trainer.parse_args())
