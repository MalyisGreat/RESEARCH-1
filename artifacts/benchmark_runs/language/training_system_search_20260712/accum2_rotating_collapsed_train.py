from __future__ import annotations

import torch

import rotating_anchor_collapsed_train as experiment


state = {"micro_step": 0}
original_zero_grad = torch.optim.AdamW.zero_grad
original_scaler_unscale = torch.amp.GradScaler.unscale_
original_scaler_step = torch.amp.GradScaler.step
original_scaler_update = torch.amp.GradScaler.update
original_clip = torch.nn.utils.clip_grad_norm_


def is_update_microstep() -> bool:
    return (state["micro_step"] + 1) % 2 == 0


def accumulated_zero_grad(optimizer, *args, **kwargs):
    if state["micro_step"] % 2 == 0:
        return original_zero_grad(optimizer, *args, **kwargs)
    return None


def accumulated_unscale(scaler, optimizer):
    if is_update_microstep():
        return original_scaler_unscale(scaler, optimizer)
    return None


def accumulated_clip(parameters, max_norm, *args, **kwargs):
    if is_update_microstep():
        return original_clip(parameters, max_norm, *args, **kwargs)
    return torch.zeros((), device="cuda")


def accumulated_step(scaler, optimizer, *args, **kwargs):
    if is_update_microstep():
        result = original_scaler_step(scaler, optimizer, *args, **kwargs)
    else:
        result = None
    state["micro_step"] += 1
    return result


def accumulated_update(scaler, *args, **kwargs):
    if state["micro_step"] % 2 == 0:
        return original_scaler_update(scaler, *args, **kwargs)
    return None


torch.optim.AdamW.zero_grad = accumulated_zero_grad
torch.amp.GradScaler.unscale_ = accumulated_unscale
torch.amp.GradScaler.step = accumulated_step
torch.amp.GradScaler.update = accumulated_update
torch.nn.utils.clip_grad_norm_ = accumulated_clip


if __name__ == "__main__":
    experiment.trainer.train(experiment.trainer.parse_args())
