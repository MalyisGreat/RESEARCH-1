from __future__ import annotations

import torch

import rotating_anchor_collapsed_train as experiment


def no_clip(parameters, max_norm, *args, **kwargs):
    del parameters, max_norm, args, kwargs
    return torch.zeros((), device="cuda")


torch.nn.utils.clip_grad_norm_ = no_clip


if __name__ == "__main__":
    experiment.trainer.train(experiment.trainer.parse_args())
