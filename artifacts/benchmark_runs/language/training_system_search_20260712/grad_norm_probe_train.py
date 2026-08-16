from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import rotating_anchor_collapsed_train as experiment


def main() -> None:
    output_arg = sys.argv[sys.argv.index("--output-dir") + 1]
    output_dir = Path(output_arg)
    norms: list[float] = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def measured_clip(parameters, max_norm, *args, **kwargs):
        norm = original_clip(parameters, max_norm, *args, **kwargs)
        norms.append(float(norm.detach()))
        return norm

    torch.nn.utils.clip_grad_norm_ = measured_clip
    try:
        experiment.trainer.train(experiment.trainer.parse_args())
    finally:
        output_dir.mkdir(parents=True, exist_ok=True)
        finite = [value for value in norms if value == value and abs(value) != float("inf")]
        payload = {
            "count": len(norms),
            "finite_count": len(finite),
            "minimum": min(finite) if finite else None,
            "maximum": max(finite) if finite else None,
            "mean": sum(finite) / len(finite) if finite else None,
            "above_one": sum(value > 1.0 for value in finite),
            "values": norms,
        }
        (output_dir / "grad_norms.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"GRAD_NORMS {json.dumps({key: value for key, value in payload.items() if key != 'values'})}", flush=True)


if __name__ == "__main__":
    main()
