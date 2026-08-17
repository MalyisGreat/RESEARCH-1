from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
LANGUAGE = ROOT.parent
SEARCH = LANGUAGE / "training_system_search_20260712"
sys.path.insert(0, str(SEARCH))
sys.path.insert(0, str(LANGUAGE / "local_optimization_tests_20260816"))

import evaluate_repetition as evaluation
import phrase_induction_train as phrase
import phrase_semantic_induction_train as semantic


SEEDS = (13, 17, 23, 29, 31)
CONDITIONS = {
    "phrase23": (phrase.PhraseInductionModel, "2,3"),
    "phrase234": (phrase.PhraseInductionModel, "2,3,4"),
    "phrase23_semantic": (semantic.PhraseSemanticInductionModel, "2,3"),
}


def main() -> None:
    import tiktoken

    device = torch.device("cuda")
    encoding = tiktoken.get_encoding("gpt2")

    class Tokenizer:
        encode = staticmethod(encoding.encode)
        decode = staticmethod(encoding.decode)

    payload: dict[str, object] = {"conditions": {}}
    for condition, (model_class, orders) in CONDITIONS.items():
        condition_payload = {}
        for seed in SEEDS:
            os.environ["PHRASE_ORDERS"] = orders
            checkpoint = ROOT / f"{condition}_seed{seed}" / "checkpoint.pt"
            model = evaluation.load_model(model_class, checkpoint, device, orders)
            samples = [
                evaluation.generate(model, Tokenizer(), prompt, 100 + index, device)
                for index, prompt in enumerate(evaluation.PROMPTS)
            ]
            aggregate = {
                metric: sum(sample["metrics"][metric] for sample in samples) / len(samples)
                for metric in samples[0]["metrics"]
            }
            condition_payload[str(seed)] = {"aggregate": aggregate, "samples": samples}
            print(condition, seed, aggregate, flush=True)
            del model
            torch.cuda.empty_cache()
        payload["conditions"][condition] = condition_payload
    (ROOT / "repetition_matrix.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
