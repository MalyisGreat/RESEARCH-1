from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


SEARCH_DIR = Path(__file__).parents[1] / "training_system_search_20260712"
sys.path.insert(0, str(SEARCH_DIR))

import phrase_induction_train as phrase
import phrase_semantic_induction_train as phrase_semantic
import bounded_phrase_recall_train as bounded_phrase
import confidence_semantic_retrieval_train as confidence_semantic
import hidden_phrase_retrieval_train as hidden_phrase
import hidden_phrase_diagonal_train as hidden_phrase_diagonal


PROMPTS = (
    "The most important thing",
    "In the year 2050",
    "Once upon a time",
    "The computer",
    "Question: What is machine learning? Answer:",
    "Python function to add two numbers:\n",
)


def load_model(model_class, checkpoint_path: Path, device: torch.device, phrase_orders: str):
    os.environ["PHRASE_ORDERS"] = phrase_orders
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = phrase.experiment.experiment.experiment.trainer.TrainConfig(**checkpoint["config"])
    model = model_class(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def ngrams(tokens: list[int], order: int) -> list[tuple[int, ...]]:
    return [tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1)]


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, seed: int, device: torch.device) -> dict[str, object]:
    ids = torch.tensor([tokenizer.encode(prompt)], device=device, dtype=torch.long)
    generated: list[int] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    vocabulary = torch.arange(model.vocab_size, device=device)
    for _ in range(64):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            hidden = model.factor_down(model.features(ids))
            position = torch.tensor([ids.size(1) - 1], device=device)
            logits = model.candidate_logits(
                hidden=hidden[:, -1:],
                input_ids=ids,
                candidate_ids=vocabulary,
                weight=model.factor_up.weight,
                bias=model.factor_up.bias,
                positions=position,
            )[0, 0]
        top_values, top_indices = torch.topk(logits.float() / 0.8, 40)
        probabilities = torch.softmax(top_values, dim=-1)
        selected = torch.multinomial(probabilities, 1, generator=generator)
        next_id = top_indices[selected]
        ids = torch.cat((ids, next_id.view(1, 1)), dim=1)
        generated.append(int(next_id))

    metrics: dict[str, float] = {}
    for order in (1, 2, 3, 4):
        spans = ngrams(generated, order)
        metrics[f"distinct_{order}"] = len(set(spans)) / max(len(spans), 1)
        metrics[f"repeat_{order}"] = 1.0 - metrics[f"distinct_{order}"]
    return {
        "prompt": prompt,
        "text": tokenizer.decode(ids[0].tolist()),
        "metrics": metrics,
    }


def main() -> None:
    import tiktoken

    device = torch.device("cuda")
    encoding = tiktoken.get_encoding("gpt2")

    class Tokenizer:
        encode = staticmethod(encoding.encode)
        decode = staticmethod(encoding.decode)

    tokenizer = Tokenizer()
    root = Path(__file__).parent
    variants = {
        "phrase23": (
            phrase.PhraseInductionModel,
            root / "phrase23_reference_10m_seed13" / "checkpoint.pt",
            "2,3",
        ),
        "phrase23_semantic": (
            phrase_semantic.PhraseSemanticInductionModel,
            root / "phrase23_semantic_10m_seed13" / "checkpoint.pt",
            "2,3",
        ),
        "bounded_phrase23": (
            bounded_phrase.BoundedPhraseRecallModel,
            root / "bounded_phrase23_10m_seed13" / "checkpoint.pt",
            "2,3",
        ),
        "confidence_semantic_phrase23": (
            confidence_semantic.ConfidenceSemanticRetrievalModel,
            root / "confidence_semantic_phrase23_10m_seed13" / "checkpoint.pt",
            "2,3",
        ),
        "confidence_semantic_soft_phrase23": (
            confidence_semantic.ConfidenceSemanticRetrievalModel,
            root / "confidence_semantic_soft_phrase23_10m_seed13" / "checkpoint.pt",
            "2,3",
        ),
        "hidden_phrase23": (
            hidden_phrase.HiddenPhraseRetrievalModel,
            root / "hidden_phrase23_10m_seed13" / "checkpoint.pt",
            "2,3",
        ),
        "hidden_phrase23_diagonal": (
            hidden_phrase_diagonal.HiddenPhraseDiagonalModel,
            root / "hidden_phrase23_diagonal_10m_seed13" / "checkpoint.pt",
            "2,3",
        ),
    }
    payload: dict[str, object] = {"variants": {}}
    for name, (model_class, checkpoint_path, phrase_orders) in variants.items():
        model = load_model(model_class, checkpoint_path, device, phrase_orders)
        if name == "confidence_semantic_soft_phrase23":
            model.retrieval_confidence_threshold = 0.30
            model.retrieval_confidence_temperature = 0.10
            model.retrieval_entropy_threshold = 0.50
            model.retrieval_entropy_temperature = 0.10
        samples = [
            generate(model, tokenizer, prompt, 100 + index, device)
            for index, prompt in enumerate(PROMPTS)
        ]
        aggregate = {
            metric: sum(sample["metrics"][metric] for sample in samples) / len(samples)
            for metric in samples[0]["metrics"]
        }
        payload["variants"][name] = {"aggregate": aggregate, "samples": samples}
        del model
        torch.cuda.empty_cache()

    output = root / "repetition_evaluation.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: data["aggregate"] for name, data in payload["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
