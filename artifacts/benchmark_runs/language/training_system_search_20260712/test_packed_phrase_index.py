from __future__ import annotations

import torch

from packed_phrase_induction_train import PackedPhraseInductionModel
from phrase_induction_train import PhraseInductionModel


def make_indexer(cls, device: torch.device):
    indexer = object.__new__(cls)
    indexer.vocab_size = 50_257
    indexer.phrase_orders = (2, 3, 4)
    indexer.phrase_history = 1
    return indexer


def check_device(device: torch.device) -> None:
    generator = torch.Generator(device=device).manual_seed(13)
    input_ids = torch.randint(0, 32, (3, 257), generator=generator, device=device)
    old = make_indexer(PhraseInductionModel, device)
    packed = make_indexer(PackedPhraseInductionModel, device)
    actual = packed._build_packed_indices(input_ids)
    for order in packed.phrase_orders:
        expected_tokens, expected_valid = old._build_order_index(input_ids, order)
        actual_tokens, actual_valid = actual[order]
        torch.testing.assert_close(actual_valid, expected_valid)
        torch.testing.assert_close(actual_tokens[actual_valid], expected_tokens[expected_valid])


def test_packed_phrase_index_matches_reference() -> None:
    check_device(torch.device("cpu"))
    if torch.cuda.is_available():
        check_device(torch.device("cuda"))


if __name__ == "__main__":
    test_packed_phrase_index_matches_reference()
    print("packed phrase index matches reference on all available devices")
