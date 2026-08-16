from __future__ import annotations

import torch

import phrase_induction_train as phrase


class PackedPhraseInductionModel(phrase.PhraseInductionModel):
    """Build every phrase-order index with one stable sort per batch item."""

    def _build_packed_indices(
        self, input_ids: torch.Tensor
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        batch, length = input_ids.shape
        packed_batch_tokens: dict[int, list[torch.Tensor]] = {
            order: [] for order in self.phrase_orders
        }
        packed_batch_valid: dict[int, list[torch.Tensor]] = {
            order: [] for order in self.phrase_orders
        }

        key_offset = 0
        order_keys = []
        for order in self.phrase_orders:
            keys = torch.zeros_like(input_ids)
            for offset in range(order):
                shifted = torch.zeros_like(input_ids)
                if offset == 0:
                    shifted.copy_(input_ids)
                else:
                    shifted[:, offset:] = input_ids[:, :-offset]
                keys = keys * self.vocab_size + shifted
            order_keys.append(keys + key_offset)
            key_offset += self.vocab_size**order

        packed_keys = torch.cat(order_keys, dim=1)
        packed_length = packed_keys.shape[1]
        packed_slots = torch.arange(packed_length, device=input_ids.device)

        for batch_index in range(batch):
            local_keys = packed_keys[batch_index]
            sort_order = torch.argsort(local_keys, stable=True)
            sorted_keys = local_keys.index_select(0, sort_order)
            previous_slots = torch.roll(sort_order, 1)
            current_order = torch.div(sort_order, length, rounding_mode="floor")
            previous_order = torch.div(previous_slots, length, rounding_mode="floor")
            current_local = torch.remainder(sort_order, length)
            previous_local = torch.remainder(previous_slots, length)

            valid_sorted = packed_slots > 0
            valid_sorted &= sorted_keys == torch.roll(sorted_keys, 1)
            valid_sorted &= current_order == previous_order
            valid_sorted &= previous_local + 1 < length

            order_minimums = torch.tensor(
                [order - 1 for order in self.phrase_orders],
                device=input_ids.device,
                dtype=torch.long,
            )
            valid_sorted &= current_local >= order_minimums[current_order]
            valid_sorted &= previous_local >= order_minimums[previous_order]

            retrieved_sorted = input_ids[batch_index].index_select(
                0, (previous_local + 1).clamp(max=length - 1)
            )
            retrieved = torch.zeros(packed_length, device=input_ids.device, dtype=torch.long)
            valid = torch.zeros(packed_length, device=input_ids.device, dtype=torch.bool)
            retrieved.scatter_(0, sort_order, retrieved_sorted)
            valid.scatter_(0, sort_order, valid_sorted)

            for order_index, order in enumerate(self.phrase_orders):
                start = order_index * length
                stop = start + length
                packed_batch_tokens[order].append(retrieved[start:stop].unsqueeze(-1))
                packed_batch_valid[order].append(valid[start:stop].unsqueeze(-1))

        return {
            order: (
                torch.stack(packed_batch_tokens[order], dim=0),
                torch.stack(packed_batch_valid[order], dim=0),
            )
            for order in self.phrase_orders
        }

    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = super(phrase.PhraseInductionModel, self).features(input_ids)
        self._phrase_tokens.clear()
        self._phrase_valid.clear()
        for order, (tokens, valid) in self._build_packed_indices(input_ids).items():
            self._phrase_tokens[order] = tokens
            self._phrase_valid[order] = valid
        return features


trainer = phrase.experiment.experiment.experiment.trainer
trainer.CausalConvFactorizedLM = PackedPhraseInductionModel


if __name__ == "__main__":
    trainer.train(trainer.parse_args())
