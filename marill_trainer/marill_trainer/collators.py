from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch


class CausalLMDataCollator:
    """
    Minimal collator for causal LM training.

    Expected per-sample input:
    - input_ids: List[int]
    - attention_mask: List[int]
    """

    def __init__(self, tokenizer: Any, label_pad_token_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("CausalLMDataCollator received an empty batch")

        input_features = []
        for feature in features:
            input_ids = feature.get("input_ids")
            attention_mask = feature.get("attention_mask")
            if input_ids is None or attention_mask is None:
                raise ValueError("Each feature must contain 'input_ids' and 'attention_mask'")
            input_features.append(
                {
                    "input_ids": list(input_ids),
                    "attention_mask": list(attention_mask),
                }
            )

        padded = self.tokenizer.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        input_ids = padded["input_ids"]
        attention_mask = padded["attention_mask"]
        labels = input_ids.clone()
        labels = labels.masked_fill(attention_mask == 0, self.label_pad_token_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_data_collator(tokenizer: Any, label_pad_token_id: int = -100) -> CausalLMDataCollator:
    """Build the default minimal collator for causal LM training."""
    return CausalLMDataCollator(
        tokenizer=tokenizer,
        label_pad_token_id=label_pad_token_id,
    )


# TODO: If later we confirm that the training objective should only supervise
# response tokens, add a prompt/response-aware masking path here instead of
# labeling every non-padding token.
# TODO: Some LLaMA tokenizers do not define a pad token. The caller may need to
# set tokenizer.pad_token = tokenizer.eos_token before using this collator.
