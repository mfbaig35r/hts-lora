"""Completion-only data collator for causal LM fine-tuning.

Masks loss on all tokens before the assistant response boundary.
The assistant boundary is detected using the token sequence for
`<|start_header_id|>assistant<|end_header_id|>`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from hts_lora.utils.logging import get_logger

logger = get_logger("training.collators")

IGNORE_INDEX = -100


@dataclass
class CompletionOnlyCollator:
    """Collator that masks loss on system+user prompt tokens.

    Only the assistant completion tokens contribute to the training loss.
    The assistant boundary is found by searching for the header token sequence.
    """

    tokenizer: PreTrainedTokenizerBase
    max_seq_length: int = 2048

    def __post_init__(self) -> None:
        # Pre-compute the assistant header token sequence
        # For Llama-3 chat format: <|start_header_id|>assistant<|end_header_id|>\n\n
        self.assistant_header_tokens = self.tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False,
        )
        self.header_len = len(self.assistant_header_tokens)
        logger.info(
            f"Assistant header tokens ({self.header_len}): "
            f"{self.tokenizer.decode(self.assistant_header_tokens)!r}"
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a batch of examples with completion-only loss masking."""
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            messages = feature["messages"]

            # Apply chat template to get the full token sequence
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors=None,
            )

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # Find the LAST assistant header boundary
            labels = [IGNORE_INDEX] * len(input_ids)
            assistant_start = self._find_last_assistant_start(input_ids)

            if assistant_start is not None:
                # Unmask everything after the assistant header
                for i in range(assistant_start, len(input_ids)):
                    labels[i] = input_ids[i]
            else:
                # Fallback: unmask everything (shouldn't happen with valid data)
                logger.warning("Could not find assistant header boundary, unmasking all tokens")
                labels = list(input_ids)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        # Pad to max length in batch
        max_len = min(max(len(ids) for ids in batch_input_ids), self.max_seq_length)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, lbls in zip(batch_input_ids, batch_attention_mask, batch_labels):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
            padded_labels.append(lbls + [IGNORE_INDEX] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    def _find_last_assistant_start(self, input_ids: list[int]) -> int | None:
        """Find the token index where the last assistant response starts.

        Returns the index AFTER the header tokens (i.e., the first token
        of the actual assistant content).
        """
        header = self.assistant_header_tokens
        header_len = self.header_len

        # Search backwards for the last occurrence
        for i in range(len(input_ids) - header_len, -1, -1):
            if input_ids[i : i + header_len] == header:
                return i + header_len

        return None
