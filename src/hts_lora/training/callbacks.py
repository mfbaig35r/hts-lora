"""Training callbacks for metrics logging and sample predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from hts_lora.utils.io import append_jsonl
from hts_lora.utils.logging import get_logger

logger = get_logger("training.callbacks")


class JSONMetricsCallback(TrainerCallback):
    """Write training/eval metrics to a JSONL file after each log step."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return
        record = {
            "step": state.global_step,
            "epoch": state.epoch,
            **logs,
        }
        append_jsonl(record, self.output_path)


class SamplePredictionCallback(TrainerCallback):
    """Generate and log sample predictions at each eval step.

    Runs inference on a small set of held-out examples to track
    qualitative output quality during training.
    """

    def __init__(
        self,
        sample_inputs: list[dict[str, Any]],
        tokenizer: Any,
        output_path: str | Path,
        max_new_tokens: int = 256,
    ):
        self.sample_inputs = sample_inputs
        self.tokenizer = tokenizer
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = max_new_tokens

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        if model is None:
            return

        model.eval()
        predictions = []

        for sample in self.sample_inputs:
            messages = sample["messages"][:2]  # system + user only
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # Pre-fill the think tags
            text += "<think>\n</think>\n\n"

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_seq_length if hasattr(args, "max_seq_length") else 2048,
            ).to(model.device)

            import torch

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            expected = sample["messages"][2]["content"] if len(sample["messages"]) > 2 else ""

            predictions.append({
                "step": state.global_step,
                "input_preview": messages[1]["content"][:200],
                "expected": expected[:500],
                "generated": generated[:500],
            })

        for pred in predictions:
            append_jsonl(pred, self.output_path)

        logger.info(f"Logged {len(predictions)} sample predictions at step {state.global_step}")
