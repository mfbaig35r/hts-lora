"""Batch inference with left-padding and progress tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from hts_lora.inference.predict import TaskMode, _parse_response, build_messages
from hts_lora.utils.io import append_jsonl, read_jsonl
from hts_lora.utils.logging import get_logger

logger = get_logger("inference.batch_predict")


def batch_predict(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_path: str | Path,
    output_path: str | Path,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    default_mode: TaskMode = "direct_classify",
) -> dict[str, int]:
    """Run batch inference on a JSONL file.

    Input JSONL format (each line):
        {"description": "...", "mode": "...", "candidates": [...], "context": "..."}

    Output JSONL format (each line):
        {<input fields>, "prediction": {...}, "raw": "...", "parse_ok": true/false}

    Returns summary stats.
    """
    records = read_jsonl(input_path)
    logger.info(f"Loaded {len(records)} records from {input_path}")

    # Ensure left-padding for batched generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "parse_ok": 0, "parse_fail": 0}

    for batch_start in tqdm(range(0, len(records), batch_size), desc="Predicting"):
        batch = records[batch_start : batch_start + batch_size]
        results = _predict_batch(
            model, tokenizer, batch, max_new_tokens, default_mode
        )

        for record, result in zip(batch, results):
            output_record = {
                **record,
                "prediction": result["parsed"],
                "raw": result["raw"],
                "parse_ok": result["parse_ok"],
            }
            append_jsonl(output_record, output_path)
            stats["total"] += 1
            if result["parse_ok"]:
                stats["parse_ok"] += 1
            else:
                stats["parse_fail"] += 1

    logger.info(
        f"Batch prediction complete: {stats['total']} total, "
        f"{stats['parse_ok']} parsed, {stats['parse_fail']} failed"
    )
    return stats


def _predict_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch: list[dict[str, Any]],
    max_new_tokens: int,
    default_mode: TaskMode,
) -> list[dict[str, Any]]:
    """Run inference on a batch of records."""
    # Build prompts
    prompts = []
    for record in batch:
        mode = record.get("mode", default_mode)
        messages = build_messages(
            description=record["description"],
            mode=mode,
            candidates=record.get("candidates"),
            context=record.get("context"),
        )
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        text += "<think>\n</think>\n\n"
        prompts.append(text)

    # Tokenize with left-padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    prompt_lengths = [
        (inputs["attention_mask"][i] == 1).sum().item()
        for i in range(len(batch))
    ]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.05,
        )

    # Decode each result
    results = []
    for i, (output_ids, prompt_len) in enumerate(zip(outputs, prompt_lengths)):
        # For left-padded: generated tokens start after the prompt
        generated_ids = output_ids[inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed = _parse_response(raw_text)
        results.append({
            "parsed": parsed,
            "raw": raw_text,
            "parse_ok": parsed is not None,
        })

    return results
