"""Single-example inference with JSON parsing and fallbacks."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hts_lora.utils.logging import get_logger

logger = get_logger("inference.predict")

TaskMode = Literal["rerank", "rag_classify", "direct_classify"]


def build_messages(
    description: str,
    mode: TaskMode,
    candidates: list[str] | None = None,
    context: str | None = None,
) -> list[dict[str, str]]:
    """Build the chat messages for a prediction request."""
    from hts_lora.data.formatters import _SYSTEM_PROMPTS

    system_prompt = _SYSTEM_PROMPTS[mode]

    if mode == "rerank" and candidates:
        candidates_str = "\n".join(f"  - {c}" for c in candidates)
        user_content = f"Product description: {description}\n\nCandidate HTS codes:\n{candidates_str}"
    elif mode == "rag_classify" and context:
        user_content = f"Product description: {description}\n\n{context}"
    else:
        user_content = f"Product description: {description}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def predict(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    description: str,
    mode: TaskMode = "direct_classify",
    candidates: list[str] | None = None,
    context: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> dict[str, Any]:
    """Run a single prediction and parse the JSON response.

    Returns a dict with:
      - parsed: the parsed JSON response (or None if parsing failed)
      - raw: the raw generated text
      - parse_ok: whether JSON parsing succeeded
    """
    messages = build_messages(description, mode, candidates, context)

    # Apply chat template with generation prompt
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # Pre-fill the think tags to suppress reasoning
    text += "<think>\n</think>\n\n"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.05,
        )

    # Decode only the generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Parse the JSON response
    parsed = _parse_response(raw_text)

    return {
        "parsed": parsed,
        "raw": raw_text,
        "parse_ok": parsed is not None,
    }


def _parse_response(text: str) -> dict[str, Any] | None:
    """Try to parse JSON from the model response with multiple fallback strategies."""
    text = text.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 4: Try to fix common JSON issues
    # Trailing comma before closing brace
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning(f"Failed to parse JSON from response: {text[:200]}")
    return None
