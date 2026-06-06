"""Single-example inference for v2 structured text output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from hts_lora.inference.parse_output import ParsedPrediction, parse_prediction
from hts_lora.utils.logging import get_logger

if TYPE_CHECKING:
    # Heavy ML deps are only needed by predict(); deferring the import keeps
    # build_v2_messages importable from environments without torch installed
    # (the wrapper Docker image, lightweight test contexts, etc.).
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

logger = get_logger("inference.predict")

InputVariant = Literal["rich", "minimal", "glossary_enriched", "materials_only"]


def build_v2_messages(
    description: str,
    variant: InputVariant = "rich",
    materials: str | None = None,
    product_use: str | None = None,
    country: str | None = None,
    glossary_terms: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build system + user messages matching v2 training format."""
    from hts_lora.data.formatters import _SYSTEM_PROMPT

    if variant == "rich":
        parts = [f"Product: {description}"]
        if materials:
            parts.append(f"Materials: {materials}")
        if product_use:
            parts.append(f"Use: {product_use}")
        if country:
            parts.append(f"Country of origin: {country}")
        user_content = "\n".join(parts)

    elif variant == "minimal":
        user_content = f"Product: {description}"

    elif variant == "glossary_enriched":
        parts = [f"Product: {description}"]
        if glossary_terms:
            defs = "\n".join(f"- {t['term']}: {t['definition']}" for t in glossary_terms)
            parts.append(f"\nRelevant trade terms:\n{defs}")
        user_content = "\n".join(parts)

    elif variant == "materials_only":
        parts = []
        if materials:
            parts.append(f"Materials: {materials}")
        if product_use:
            parts.append(f"Use: {product_use}")
        if country:
            parts.append(f"Country of origin: {country}")
        user_content = "\n".join(parts) if parts else f"Product: {description}"

    else:
        user_content = f"Product: {description}"

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def predict(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    description: str,
    variant: InputVariant = "rich",
    materials: str | None = None,
    product_use: str | None = None,
    country: str | None = None,
    glossary_terms: list[dict[str, str]] | None = None,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    repetition_penalty: float = 1.05,
) -> dict[str, Any]:
    """Run a single v2 prediction and parse the structured text response.

    Returns a dict with:
      - prediction: ParsedPrediction object
      - raw: the raw generated text
      - parse_ok: whether parsing succeeded
    """
    import torch  # local import: keeps module importable in torch-less envs

    messages = build_v2_messages(
        description, variant, materials, product_use, country, glossary_terms
    )

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
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
        )

    # Decode only the generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Parse the structured text response
    parsed = parse_prediction(raw_text)

    return {
        "prediction": parsed,
        "raw": raw_text,
        "parse_ok": parsed.parse_ok,
    }
