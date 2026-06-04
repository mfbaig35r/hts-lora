"""Prompt building for the serving wrapper.

Single source of truth: imports from `hts_lora.inference.predict.build_v2_messages`
and `hts_lora.data.formatters._SYSTEM_PROMPT`. The serving prompt MUST match
the training prompt exactly — never reimplement here.

We talk to the upstream MLX server via the `/v1/completions` (raw prompt)
endpoint rather than `/v1/chat/completions` because the Nemotron model expects
a `<think>\\n</think>\\n\\n` prefix at the start of every assistant response —
that prefix is not part of the standard chat template, so we hand-render the
Llama 3.1 chat format and append it ourselves.
"""

from __future__ import annotations

from hts_lora.data.formatters import _SYSTEM_PROMPT
from hts_lora.inference.predict import InputVariant, build_v2_messages

SYSTEM_PROMPT = _SYSTEM_PROMPT

# Nemotron "thinking off" assistant prefix — must follow the assistant header
# in every generated response. Trained into the model as the leading tokens.
THINK_PREFIX = "<think>\n</think>\n\n"


def build_messages(
    description: str,
    materials: str | None = None,
    use: str | None = None,
    country_of_origin: str | None = None,
) -> list[dict[str, str]]:
    """Build [system, user] messages for an HTS classification request.

    Picks the variant matching the available fields:
      - rich:     description + at least one of (materials, use, country)
      - minimal:  description only
    """
    has_extras = bool(materials or use or country_of_origin)
    variant: InputVariant = "rich" if has_extras else "minimal"

    return build_v2_messages(
        description=description,
        variant=variant,
        materials=materials,
        product_use=use,
        country=country_of_origin,
    )


def render_prompt(messages: list[dict[str, str]]) -> str:
    """Hand-render the Llama 3.1 chat format + Nemotron think prefix.

    The Llama 3.1 chat template uses these special tokens:
        <|begin_of_text|>
        <|start_header_id|>{role}<|end_header_id|>\\n\\n{content}<|eot_id|>
        ...
        <|start_header_id|>assistant<|end_header_id|>\\n\\n

    We then append `<think>\\n</think>\\n\\n` so the model continues from the
    expected starting point.
    """
    parts = ["<|begin_of_text|>"]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    parts.append(THINK_PREFIX)
    return "".join(parts)
