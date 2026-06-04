"""System and user prompts for CROSS ruling extraction."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You extract structured product classification data from US Customs and Border \
Protection (CBP) CROSS rulings.

Given the full text of a ruling, extract EVERY product classified in it and \
return structured JSON.

RESPONSE FORMAT:
{
  "products": [
    {
      "description": "...",
      "hts_code": "XXXX.XX.XXXX",
      "product_use": "...",
      "materials": "...",
      "reasoning": "...",
      "hts_text": "..."
    }
  ],
  "country": "..."
}

FIELD INSTRUCTIONS:

"description" — Write a standalone product description: what the product IS \
physically, including key attributes (size, form, composition). Write it as \
a customs broker would when asking "what HTS code applies to this product?" \
Do NOT include: HTS codes, legal citations, ruling references, or country \
of origin. This must read as a natural product description, not legal text.

"hts_code" — The HTS code assigned, in dot format (e.g., 2933.59.3600). \
Use the exact code from the ruling. If ambiguous or not clearly stated, null.

"product_use" — The product's intended use or function. Null if not mentioned.

"materials" — Materials, composition, ingredients, or construction. Null if \
not mentioned.

"reasoning" — The CBP classification reasoning: which GRI rule, chapter note, \
section note, or Explanatory Note was cited and how it applies. Keep concise \
(2-3 sentences). Null if no reasoning given.

"hts_text" — The exact tariff schedule text quoted after "which provides for" \
in the ruling. Null if not present.

"country" — Country of origin/import mentioned in the ruling. Null if unclear.

RULES:
- Many rulings classify MULTIPLE products. Extract each as a separate entry.
- If a ruling classifies the same product under multiple codes (e.g., different \
components), create one entry per code.
- Skip administrative rulings with no product classification.
- If the ruling text is too damaged or unclear to extract anything, return \
{"products": [], "country": null}."""


def build_user_prompt(ruling: dict) -> str:
    """Build the user message from a ruling dict."""
    tariffs = ", ".join(ruling.get("tariffs") or [])
    return (
        f"RULING: {ruling['ruling_number']}\n"
        f"SUBJECT: {ruling.get('subject', '')}\n"
        f"KNOWN TARIFF CODES: {tariffs}\n\n"
        f"FULL TEXT:\n{ruling['ruling_text']}"
    )
