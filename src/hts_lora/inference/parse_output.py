"""Parse v2 structured text output into typed predictions.

The v2 model outputs structured text like:
    Chapter 85: ELECTRICAL MACHINERY...
    Heading 85.44: Insulated wire, cable...
    Subheading 8544.30: Insulated winding wire...
    HTS Code: 8544.30.0000

    Reasoning: Classified under...

    Provides for: Insulated winding wire of copper

Or for abstentions:
    Cannot classify: The product description is too vague...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ParsedPrediction:
    """Structured prediction parsed from v2 model output."""

    chapter_code: str | None = None
    chapter_desc: str | None = None
    heading_code: str | None = None
    heading_desc: str | None = None
    subheading_code: str | None = None
    subheading_desc: str | None = None
    hts_code: str | None = None
    reasoning: str | None = None
    provides_for: str | None = None
    is_abstention: bool = False
    abstention_reason: str | None = None
    parse_ok: bool = False
    raw: str = ""


# Section-start patterns used to delimit multiline fields
_SECTION_STARTS = re.compile(
    r"^(?:Chapter \d|Heading [\d.]+|Subheading [\d.]+|HTS Code:|Reasoning:|Provides for:|Cannot classify:)",
    re.MULTILINE,
)


def parse_prediction(text: str) -> ParsedPrediction:
    """Parse v2 structured text into a ParsedPrediction.

    Handles:
    - Full hierarchy output (Chapter/Heading/Subheading/HTS Code/Reasoning/Provides for)
    - Abstention output ("Cannot classify: ...")
    - Partial outputs (missing fields)
    - Think tags (strips <think>...</think> prefix)
    """
    result = ParsedPrediction(raw=text)

    # Strip think tags if present
    body = text
    think_end = body.find("</think>")
    if think_end != -1:
        body = body[think_end + len("</think>"):].strip()

    if not body:
        return result

    # Check for abstention
    abstain_match = re.match(r"Cannot classify:\s*(.+)", body, re.DOTALL)
    if abstain_match:
        result.is_abstention = True
        result.abstention_reason = abstain_match.group(1).strip()
        result.parse_ok = True
        return result

    # Parse chapter
    chap_match = re.search(r"Chapter (\d{1,2}):\s*(.+?)(?:\n|$)", body)
    if chap_match:
        result.chapter_code = chap_match.group(1)
        result.chapter_desc = chap_match.group(2).strip()

    # Parse heading
    head_match = re.search(r"Heading ([\d.]+):\s*(.+?)(?:\n|$)", body)
    if head_match:
        result.heading_code = head_match.group(1)
        result.heading_desc = head_match.group(2).strip()

    # Parse subheading
    sub_match = re.search(r"Subheading ([\d.]+):\s*(.+?)(?:\n|$)", body)
    if sub_match:
        result.subheading_code = sub_match.group(1)
        result.subheading_desc = sub_match.group(2).strip()

    # Parse HTS code
    hts_match = re.search(r"HTS Code:\s*([\d.]+)", body)
    if hts_match:
        result.hts_code = hts_match.group(1).strip()

    # Parse reasoning (multiline: from "Reasoning:" to next section or end)
    result.reasoning = _extract_multiline(body, "Reasoning:")

    # Parse provides_for (multiline: from "Provides for:" to next section or end)
    result.provides_for = _extract_multiline(body, "Provides for:")

    # parse_ok if we got at least an HTS code
    result.parse_ok = result.hts_code is not None

    return result


def _extract_multiline(text: str, prefix: str) -> str | None:
    """Extract a multiline field starting with prefix, ending at next section or EOF."""
    idx = text.find(prefix)
    if idx == -1:
        return None

    start = idx + len(prefix)
    # Find the next section header after our content
    remaining = text[start:]
    # Look for the next section-starting pattern
    next_section = _SECTION_STARTS.search(remaining)
    if next_section:
        content = remaining[:next_section.start()]
    else:
        content = remaining

    content = content.strip()
    return content if content else None
