"""Demo: Tier 1 vs Tier 2 extraction on real CROSS rulings."""

import json
import re

from rich.console import Console
from rich.panel import Panel

console = Console(width=100)

# ── Real data from the database ──────────────────────────────────────────────

RULING_1_META = {
    "ruling_number": "N033137",
    "subject": "The tariff classification of silicone breast forms from China.  Correction to Ruling Number N030676.",
    "tariffs": ["3926.20.9050"],
    "ruling_date": "2008-07-17",
}

RULING_2_META = {
    "ruling_number": "N047279",
    "subject": "The tariff classification of Voriconazole (CAS-137234-62-9) in bulk form, from China",
    "tariffs": ["2933.59.3600"],
    "ruling_date": "2008-12-23",
}

RULING_2_TEXT = """N047279\r\rDecember 23, 2008\r\rCLA-2-29:OT:RR:E:NC:2:238 \r\rCATEGORY:\tClassification\r\rTARIFF NO.: 2933.59.3600  \r\rMr. Cesar Eduardo Cuneo\rFirstmed Holding Corp.\r7820 SW 196 Terrace\rMiami, FL 33189\r\rRE:\tThe tariff classification of Voriconazole (CAS-137234-62-9) in bulk form, from China \r\rDear Mr. Cuneo:\r\rIn your letter dated December 17, 2008, you requested a tariff classification ruling.\r\rThe subject product, Voriconazole, is used to treat serious fungal infections such as invasive aspergillosis (a fungal infection that begins in the lungs and spreads through the bloodstream to other organs) and esophageal candidiasis (infection by a yeast-like fungus that may cause white patching in the mouth and throat). Voriconazole is in a class of antifungal medications called triazoles. It works by slowing the growth of the fungi that cause infection.\r\rThe applicable subheading for the Voriconazole will be 2933.59.3600, Harmonized Tariff Schedule of the United States (HTSUS), which provides for "Heterocyclic compounds with nitrogen hetero-atom(s) only: Compounds containing a pyrimidine ring (whether or not hydrogenated) or piperazine ring in the structure: Other: Drugs: Aromatic or modified aromatic: Anti-infective agents: Other." Pursuant to General Note 13, HTSUS, the rate of duty will be free.\r\rDuty rates are provided for your convenience and are subject to change.  The text of the most recent HTSUS and the accompanying duty rates are provided on World Wide Web at http://www.usitc.gov/tata/hts/.\r\rThis merchandise may be subject to the Federal Food, Drug, and Cosmetic Act and/or The Public Health Security and Bioterrorism Preparedness and Response Act of 2002 (The Bioterrorism Act), which are administered by the U.S. Food and Drug Administration (FDA). Information on the Federal Food, Drug, and Cosmetic Act, as well as The Bioterrorism Act, can be obtained by calling the FDA at 1-888-463-6332, or by visiting their website at www.fda.gov.\r\rThis ruling is being issued under the provisions of Part 177 of the Customs Regulations (19 C.F.R. 177).\r\rA copy of the ruling or the control number indicated above should be provided with the entry documents filed at the time this merchandise is imported.  If you have any questions regarding the ruling, contact National Import Specialist Harvey Kuperstein at (646) 733-3033.\r\rSincerely,\r\r\r\r\rRobert B. Swierupski\rDirector\rNational Commodity Specialist Division"""


# ── Tier 1: Subject cleaning ─────────────────────────────────────────────────

SUBJECT_PREFIXES = [
    r"^The tariff classification (?:and [^of]+ )?of ",
    r"^Classification of ",
    r"^Tariff classification of ",
    r"^RE:\s*The tariff classification of ",
    r"^RE:\s*",
]
PREFIX_PATTERN = re.compile("|".join(f"(?:{p})" for p in SUBJECT_PREFIXES), re.IGNORECASE)


def tier1_clean_subject(subject: str) -> str:
    cleaned = PREFIX_PATTERN.sub("", subject).strip()
    # Remove trailing corrections/references
    cleaned = re.sub(r"\s*Correction to Ruling.*$", "", cleaned, flags=re.IGNORECASE).strip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1].strip()
    return cleaned if len(cleaned) > 10 else subject.strip()


# ── Tier 2: Ruling text regex extraction ─────────────────────────────────────

def tier2_extract(ruling_text: str) -> dict[str, str | None]:
    text = ruling_text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Extract product description
    product_desc = None
    desc_patterns = [
        re.compile(
            r"(?:The subject (?:product|merchandise|item|article)s?\s*,\s*)(.*?)(?=\n\n|\nThe applicable)",
            re.DOTALL | re.IGNORECASE,
        ),
        re.compile(
            r"(?:The subject (?:product|merchandise|item|article)s?\s+(?:is|are)\s+)(.*?)(?=\n\n|\nThe applicable)",
            re.DOTALL | re.IGNORECASE,
        ),
    ]
    for p in desc_patterns:
        m = p.search(text)
        if m:
            product_desc = re.sub(r"\s+", " ", m.group(1)).strip()
            break

    # Extract classification reasoning
    reasoning = None
    reasoning_pattern = re.compile(
        r'(The applicable (?:sub)?heading.*?which provides for\s*["\u201c\u201d].*?["\u201c\u201d]\.?)\s',
        re.DOTALL | re.IGNORECASE,
    )
    m = reasoning_pattern.search(text)
    if m:
        reasoning = re.sub(r"\s+", " ", m.group(1)).strip()

    return {"product_description": product_desc, "reasoning": reasoning}


# ── Demo ─────────────────────────────────────────────────────────────────────

def main():
    # ── Ruling 1: N033137 (no ruling text available for this demo) ────────
    console.print("\n[bold yellow]═══ RULING N033137: Silicone Breast Forms ═══[/bold yellow]\n")

    console.print(Panel(
        f"subject: {RULING_1_META['subject']}\ntariffs: {RULING_1_META['tariffs']}",
        title="Raw DB Record",
        border_style="dim",
    ))

    t1_desc = tier1_clean_subject(RULING_1_META["subject"])
    tier1_record = {
        "description": t1_desc,
        "hts_code": RULING_1_META["tariffs"][0],
        "source": "cross_ruling",
        "ruling_number": "N033137",
    }
    console.print(Panel(
        json.dumps(tier1_record, indent=2),
        title="TIER 1 Output (cleaned subject)",
        border_style="cyan",
    ))

    console.print("[dim]No ruling text shown for this ruling — Tier 2 would need the full text.[/dim]\n")

    # ── Ruling 2: N047279 (Voriconazole — has full ruling text) ───────────
    console.print("\n[bold yellow]═══ RULING N047279: Voriconazole ═══[/bold yellow]\n")

    console.print(Panel(
        f"subject: {RULING_2_META['subject']}\ntariffs: {RULING_2_META['tariffs']}",
        title="Raw DB Record (metadata)",
        border_style="dim",
    ))

    # Tier 1
    t1_desc = tier1_clean_subject(RULING_2_META["subject"])
    tier1_record = {
        "description": t1_desc,
        "hts_code": RULING_2_META["tariffs"][0],
        "source": "cross_ruling",
        "ruling_number": "N047279",
    }
    console.print(Panel(
        json.dumps(tier1_record, indent=2),
        title="TIER 1 Output (cleaned subject)",
        border_style="cyan",
    ))

    # Tier 2
    extracted = tier2_extract(RULING_2_TEXT)
    tier2_record = {
        "description": extracted["product_description"] or t1_desc,
        "hts_code": RULING_2_META["tariffs"][0],
        "source": "cross_ruling",
        "ruling_number": "N047279",
        "subject": t1_desc,
        "reasoning": extracted["reasoning"],
        "description_source": "ruling_text" if extracted["product_description"] else "subject",
    }
    console.print(Panel(
        json.dumps(tier2_record, indent=2, ensure_ascii=False),
        title="TIER 2 Output (regex-extracted from ruling text)",
        border_style="green",
    ))

    # ── Side-by-side comparison ───────────────────────────────────────────
    console.print("\n[bold yellow]═══ SIDE-BY-SIDE: What the model would train on ═══[/bold yellow]\n")

    console.print("[bold cyan]Tier 1 description:[/bold cyan]")
    console.print(f"  \"{tier1_record['description']}\"\n")

    console.print("[bold green]Tier 2 description:[/bold green]")
    console.print(f"  \"{tier2_record['description']}\"\n")

    if tier2_record.get("reasoning"):
        console.print("[bold magenta]Tier 2 reasoning (RAG context):[/bold magenta]")
        console.print(f"  \"{tier2_record['reasoning']}\"\n")


if __name__ == "__main__":
    main()
