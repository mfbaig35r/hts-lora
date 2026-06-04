"""Tests for v2 structured text output parsing."""

from hts_lora.inference.parse_output import ParsedPrediction, parse_prediction


class TestParseClassification:
    def test_full_hierarchy(self):
        text = (
            "<think>\n</think>\n\n"
            "Chapter 85: ELECTRICAL MACHINERY AND EQUIPMENT\n"
            "Heading 85.44: Insulated wire, cable\n"
            "Subheading 8544.30: Insulated winding wire\n"
            "HTS Code: 8544.30.0000\n"
            "\n"
            "Reasoning: Classified under Chapter 85 for insulated wire.\n"
            "\n"
            "Provides for: Insulated winding wire of copper"
        )
        p = parse_prediction(text)

        assert p.parse_ok is True
        assert p.is_abstention is False
        assert p.chapter_code == "85"
        assert p.chapter_desc == "ELECTRICAL MACHINERY AND EQUIPMENT"
        assert p.heading_code == "85.44"
        assert p.heading_desc == "Insulated wire, cable"
        assert p.subheading_code == "8544.30"
        assert p.subheading_desc == "Insulated winding wire"
        assert p.hts_code == "8544.30.0000"
        assert p.reasoning == "Classified under Chapter 85 for insulated wire."
        assert p.provides_for == "Insulated winding wire of copper"

    def test_no_think_tags(self):
        text = (
            "Chapter 01: Live animals\n"
            "Heading 01.01: Live horses, asses, mules\n"
            "Subheading 0101.21: Purebred breeding animals\n"
            "HTS Code: 0101.21.0010\n"
            "\n"
            "Reasoning: Live purebred horses for breeding.\n"
            "\n"
            "Provides for: Purebred breeding horses"
        )
        p = parse_prediction(text)

        assert p.parse_ok is True
        assert p.chapter_code == "01"
        assert p.hts_code == "0101.21.0010"

    def test_minimal_output_hts_only(self):
        text = "HTS Code: 8544.30.0000"
        p = parse_prediction(text)

        assert p.parse_ok is True
        assert p.hts_code == "8544.30.0000"
        # Upper levels are backfilled from the HTS code (see TestHierarchyBackfill)
        assert p.chapter_code == "85"
        assert p.heading_code == "85.44"
        assert p.subheading_code == "8544.30"
        # Descriptions stay None — we only derive codes, not fabricated text
        assert p.chapter_desc is None
        assert p.heading_desc is None
        assert p.reasoning is None

    def test_multiline_reasoning(self):
        text = (
            "Chapter 85: Electrical machinery\n"
            "Heading 85.44: Insulated wire\n"
            "Subheading 8544.30: Winding wire\n"
            "HTS Code: 8544.30.0000\n"
            "\n"
            "Reasoning: This product is classified here because:\n"
            "1. It is made of copper\n"
            "2. It is insulated with PVC\n"
            "3. It is used for electrical winding\n"
            "\n"
            "Provides for: Insulated winding wire"
        )
        p = parse_prediction(text)

        assert p.parse_ok is True
        assert "1. It is made of copper" in p.reasoning
        assert "3. It is used for electrical winding" in p.reasoning

    def test_multiline_provides_for(self):
        text = (
            "Chapter 85: Electrical machinery\n"
            "Heading 85.44: Insulated wire\n"
            "Subheading 8544.30: Winding wire\n"
            "HTS Code: 8544.30.0000\n"
            "\n"
            "Reasoning: Classified correctly.\n"
            "\n"
            "Provides for: Insulated winding wire, including\n"
            "those of copper, fitted with connectors"
        )
        p = parse_prediction(text)

        assert p.provides_for is not None
        assert "those of copper" in p.provides_for


class TestHierarchyBackfill:
    """Model sometimes skips a structured line (commonly Heading).
    The parser should reconstruct missing levels from lower levels.
    """

    def test_missing_heading_derived_from_subheading(self):
        # Real wool-sweater case: model emits chapter + subheading + HTS but no Heading line
        text = (
            "Chapter 61: ARTICLES OF APPAREL AND CLOTHING ACCESSORIES, KNITTED OR CROCHETED\n"
            "Subheading 6110.11: Sweaters, knitted or crocheted, of wool\n"
            "HTS Code: 6110.11.0000\n"
            "\n"
            "Reasoning: Classified under heading 6110 for knitted sweaters."
        )
        p = parse_prediction(text)

        assert p.parse_ok is True
        assert p.chapter_code == "61"
        assert p.heading_code == "61.10"  # derived from subheading 6110.11
        assert p.heading_desc is None  # we don't fabricate descriptions
        assert p.subheading_code == "6110.11"
        assert p.hts_code == "6110.11.0000"

    def test_missing_heading_and_subheading_derived_from_hts(self):
        text = (
            "Chapter 85: Electrical machinery\n"
            "HTS Code: 8544.30.0000\n"
            "Reasoning: Insulated wire."
        )
        p = parse_prediction(text)

        assert p.parse_ok is True
        assert p.chapter_code == "85"
        assert p.heading_code == "85.44"
        assert p.subheading_code == "8544.30"
        assert p.hts_code == "8544.30.0000"

    def test_missing_chapter_derived_from_heading(self):
        text = (
            "Heading 85.44: Insulated wire\n"
            "Subheading 8544.30: Winding wire\n"
            "HTS Code: 8544.30.0000"
        )
        p = parse_prediction(text)

        assert p.chapter_code == "85"
        assert p.heading_code == "85.44"

    def test_heading_without_dot_normalized(self):
        # Model writes "Heading 6110:" instead of "Heading 61.10:"
        text = (
            "Chapter 61: Apparel\n"
            "Heading 6110: Knitted sweaters\n"
            "Subheading 6110.11: Of wool\n"
            "HTS Code: 6110.11.0010"
        )
        p = parse_prediction(text)

        assert p.heading_code == "61.10"  # normalized to dotted form
        assert p.heading_desc == "Knitted sweaters"

    def test_subheading_without_dot_normalized(self):
        text = (
            "Chapter 85: Electrical\n"
            "Heading 85.44: Wire\n"
            "Subheading 854430: Winding\n"
            "HTS Code: 8544.30.0000"
        )
        p = parse_prediction(text)

        assert p.subheading_code == "8544.30"

    def test_only_hts_backfills_everything(self):
        p = parse_prediction("HTS Code: 8714.91.9000")

        assert p.parse_ok is True
        assert p.chapter_code == "87"
        assert p.heading_code == "87.14"
        assert p.subheading_code == "8714.91"
        assert p.hts_code == "8714.91.9000"

    def test_no_backfill_when_no_source_data(self):
        p = parse_prediction("")
        assert p.chapter_code is None
        assert p.heading_code is None
        assert p.subheading_code is None


class TestParseAbstention:
    def test_abstention(self):
        text = (
            "<think>\n</think>\n\n"
            "Cannot classify: The product description is too vague or generic "
            "to determine a specific tariff classification."
        )
        p = parse_prediction(text)

        assert p.parse_ok is True
        assert p.is_abstention is True
        assert "too vague" in p.abstention_reason
        assert p.hts_code is None

    def test_abstention_missing_materials(self):
        text = (
            "Cannot classify: The product's material composition is not specified. "
            "HTS classification often depends on the primary material."
        )
        p = parse_prediction(text)

        assert p.is_abstention is True
        assert p.parse_ok is True
        assert "material" in p.abstention_reason.lower()

    def test_abstention_no_think_tags(self):
        text = "Cannot classify: Insufficient information provided."
        p = parse_prediction(text)

        assert p.is_abstention is True
        assert p.parse_ok is True


class TestParseMalformed:
    def test_empty_string(self):
        p = parse_prediction("")
        assert p.parse_ok is False
        assert p.is_abstention is False

    def test_garbage_text(self):
        p = parse_prediction("This is just random text with no structure.")
        assert p.parse_ok is False
        assert p.is_abstention is False
        assert p.raw == "This is just random text with no structure."

    def test_only_think_tags(self):
        p = parse_prediction("<think>\n</think>\n\n")
        assert p.parse_ok is False

    def test_partial_hierarchy_no_hts(self):
        text = (
            "Chapter 85: Electrical machinery\n"
            "Heading 85.44: Insulated wire"
        )
        p = parse_prediction(text)

        assert p.parse_ok is False
        assert p.chapter_code == "85"
        assert p.heading_code == "85.44"
        assert p.hts_code is None

    def test_raw_always_preserved(self):
        text = "Some weird output"
        p = parse_prediction(text)
        assert p.raw == text
