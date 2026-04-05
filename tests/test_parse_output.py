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
        assert p.chapter_code is None
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
