"""Tests for v2 chat message formatting with hierarchical output."""

from hts_lora.data.build_examples import TrainingExample
from hts_lora.data.formatters import format_dataset, format_example


class TestFormatExample:
    def test_classify_structure(self):
        ex = TrainingExample(
            description="Live horses for breeding",
            hts_code="0101210010",
            task_type="hierarchical_classify",
            input_variant="rich",
            chapter_code="01",
            chapter_description="Live animals",
            heading_code="0101",
            heading_description="Live horses, asses, mules and hinnies",
        )
        result = format_example(ex)

        assert "messages" in result
        messages = result["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_system_prompt_has_thinking_off(self):
        ex = TrainingExample(
            description="Test product",
            hts_code="0101210010",
            task_type="hierarchical_classify",
        )
        result = format_example(ex)
        assert result["messages"][0]["content"].startswith("detailed thinking off")

    def test_assistant_has_think_tags(self):
        ex = TrainingExample(
            description="Test product",
            hts_code="0101210010",
            task_type="hierarchical_classify",
        )
        result = format_example(ex)
        assistant = result["messages"][2]["content"]
        assert assistant.startswith("<think>\n</think>\n\n")

    def test_assistant_has_hierarchy_lines(self):
        ex = TrainingExample(
            description="Copper wire insulated",
            hts_code="8544300000",
            task_type="hierarchical_classify",
            chapter_code="85",
            chapter_description="Electrical machinery",
            heading_code="8544",
            heading_description="Insulated wire, cable",
            tariff_description="Insulated winding wire",
        )
        result = format_example(ex)
        assistant = result["messages"][2]["content"]
        body = assistant.split("</think>\n\n", 1)[1]

        assert "Chapter 85:" in body
        assert "Heading 85.44:" in body
        assert "Subheading 8544.30:" in body
        assert "HTS Code: 8544.30.0000" in body
        assert "Reasoning:" in body

    def test_rich_variant_includes_materials(self):
        ex = TrainingExample(
            description="Copper wire",
            hts_code="8544300000",
            task_type="hierarchical_classify",
            input_variant="rich",
            materials="copper",
            product_use="electrical wiring",
            country="China",
        )
        result = format_example(ex)
        user = result["messages"][1]["content"]
        assert "Materials: copper" in user
        assert "Use: electrical wiring" in user
        assert "Country of origin: China" in user

    def test_minimal_variant_description_only(self):
        ex = TrainingExample(
            description="Copper wire",
            hts_code="8544300000",
            task_type="hierarchical_classify",
            input_variant="minimal",
            materials="copper",
        )
        result = format_example(ex)
        user = result["messages"][1]["content"]
        assert "Product: Copper wire" in user
        assert "Materials" not in user

    def test_glossary_variant_includes_terms(self):
        ex = TrainingExample(
            description="Alloy steel bolts",
            hts_code="7318152065",
            task_type="hierarchical_classify",
            input_variant="glossary_enriched",
            glossary_terms=[{"term": "alloy steel", "definition": "Steel with added elements"}],
        )
        result = format_example(ex)
        user = result["messages"][1]["content"]
        assert "alloy steel" in user
        assert "Steel with added elements" in user

    def test_materials_only_variant(self):
        ex = TrainingExample(
            description="Some product",
            hts_code="8544300000",
            task_type="hierarchical_classify",
            input_variant="materials_only",
            materials="copper, PVC insulation",
            product_use="electrical distribution",
        )
        result = format_example(ex)
        user = result["messages"][1]["content"]
        assert "Materials: copper, PVC insulation" in user
        assert "Use: electrical distribution" in user

    def test_abstain_example(self):
        ex = TrainingExample(
            description="Ambiguous product",
            hts_code="__ABSTAIN__",
            task_type="abstention",
            abstain=True,
            abstain_category="vague_description",
        )
        result = format_example(ex)
        assistant = result["messages"][2]["content"]
        body = assistant.split("</think>\n\n", 1)[1]
        assert "Cannot classify" in body

    def test_abstain_missing_materials(self):
        ex = TrainingExample(
            description="Product of unspecified composition",
            hts_code="__ABSTAIN__",
            task_type="abstention",
            abstain=True,
            abstain_category="missing_materials",
        )
        result = format_example(ex)
        assistant = result["messages"][2]["content"]
        assert "material composition" in assistant.lower()

    def test_abstain_ambiguous_use(self):
        ex = TrainingExample(
            description="Product...",
            hts_code="__ABSTAIN__",
            task_type="abstention",
            abstain=True,
            abstain_category="ambiguous_use",
        )
        result = format_example(ex)
        assistant = result["messages"][2]["content"]
        assert "use" in assistant.lower()

    def test_metadata_preserved(self):
        ex = TrainingExample(
            description="Test",
            hts_code="0101210010",
            task_type="hierarchical_classify",
            input_variant="rich",
            ruling_number="N999",
        )
        result = format_example(ex)
        assert result["task_type"] == "hierarchical_classify"
        assert result["hts_code"] == "0101210010"
        assert result["abstain"] is False
        assert result["input_variant"] == "rich"
        assert result["ruling_number"] == "N999"

    def test_provides_for_from_hts_text(self):
        ex = TrainingExample(
            description="Copper wire",
            hts_code="8544300000",
            task_type="hierarchical_classify",
            hts_text="Insulated winding wire of copper",
        )
        result = format_example(ex)
        assistant = result["messages"][2]["content"]
        assert "Provides for: Insulated winding wire of copper" in assistant


class TestFormatDataset:
    def test_formats_multiple(self, sample_training_examples):
        formatted = format_dataset(sample_training_examples)
        assert len(formatted) == len(sample_training_examples)
        for f in formatted:
            assert "messages" in f
            assert len(f["messages"]) == 3
