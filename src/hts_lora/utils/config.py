"""Pydantic v2 configuration models for data, training, evaluation, and export."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


# ── Export Config ────────────────────────────────────────────────────────────


class ExportDatabaseConfig(BaseModel):
    batch_size: int = 5000


class ExportExtractionsConfig(BaseModel):
    min_description_length: int = 30
    min_reasoning_length: int = 20
    max_reasoning_length: int = 500
    require_hts_code: bool = True
    require_reasoning: bool = True
    exclude_failed: bool = True


class ExportCurrentValidConfig(BaseModel):
    enabled: bool = True
    match_level: Literal["hts8", "hts6", "hts4"] = "hts8"
    log_excluded: bool = True


class ExportGlossaryConfig(BaseModel):
    enabled: bool = True
    min_senses: int = 1


class ExportEnrichmentsConfig(BaseModel):
    enabled: bool = True
    min_description_length: int = 30


class ExportOutputConfig(BaseModel):
    base_dir: str = "data/raw"
    versioned: bool = True


class ExportConfig(BaseModel):
    database: ExportDatabaseConfig = Field(default_factory=ExportDatabaseConfig)
    extractions: ExportExtractionsConfig = Field(default_factory=ExportExtractionsConfig)
    current_valid: ExportCurrentValidConfig = Field(default_factory=ExportCurrentValidConfig)
    glossary: ExportGlossaryConfig = Field(default_factory=ExportGlossaryConfig)
    enrichments: ExportEnrichmentsConfig = Field(default_factory=ExportEnrichmentsConfig)
    output: ExportOutputConfig = Field(default_factory=ExportOutputConfig)


# ── Data Config ──────────────────────────────────────────────────────────────


class SourceConfig(BaseModel):
    path: str
    format: Literal["csv", "jsonl", "json"]
    text_field: str
    code_field: str


class NormalizationConfig(BaseModel):
    min_description_length: int = 10
    max_description_length: int = 2048
    dedup_minhash_threshold: float = 0.8
    dedup_minhash_num_perm: int = 128


class TaskWeightsConfig(BaseModel):
    hierarchical_classify: float = 0.90
    abstention: float = 0.10


class AbstentionConfig(BaseModel):
    rate: float = 0.10
    label: str = "__ABSTAIN__"
    categories: list[str] = Field(
        default_factory=lambda: ["vague_description", "missing_materials", "ambiguous_use"]
    )


class FrequencyCapConfig(BaseModel):
    enabled: bool = True
    max_per_code: int = 100
    apply_to: Literal["train", "all"] = "train"


class SplitConfig(BaseModel):
    train: float = 0.80
    val: float = 0.10
    test: float = 0.10
    seed: int = 42
    stratify_by: str = "chapter"
    split_by: Literal["row", "ruling"] = "ruling"


class DataConfig(BaseModel):
    sources: list[SourceConfig] = Field(default_factory=list)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    task_weights: TaskWeightsConfig = Field(default_factory=TaskWeightsConfig)
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)
    frequency_cap: FrequencyCapConfig = Field(default_factory=FrequencyCapConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    glossary_path: str = ""
    output_dir: str = "data/processed"
    formatted_dir: str = "data/formatted"


# ── Training Config ──────────────────────────────────────────────────────────


class ModelConfig(BaseModel):
    model_id: str = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"


class QuantizationConfig(BaseModel):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


class LoraConfig(BaseModel):
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class TrainingHyperparams(BaseModel):
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 2048
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"


class TrainConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    training: TrainingHyperparams = Field(default_factory=TrainingHyperparams)
    data_dir: str = "data/formatted"
    output_dir: str = "outputs"


# ── Eval Config ──────────────────────────────────────────────────────────────


class GenerationConfig(BaseModel):
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = False
    repetition_penalty: float = 1.05


class EvalConfig(BaseModel):
    adapter_path: str = "outputs/latest/adapter"
    test_data: str = "data/formatted/test.jsonl"
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    metrics: list[str] = Field(
        default_factory=lambda: [
            "exact_match", "chapter_match", "heading_match",
            "subheading_match", "top_k_accuracy", "abstain_rate",
            "json_parse_rate", "confidence_calibration",
        ]
    )
    top_k_values: list[int] = Field(default_factory=lambda: [1, 3, 5])
    output_dir: str = "outputs/latest/eval"


# ── Loaders ──────────────────────────────────────────────────────────────────


def load_config(path: str | Path, config_class: type[BaseModel]) -> BaseModel:
    """Load a YAML config file into a Pydantic model."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return config_class.model_validate(raw)


def load_data_config(path: str | Path) -> DataConfig:
    return load_config(path, DataConfig)  # type: ignore[return-value]


def load_train_config(path: str | Path) -> TrainConfig:
    return load_config(path, TrainConfig)  # type: ignore[return-value]


def load_eval_config(path: str | Path) -> EvalConfig:
    return load_config(path, EvalConfig)  # type: ignore[return-value]


def load_export_config(path: str | Path) -> ExportConfig:
    return load_config(path, ExportConfig)  # type: ignore[return-value]
