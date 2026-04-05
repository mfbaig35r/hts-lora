"""Model loading: 4-bit quantized base model + LoRA adapter application."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from hts_lora.utils.config import TrainConfig
from hts_lora.utils.logging import get_logger

logger = get_logger("training.model_factory")


def load_base_model(config: TrainConfig) -> AutoModelForCausalLM:
    """Load the base model with 4-bit quantization."""
    quant = config.quantization
    compute_dtype = getattr(torch, quant.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant.load_in_4bit,
        bnb_4bit_quant_type=quant.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant.bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        attn_implementation=config.model.attn_implementation,
        trust_remote_code=True,
    )

    logger.info(f"Loaded base model: {config.model.model_id}")
    return model


def load_tokenizer(config: TrainConfig) -> AutoTokenizer:
    """Load the tokenizer with padding configured for training."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_id,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"  # Right-padding for training
    logger.info(f"Tokenizer loaded, vocab size: {len(tokenizer)}")
    return tokenizer


def apply_lora(model: AutoModelForCausalLM, config: TrainConfig) -> PeftModel:
    """Prepare model for k-bit training and apply LoRA adapters."""
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.training.gradient_checkpointing,
    )

    lora_cfg = config.lora
    peft_config = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )

    model = get_peft_model(model, peft_config)

    trainable, total = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total
    logger.info(f"LoRA applied: {trainable:,} trainable / {total:,} total ({pct:.2f}%)")

    return model


def load_for_inference(
    config: TrainConfig,
    adapter_path: str | Path,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load quantized base model + merge LoRA adapter for inference."""
    model = load_base_model(config)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    logger.info(f"Loaded LoRA adapter from {adapter_path}")

    tokenizer = load_tokenizer(config)
    tokenizer.padding_side = "left"  # Left-padding for batched inference

    return model, tokenizer
