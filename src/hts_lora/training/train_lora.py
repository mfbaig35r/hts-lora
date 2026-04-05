"""Main training entry point: load model, apply LoRA, train, save adapter."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from transformers import TrainingArguments, Trainer

from hts_lora.training.callbacks import JSONMetricsCallback, SamplePredictionCallback
from hts_lora.training.collators import CompletionOnlyCollator
from hts_lora.training.model_factory import apply_lora, load_base_model, load_tokenizer
from hts_lora.utils.config import TrainConfig
from hts_lora.utils.io import create_run_dir, read_jsonl, snapshot_config
from hts_lora.utils.logging import get_logger, setup_logging

logger = get_logger("training.train_lora")


def train(config: TrainConfig) -> Path:
    """Run the full LoRA fine-tuning pipeline.

    Returns the path to the saved adapter directory.
    """
    # Create run directory
    run_dir = create_run_dir(config.output_dir, prefix="train")
    setup_logging(log_file=run_dir / "train.log.jsonl")
    snapshot_config(config, run_dir)
    logger.info(f"Training run directory: {run_dir}")

    # Load model and tokenizer
    logger.info("Loading base model...")
    model = load_base_model(config)
    tokenizer = load_tokenizer(config)

    # Apply LoRA
    logger.info("Applying LoRA adapters...")
    model = apply_lora(model, config)

    # Load data
    data_dir = Path(config.data_dir)
    train_records = read_jsonl(data_dir / "train.jsonl")
    val_records = read_jsonl(data_dir / "valid.jsonl")
    logger.info(f"Data: {len(train_records)} train, {len(val_records)} valid examples")

    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)

    # Build collator
    collator = CompletionOnlyCollator(
        tokenizer=tokenizer,
        max_seq_length=config.training.max_seq_length,
    )

    # Callbacks
    callbacks = [
        JSONMetricsCallback(run_dir / "metrics.jsonl"),
    ]

    # Sample prediction callback (use first 3 val examples)
    if val_records:
        sample_inputs = val_records[:3]
        callbacks.append(
            SamplePredictionCallback(
                sample_inputs=sample_inputs,
                tokenizer=tokenizer,
                output_path=run_dir / "sample_predictions.jsonl",
            )
        )

    # Training arguments
    hp = config.training
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=hp.num_train_epochs,
        per_device_train_batch_size=hp.per_device_train_batch_size,
        per_device_eval_batch_size=hp.per_device_eval_batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        learning_rate=hp.learning_rate,
        weight_decay=hp.weight_decay,
        warmup_ratio=hp.warmup_ratio,
        lr_scheduler_type=hp.lr_scheduler_type,
        logging_steps=hp.logging_steps,
        eval_strategy="steps",
        eval_steps=hp.eval_steps,
        save_strategy="steps",
        save_steps=hp.save_steps,
        save_total_limit=3,
        seed=hp.seed,
        bf16=hp.bf16,
        fp16=hp.fp16,
        gradient_checkpointing=hp.gradient_checkpointing,
        optim=hp.optim,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # Build trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save adapter
    adapter_dir = run_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info(f"Adapter saved to {adapter_dir}")

    return adapter_dir
