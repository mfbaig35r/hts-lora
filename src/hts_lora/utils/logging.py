"""Structured logging with rich console and JSON file output."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)


class JSONFileHandler(logging.Handler):
    """Handler that writes structured JSON log lines to a file."""

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "data"):
            entry["data"] = record.data  # type: ignore[attr-defined]
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            self.handleError(record)


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure root logger with rich console and optional JSON file output.

    Returns the 'hts_lora' logger.
    """
    logger = logging.getLogger("hts_lora")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    # Rich console handler
    console_handler = RichHandler(
        console=_console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
    )
    console_handler.setLevel(getattr(logging, level.upper()))
    logger.addHandler(console_handler)

    # JSON file handler
    if log_file:
        json_handler = JSONFileHandler(log_file)
        json_handler.setLevel(logging.DEBUG)
        logger.addHandler(json_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a child logger under hts_lora."""
    base = "hts_lora"
    if name:
        return logging.getLogger(f"{base}.{name}")
    return logging.getLogger(base)


def log_data(logger: logging.Logger, message: str, data: dict[str, Any], level: int = logging.INFO) -> None:
    """Log a message with attached structured data."""
    record = logger.makeRecord(
        logger.name, level, "(data)", 0, message, (), None
    )
    record.data = data  # type: ignore[attr-defined]
    logger.handle(record)
