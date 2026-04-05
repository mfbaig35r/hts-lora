"""I/O utilities for JSONL, YAML, and run directory management."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of dicts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    """Append a single record to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Write a dict to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write JSON to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with open(path) as f:
        return json.load(f)


def create_run_dir(base_dir: str | Path, prefix: str = "run") -> Path:
    """Create a timestamped run directory.

    Returns the created directory path.
    """
    base = Path(base_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Update the 'latest' symlink (safe for multi-process)
    latest = base / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)
    except (OSError, FileExistsError, FileNotFoundError):
        pass  # Another process may have raced us

    return run_dir


def snapshot_config(config: Any, run_dir: str | Path) -> Path:
    """Save a Pydantic config snapshot to a run directory."""
    run_dir = Path(run_dir)
    out_path = run_dir / "config_snapshot.yaml"
    data = config.model_dump() if hasattr(config, "model_dump") else dict(config)
    write_yaml(data, out_path)
    return out_path


def copy_to_run(src: str | Path, run_dir: str | Path) -> Path:
    """Copy a file into a run directory."""
    src = Path(src)
    dst = Path(run_dir) / src.name
    shutil.copy2(src, dst)
    return dst
