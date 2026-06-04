"""Run the HTS classification FastAPI wrapper.

Usage:
    # First start the MLX server in another terminal:
    uv run python scripts/run_serve_mlx.py serve

    # Then start the FastAPI wrapper:
    uv run python scripts/run_serve_api.py
    uv run python scripts/run_serve_api.py --port 8001
    uv run python scripts/run_serve_api.py --upstream http://localhost:8080/v1
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Run the HTS classification FastAPI wrapper")
console = Console()

# Absolute path to the local fused MLX model. mlx_lm.server is lazy and treats
# the `model` field of each request as a HuggingFace repo id unless it resolves
# to a local directory, so we must send the full path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUSED_MODEL = str(PROJECT_ROOT / "models" / "nemotron-hts-fused")


@app.command()
def main(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    upstream: str = typer.Option(
        "http://localhost:8080/v1", help="Upstream MLX server URL"
    ),
    upstream_model: str = typer.Option(
        DEFAULT_FUSED_MODEL,
        help="Upstream model name (absolute path or HF repo id)",
    ),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code change"),
) -> None:
    """Start the FastAPI wrapper that proxies to the MLX server."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]uvicorn not installed. Run: uv pip install -e '.[serving]'[/red]"
        )
        raise typer.Exit(1)

    # Inject env vars before importing the app so create_app() reads them
    os.environ["HTS_MLX_URL"] = upstream
    os.environ["HTS_MLX_MODEL"] = upstream_model

    console.print(f"[bold green]HTS API[/bold green] -> {upstream} (model={upstream_model})")
    console.print(f"[dim]Listening on http://{host}:{port}[/dim]")
    console.print(f"[dim]Docs at  http://{host}:{port}/docs[/dim]\n")

    uvicorn.run(
        "hts_lora.serving.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
