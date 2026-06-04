"""Smoke test for the MLX HTS server.

Sends one OpenAI-formatted chat completion request to localhost:8080,
prints the raw response, and parses it through parse_prediction so we can
verify both the server and the prompt format are working end-to-end.

Usage:
    # In one terminal:
    uv run python scripts/run_serve_mlx.py serve

    # In another terminal:
    uv run python scripts/smoke_test_serve.py
    uv run python scripts/smoke_test_serve.py --description "wool sweater"
"""

from __future__ import annotations

import time

import typer
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

from hts_lora.data.formatters import _SYSTEM_PROMPT
from hts_lora.inference.parse_output import parse_prediction
from hts_lora.serving.prompts import render_prompt

app = typer.Typer(help="Smoke test the local MLX HTS server")
console = Console()

DEFAULT_URL = "http://localhost:8080/v1"
DEFAULT_MODEL = "nemotron-hts-fused"  # mlx_lm.server uses the directory name


@app.command()
def main(
    url: str = typer.Option(DEFAULT_URL, help="MLX server base URL"),
    model: str = typer.Option(
        DEFAULT_MODEL, help="Model name (mlx_lm.server uses the dir name)"
    ),
    description: str = typer.Option(
        "insulated copper electrical wire, 12 AWG, stranded",
        help="Product description to classify",
    ),
    materials: str = typer.Option("copper conductor, PVC insulation", help="Materials"),
    use: str = typer.Option("residential building wiring", help="Intended use"),
    country: str = typer.Option("Mexico", help="Country of origin"),
    max_tokens: int = typer.Option(512, help="Max output tokens"),
    timeout: float = typer.Option(60.0, help="Request timeout in seconds"),
) -> None:
    """Hit the MLX server once and print + parse the response."""
    user_parts = [f"Product: {description}"]
    if materials:
        user_parts.append(f"Materials: {materials}")
    if use:
        user_parts.append(f"Use: {use}")
    if country:
        user_parts.append(f"Country of origin: {country}")
    user_content = "\n".join(user_parts)

    console.print(Panel(user_content, title="user message", border_style="blue"))

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = render_prompt(messages)

    client = OpenAI(base_url=url, api_key="not-needed", timeout=timeout)

    t0 = time.perf_counter()
    try:
        resp = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as e:
        console.print(f"[red]Request failed:[/red] {e}")
        console.print(
            f"[dim]Is the MLX server running? "
            f"Try: uv run python scripts/run_serve_mlx.py serve[/dim]"
        )
        raise typer.Exit(1)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    raw = resp.choices[0].text or ""
    console.print(Panel(raw, title="raw model output", border_style="cyan"))

    parsed = parse_prediction(raw)
    summary_lines = [
        f"parse_ok       : {parsed.parse_ok}",
        f"is_abstention  : {parsed.is_abstention}",
        f"hts_code       : {parsed.hts_code}",
        f"chapter        : {parsed.chapter_code} - {parsed.chapter_desc}",
        f"heading        : {parsed.heading_code} - {parsed.heading_desc}",
        f"subheading     : {parsed.subheading_code} - {parsed.subheading_desc}",
        f"latency_ms     : {elapsed_ms:.0f}",
    ]
    color = "green" if parsed.parse_ok else "red"
    console.print(
        Panel("\n".join(summary_lines), title="parsed", border_style=color)
    )

    if not parsed.parse_ok:
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
