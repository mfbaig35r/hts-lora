"""Fresh MLX inference on a small slice, fed into the v2 eval pipeline.

Smoke test for the inference+eval loop, end to end:
  data/pilot/test.jsonl  -->  mlx_lm.generate  -->  parse_prediction
                          -->  generate_report  -->  outputs/.../eval_smoketest_mlx/

Uses the MLX pilot adapter at adapters/. Produces real numbers on fresh
output (no truncation, unlike run_pilot_mlx.py:263). The pilot adapter is
weak (rank 8, 3000 iters) so the numbers themselves aren't the v1 numbers,
but they prove the inference+parse+metrics+reports loop closes.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console

from hts_lora.evaluation.reports import generate_report
from hts_lora.inference.parse_output import parse_prediction
from hts_lora.utils.io import read_jsonl

DEFAULT_MODEL = "bourn23/nvidia-llama-3.1-nemotron-nano-8b-v1-mlx-4bit"
DEFAULT_ADAPTER = "adapters"
DEFAULT_DATA = "data/pilot/test.jsonl"

app = typer.Typer(help="Fresh MLX inference + v2 eval pipeline smoke test")
console = Console()


@app.command()
def main(
    model_id: str = typer.Option(DEFAULT_MODEL),
    adapter_path: str = typer.Option(DEFAULT_ADAPTER),
    data_path: str = typer.Option(DEFAULT_DATA),
    num_examples: int = typer.Option(100, help="How many test examples to run"),
    max_tokens: int = typer.Option(512),
    output_dir: str = typer.Option("outputs/mlx_pilot/eval_smoketest_mlx"),
) -> None:
    try:
        from mlx_lm import generate as mlx_generate
        from mlx_lm import load as mlx_load
    except ImportError as e:
        raise typer.Exit(f"mlx_lm not available: {e}")

    examples = read_jsonl(data_path)[:num_examples]
    console.print(f"Loaded {len(examples)} examples from {data_path}")

    console.print(f"Loading {model_id} + adapter={adapter_path}")
    model, tokenizer = mlx_load(model_id, adapter_path=adapter_path)

    eval_records: list[dict] = []
    parse_ok = 0
    for i, ex in enumerate(examples):
        messages = ex["messages"][:2]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt += "<think>\n</think>\n\n"

        generated = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
        parsed = parse_prediction(generated)
        if parsed.parse_ok:
            parse_ok += 1

        eval_records.append({
            "hts_code": ex.get("hts_code", ""),
            "prediction": asdict(parsed),
            "parse_ok": parsed.parse_ok,
            "abstain": ex.get("abstain", False),
            "description": ex["messages"][1]["content"][:200],
            "raw": generated,
            "ruling_number": ex.get("ruling_number"),
            "input_variant": ex.get("input_variant"),
        })

        if (i + 1) % 10 == 0:
            console.print(f"  [{i + 1}/{len(examples)}] parse_ok={parse_ok}/{i + 1}")

    out = Path(output_dir)
    report = generate_report(eval_records, out)

    m = report["metrics"]
    console.print("\n[bold]MLX pilot adapter on {} test examples:[/bold]".format(len(examples)))
    console.print(f"  Parse rate:             {m.get('parse_rate', 0):.1%}")
    console.print(f"  Exact match:            {m.get('exact_match', 0):.3f}")
    console.print(f"  Chapter match:          {m.get('chapter_match', 0):.3f}")
    console.print(f"  Heading match:          {m.get('heading_match', 0):.3f}")
    console.print(f"  Subheading match:       {m.get('subheading_match', 0):.3f}")
    console.print(f"  Hierarchy consistency:  {m.get('hierarchy_consistency', 0):.3f}")
    console.print(f"  Abstain rate (on GT):   {m.get('abstain_rate', 0):.3f}")
    console.print(f"\n[green]Report: {out}[/green]")


if __name__ == "__main__":
    app()
