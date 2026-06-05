"""Convert an HF PEFT LoRA adapter to mlx_lm-compatible format.

HF PEFT layout:
    adapter_config.json:  {peft_type: LORA, r, lora_alpha, target_modules, ...}
    adapter_model.safetensors:
        keys like  base_model.model.model.layers.{N}.{mlp|self_attn}.{X}_proj.lora_{A|B}.weight
        lora_A.weight shape: (rank, in_features)
        lora_B.weight shape: (out_features, rank)

mlx_lm layout:
    adapter_config.json:  {fine_tune_type: lora, num_layers, lora_parameters: {rank, scale, dropout}, ...}
    adapters.safetensors:
        keys like  model.layers.{N}.{mlp|self_attn}.{X}_proj.lora_{a|b}
        lora_a shape: (in_features, rank)     <- transposed from HF lora_A
        lora_b shape: (rank, out_features)    <- transposed from HF lora_B

Both formats represent the same LoRA delta:
    HF:  delta = (alpha/rank) * x @ A.T @ B.T
    MLX: delta = scale * x @ a @ b
    Setting scale = alpha/rank and a = A.T, b = B.T makes them equivalent.

Usage:
    python scripts/convert_hf_adapter_to_mlx.py \\
        --hf-adapter outputs/train_h100_20260406/adapter \\
        --out adapters_v1_mlx
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import typer
from rich.console import Console
from safetensors import safe_open
from safetensors.numpy import save_file as save_safetensors_numpy

app = typer.Typer(help="Convert HF PEFT LoRA adapter to mlx_lm format")
console = Console()


_HF_KEY_RE = re.compile(
    r"^base_model\.model\.model\.layers\.(\d+)\.([\w\.]+)\.lora_([AB])\.weight$"
)


def _convert_key(hf_key: str) -> str | None:
    """Map an HF PEFT weight key to its mlx_lm equivalent. Return None for keys
    that aren't recognized layer LoRA weights."""
    m = _HF_KEY_RE.match(hf_key)
    if not m:
        return None
    layer_idx, module_path, ab = m.groups()
    return f"model.layers.{layer_idx}.{module_path}.lora_{ab.lower()}"


@app.command()
def main(
    hf_adapter: str = typer.Option(..., help="Path to HF PEFT adapter directory"),
    out: str = typer.Option(..., help="Output directory for the MLX adapter"),
) -> None:
    src = Path(hf_adapter)
    src_cfg_path = src / "adapter_config.json"
    src_weights_path = src / "adapter_model.safetensors"
    if not src_cfg_path.exists() or not src_weights_path.exists():
        raise typer.Exit(f"Missing adapter_config.json or adapter_model.safetensors under {src}")

    hf_cfg = json.loads(src_cfg_path.read_text())
    rank = int(hf_cfg["r"])
    alpha = int(hf_cfg["lora_alpha"])
    dropout = float(hf_cfg.get("lora_dropout", 0.0))
    target_modules = list(hf_cfg.get("target_modules", []))

    dst = Path(out)
    dst.mkdir(parents=True, exist_ok=True)

    # Build the MLX safetensors: rename keys, transpose values, store as numpy.
    converted: dict[str, "np.ndarray"] = {}
    layer_idxs: set[int] = set()
    skipped: list[str] = []

    import numpy as np  # local import keeps top-of-file lean

    with safe_open(str(src_weights_path), framework="np") as f:
        for hf_key in f.keys():
            mlx_key = _convert_key(hf_key)
            if mlx_key is None:
                skipped.append(hf_key)
                continue
            arr = f.get_tensor(hf_key)
            # Both lora_A (rank, in) and lora_B (out, rank) get transposed.
            arr = np.ascontiguousarray(arr.T)
            converted[mlx_key] = arr
            # Track which layer indices we saw
            mlayer = re.match(r"^model\.layers\.(\d+)\.", mlx_key)
            if mlayer:
                layer_idxs.add(int(mlayer.group(1)))

    if not converted:
        raise typer.Exit("No matching LoRA weights found in source adapter")

    save_safetensors_numpy(converted, str(dst / "adapters.safetensors"))

    # MLX expects num_layers = the TOP-N layers to wrap with LoRA. v1 trained
    # all 32 layers; max(layer_idxs)+1 captures that.
    num_layers = max(layer_idxs) + 1
    if min(layer_idxs) != 0:
        console.print(
            f"[yellow]Warning: lowest layer index is {min(layer_idxs)} (not 0). "
            "MLX wraps the top-N layers; if v1 skipped some lower layers this "
            "conversion still wraps them with zero-init weights (no-op)."
            "[/yellow]"
        )

    mlx_cfg = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": rank,
            "scale": alpha / rank,
            "dropout": dropout,
        },
        "_source": {
            "kind": "converted_from_hf_peft",
            "hf_adapter_path": str(src),
            "target_modules": target_modules,
            "alpha": alpha,
            "n_keys": len(converted),
        },
    }
    (dst / "adapter_config.json").write_text(json.dumps(mlx_cfg, indent=2))

    # Copy the tokenizer files alongside so mlx_lm.load can find them
    for fname in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        srcf = src / fname
        if srcf.exists():
            (dst / fname).write_bytes(srcf.read_bytes())

    console.print(f"\n[bold]Converted v1 LoRA -> MLX[/bold]")
    console.print(f"  Source:      {src}")
    console.print(f"  Output:      {dst}")
    console.print(f"  Rank/alpha:  {rank} / {alpha} (scale = {alpha/rank:.3f})")
    console.print(f"  Target mods: {target_modules}")
    console.print(f"  Layers seen: {len(layer_idxs)} ({min(layer_idxs)}..{max(layer_idxs)})")
    console.print(f"  num_layers:  {num_layers}")
    console.print(f"  Keys mapped: {len(converted)}")
    if skipped:
        console.print(f"  [yellow]Skipped {len(skipped)} non-LoRA keys[/yellow]")


if __name__ == "__main__":
    app()
