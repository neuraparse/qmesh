"""Side-by-side: MLP vs transformer syndrome decoder on d=3 surface code.

Both decoders train on Stim-generated detection events at p=1e-3 and slot
into the same `qmesh.ftmode.decoders.NeuralDecoder` interface — the only
thing that changes is the underlying nn.Module. The MLP is the Phase 5α
baseline; the transformer is α-quality but architecturally faithful to
AlphaQubit-2 (arXiv 2512.07737, March 2026), which uses the same
sequence-of-rounds → encoder → observable head shape at ~100M params.

Run:
    PYTHONPATH=. python3 examples/transformer_decoder.py
"""

from __future__ import annotations

from pathlib import Path

from rich import print
from rich.table import Table

from qmesh.ai import train_neural_decoder, train_transformer_decoder
from qmesh.ftmode import SurfaceCode


def main() -> None:
    code = SurfaceCode(distance=3, rounds=3)
    weights_dir = Path("ai_weights")

    print("[bold cyan]Training MLP decoder...[/]")
    _mlp_decoder, mlp_tr = train_neural_decoder(
        code, physical_error_rate=1e-3,
        n_train=5_000, n_val=1_000, epochs=4, hidden_dim=128,
        weights_dir=weights_dir, seed=42,
    )
    print(f"  done: val_acc={mlp_tr.accuracy:.4f}, wall={mlp_tr.duration_seconds:.2f}s")

    print("[bold cyan]Training transformer decoder...[/]")
    _xfmr_decoder, xfmr_tr = train_transformer_decoder(
        code, physical_error_rate=1e-3,
        n_train=5_000, n_val=1_000, epochs=4,
        d_model=64, n_heads=4, n_layers=2,
        weights_dir=weights_dir, seed=42,
    )
    print(f"  done: val_acc={xfmr_tr.accuracy:.4f}, wall={xfmr_tr.duration_seconds:.2f}s")

    table = Table(title="MLP vs transformer syndrome decoder (d=3, p=1e-3, seed=42)")
    table.add_column("model", style="cyan")
    table.add_column("val acc", justify="right")
    table.add_column("train wall (s)", justify="right")
    table.add_column("epochs", justify="right")
    table.add_column("samples", justify="right")
    table.add_column("weights sha256")
    table.add_column("manifest path", overflow="fold", max_width=40)

    table.add_row(
        "MLP (hidden=128)",
        f"{mlp_tr.accuracy:.4f}",
        f"{mlp_tr.duration_seconds:.2f}",
        str(mlp_tr.epochs),
        str(mlp_tr.train_samples),
        mlp_tr.weights_sha256[:18] + "…",
        str(mlp_tr.weights_path),
    )
    table.add_row(
        "Transformer (d=64,L=2,H=4)",
        f"{xfmr_tr.accuracy:.4f}",
        f"{xfmr_tr.duration_seconds:.2f}",
        str(xfmr_tr.epochs),
        str(xfmr_tr.train_samples),
        xfmr_tr.weights_sha256[:18] + "…",
        str(xfmr_tr.weights_path),
    )
    print(table)
    print()
    print("[dim]Notes:[/]")
    print("[dim]  - AlphaQubit-2 uses 100M+ params trained on hundreds of millions of[/]")
    print("[dim]    real Sycamore shots; this demo trains a 50K-param transformer on[/]")
    print("[dim]    5K Stim shots in seconds — the architecture is faithful, the scale is α.[/]")
    print("[dim]  - Both decoders save sha256-signed weights and slot into the same[/]")
    print("[dim]    NeuralDecoder interface so FT-mode manifests audit the exact model.[/]")


if __name__ == "__main__":
    main()
