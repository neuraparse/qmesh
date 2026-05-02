"""Neural-decoder training demo: Stim data → PyTorch MLP → signed weights.

Compares the trained neural decoder against PyMatching on the same surface
code at d=3 and d=5. The MLP infrastructure is faithful to how AlphaQubit
2 / open-weight successors work in 2026, but the actual quality depends on
data scale and architecture (MLP loses to MWPM at low p; transformer-class
models reportedly close the gap at high distance).

Run:
    PYTHONPATH=. python3 examples/train_neural_decoder.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rich import print
from rich.table import Table

from qmesh.ai import train_neural_decoder, generate_dataset
from qmesh.ftmode import SurfaceCode
from qmesh.ftmode.decoders.pymatching_decoder import PyMatchingDecoder


def main() -> None:
    table = Table(title="Neural decoder vs PyMatching MWPM (surface code, p=1e-3)")
    table.add_column("d", justify="right")
    table.add_column("train samples", justify="right")
    table.add_column("epochs", justify="right")
    table.add_column("train wall (s)", justify="right")
    table.add_column("val accuracy", justify="right")
    table.add_column("MWPM logical err", justify="right")
    table.add_column("Neural logical err", justify="right")

    for distance in (3, 5):
        code = SurfaceCode(distance=distance, rounds=4)
        decoder, tr = train_neural_decoder(
            code, physical_error_rate=1e-3,
            n_train=15_000, n_val=2_000, epochs=6, hidden_dim=128,
            weights_dir=Path("ai_weights"),
        )
        # Compare on a fresh validation pull
        stim_c, X, y = generate_dataset(code, n_samples=3_000)
        X_int = X.astype(np.uint8)
        y_int = y.astype(np.uint8)
        if y_int.ndim == 1:
            y_int = y_int.reshape(-1, 1)

        mwpm = PyMatchingDecoder().from_circuit(stim_c)
        res_mwpm = mwpm.decode_batch(X_int, y_int)
        res_neural = decoder.decode_batch(X_int, y_int)

        table.add_row(
            str(distance), str(tr.train_samples), str(tr.epochs),
            f"{tr.duration_seconds:.2f}",
            f"{tr.accuracy:.4f}",
            f"{res_mwpm.logical_error_rate:.4f}",
            f"{res_neural.logical_error_rate:.4f}",
        )
    print(table)
    print()
    print("[dim]Notes:[/]")
    print("[dim]  - MLP is small (hidden_dim=128, 6 epochs); production decoders use[/]")
    print("[dim]    transformers + 10⁶+ samples per (d, p) pair (AlphaQubit-2 setup).[/]")
    print("[dim]  - Weights persisted under ./ai_weights/ with sha256 in their filename;[/]")
    print("[dim]    NeuralDecoder.identity() reports trained=True so FT-mode manifests[/]")
    print("[dim]    audit the exact decoder used.[/]")


if __name__ == "__main__":
    main()
