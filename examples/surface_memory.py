"""Surface-code memory experiment (Stim-native).

This demonstrates the FT-mode-style flow: encode a logical |0>, run rounds of
syndrome extraction under noise, and decode the syndrome to estimate the
logical error rate per round. Uses Stim + PyMatching directly because the
qmesh.ir gate-modality is too high-level for surface-code circuits today —
Phase 2 will introduce a `qmesh.ftmode.SurfaceCode` IR primitive that lowers
to this for any backend that can run it.

Run:
    PYTHONPATH=. python3 examples/surface_memory.py
"""

from __future__ import annotations

import time

import numpy as np
import stim
from rich import print
from rich.table import Table


def measure_logical_error(distance: int, rounds: int, p: float, shots: int) -> float:
    """Sample shots of a memory experiment + decode + return logical error rate."""
    try:
        import pymatching
    except ImportError:
        print("[yellow]pymatching not installed; surface-memory demo limited[/]")
        return float("nan")

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(shots=shots, separate_observables=True)

    # build a matching graph from the detector error model
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    predictions = matcher.decode_batch(detection_events)
    n_errors = int(np.sum(predictions != observable_flips))
    return n_errors / shots


def main() -> None:
    table = Table(title="Surface-code memory experiment (Stim + PyMatching)")
    table.add_column("d", justify="right")
    table.add_column("rounds", justify="right")
    table.add_column("p (phys)", justify="right")
    table.add_column("logical err / shot", justify="right")
    table.add_column("Λ (vs d=3)", justify="right")
    table.add_column("wall (s)", justify="right")

    p = 0.001
    rounds = 8
    shots = 30_000
    base = None
    for d in (3, 5, 7):
        t0 = time.time()
        le = measure_logical_error(d, rounds, p, shots)
        wall = time.time() - t0
        if base is None:
            base = le
            ratio_str = "—"
        else:
            ratio_str = f"{base / le:.2f}" if le > 0 else "∞"
        table.add_row(str(d), str(rounds), f"{p:.0e}", f"{le:.5f}", ratio_str, f"{wall:.2f}")
    print(table)
    print()
    print("[dim]Λ should grow > 1 with distance (suppression). Google Willow's[/]")
    print("[dim]experimental Λ ≈ 2.14 between d=3,5,7 (Nature 2024).[/]")


if __name__ == "__main__":
    main()
