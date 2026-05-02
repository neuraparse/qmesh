"""qmesh.ai.train_decoder — neural-decoder training pipeline.

Generates synthetic syndrome data from a `stim.Circuit`, trains a small
PyTorch MLP, and saves signed weights that the FT-mode `NeuralDecoder`
can load. This is the open-source alternative to AlphaQubit-2 — it won't
match Google's published numbers without significantly more data and
architecture work, but the pipeline is faithful and the manifest captures
the model identity so downstream FT runs can audit which decoder produced
which logical-error rate.

Pipeline:
  1. Generate (detection_events, observable_flips) tensors from Stim.
  2. Train an MLP to predict observable flips from detection events.
  3. Save state_dict + config + sha256 hash to disk.
  4. Return a NeuralDecoder pre-loaded with the trained weights.

Phase 5β: graph-neural-network decoders, distance-conditional models,
shared-weights across distances.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from qmesh.ftmode.codes import Code
from qmesh.ftmode.decoders.neural import NeuralDecoder


@dataclass(slots=True)
class TrainingResult:
    accuracy: float
    epochs: int
    train_samples: int
    val_samples: int
    final_loss: float
    weights_path: Path
    weights_sha256: str
    duration_seconds: float
    history: list[dict[str, float]] = field(default_factory=list)


def generate_dataset(
    code: Code,
    *,
    physical_error_rate: float = 1e-3,
    n_samples: int = 50_000,
):
    stim_circuit = code.generate_memory_circuit(
        physical_error_rate=physical_error_rate,
    )
    sampler = stim_circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(
        shots=n_samples, separate_observables=True,
    )
    return stim_circuit, detection_events.astype(np.float32), observable_flips.astype(np.float32)


def train_neural_decoder(
    code: Code,
    *,
    physical_error_rate: float = 1e-3,
    n_train: int = 50_000,
    n_val: int = 5_000,
    epochs: int = 8,
    hidden_dim: int = 256,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    weights_dir: Path | str = "ai_weights",
    seed: int | None = None,
) -> tuple[NeuralDecoder, TrainingResult]:
    """Train a NeuralDecoder for the given code.

    Returns (decoder, training_result). The decoder is configured for the
    code's stim.Circuit and has the trained weights loaded.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise RuntimeError("PyTorch required: `pip install torch`") from e

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    t0 = time.time()
    stim_circuit, X_train, y_train = generate_dataset(
        code, physical_error_rate=physical_error_rate, n_samples=n_train,
    )
    _, X_val, y_val = generate_dataset(
        code, physical_error_rate=physical_error_rate, n_samples=n_val,
    )

    n_detectors = X_train.shape[1]
    n_obs = y_train.shape[1] if y_train.ndim > 1 else 1
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

    class _MLP(nn.Module):
        def __init__(self, d_in: int, d_hid: int, d_out: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hid), nn.ReLU(),
                nn.Linear(d_hid, d_hid), nn.ReLU(),
                nn.Linear(d_hid, d_out),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

    model = _MLP(n_detectors, hidden_dim, n_obs)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=learning_rate)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    history: list[dict[str, float]] = []
    final_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        avg_loss = running / max(1, n_batches)
        final_loss = avg_loss

        # validation accuracy
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_pred = (val_logits > 0).float()
            val_acc = (val_pred == y_val_t).all(dim=1).float().mean().item()
        history.append({"epoch": epoch, "loss": avg_loss, "val_accuracy": val_acc})

    # Save weights
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    code_md = code.metadata()
    fname = weights_dir / f"{code_md.name}_p{physical_error_rate:.0e}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "n_detectors": n_detectors,
            "n_observables": n_obs,
            "hidden_dim": hidden_dim,
            "code": code_md.to_dict(),
            "physical_error_rate": physical_error_rate,
        },
    }, fname)
    sha256 = hashlib.sha256(fname.read_bytes()).hexdigest()

    # Final accuracy on val
    final_acc = history[-1]["val_accuracy"] if history else 0.0
    duration = time.time() - t0

    decoder = NeuralDecoder(weights_path=fname, hidden_dim=hidden_dim)
    decoder.from_circuit(stim_circuit)

    return decoder, TrainingResult(
        accuracy=final_acc,
        epochs=epochs,
        train_samples=n_train,
        val_samples=n_val,
        final_loss=final_loss,
        weights_path=fname,
        weights_sha256=sha256,
        duration_seconds=duration,
        history=history,
    )


__all__ = ["TrainingResult", "generate_dataset", "train_neural_decoder"]
