"""Neural decoder skeleton (PyTorch).

Phase-2α delivers the *interface* and a tiny MLP that can be trained on
Stim-generated detector data. The actual AlphaQubit-2 weights aren't open
as of May 2026; when they ship, dropping them in is a matter of changing
the model class and `identity()`.

Training pipeline is in `qmesh.ai.train_decoder` (Phase 5). For now the
decoder runs uninitialised (random) and reports its untrained state in the
manifest — useful for plumbing tests and accuracy comparisons.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qmesh.ftmode.decoders.base import Decoder, DecodeResult


@dataclass
class NeuralDecoder(Decoder):
    """A 3-layer MLP over flattened detector events. Tiny but pluggable."""

    weights_path: Path | None = None
    hidden_dim: int = 256

    _model: object | None = None
    _input_dim: int | None = None
    _output_dim: int | None = None
    _trained: bool = False

    @property
    def name(self) -> str:
        return "neural-mlp-v0"

    def from_circuit(self, stim_circuit) -> "NeuralDecoder":
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise RuntimeError(
                "neural decoder requires PyTorch: `pip install torch`."
            ) from e

        dem = stim_circuit.detector_error_model()
        n_detectors = dem.num_detectors
        n_obs = dem.num_observables
        self._input_dim = n_detectors
        self._output_dim = n_obs

        class _MLP(nn.Module):
            def __init__(self, d_in: int, d_hid: int, d_out: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d_in, d_hid), nn.ReLU(),
                    nn.Linear(d_hid, d_hid), nn.ReLU(),
                    nn.Linear(d_hid, d_out),
                )

            def forward(self, x):
                return self.net(x)

        self._model = _MLP(n_detectors, self.hidden_dim, n_obs)
        if self.weights_path and Path(self.weights_path).exists():
            state = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            # Accept both raw state_dict and our training-pipeline format
            if isinstance(state, dict) and "state_dict" in state:
                self._model.load_state_dict(state["state_dict"])
            else:
                self._model.load_state_dict(state)
            self._trained = True
        return self

    def decode_batch(
        self,
        detection_events: np.ndarray,
        observable_flips: np.ndarray,
    ) -> DecodeResult:
        import torch

        if self._model is None:
            raise RuntimeError("call from_circuit() first")

        t0 = time.time()
        x = torch.from_numpy(detection_events.astype(np.float32))
        with torch.no_grad():
            logits = self._model(x).numpy()
        predictions = (logits > 0).astype(np.uint8)
        # ensure 2-D for compatibility
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        n_errors = int(np.sum(predictions != observable_flips))
        return DecodeResult(
            predictions=predictions,
            logical_error_count=n_errors,
            shots=detection_events.shape[0],
            wall_seconds=time.time() - t0,
            metadata={
                "name": self.name,
                "trained": self._trained,
                "input_dim": self._input_dim,
                "hidden_dim": self.hidden_dim,
                "warning": (
                    "untrained" if not self._trained
                    else "weights loaded from disk"
                ),
            },
        )

    def identity(self) -> dict:
        return {
            "name": self.name,
            "trained": self._trained,
            "weights_path": str(self.weights_path) if self.weights_path else None,
        }
