"""PyMatching MWPM decoder — Phase-2α default.

PyMatching is the standard MWPM decoder for surface codes; it ingests Stim's
DetectorErrorModel and runs minimum-weight perfect matching on the syndrome
graph. Sub-millisecond per shot at d=11 on a workstation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from qmesh.ftmode.decoders.base import Decoder, DecodeResult


@dataclass
class PyMatchingDecoder(Decoder):
    use_uncorrelated_detector_error_model: bool = False

    _matcher: object | None = None
    _circuit_repr: str | None = None

    @property
    def name(self) -> str:
        return "pymatching-mwpm"

    def from_circuit(self, stim_circuit) -> "PyMatchingDecoder":
        import pymatching

        dem = stim_circuit.detector_error_model(
            decompose_errors=True,
            ignore_decomposition_failures=True,
        )
        self._matcher = pymatching.Matching.from_detector_error_model(dem)
        self._circuit_repr = str(stim_circuit)[:512]
        return self

    def decode_batch(
        self,
        detection_events: np.ndarray,
        observable_flips: np.ndarray,
    ) -> DecodeResult:
        if self._matcher is None:
            raise RuntimeError("decoder not configured; call from_circuit() first")
        t0 = time.time()
        predictions = self._matcher.decode_batch(detection_events)
        n_errors = int(np.sum(predictions != observable_flips))
        return DecodeResult(
            predictions=predictions,
            logical_error_count=n_errors,
            shots=detection_events.shape[0],
            wall_seconds=time.time() - t0,
            metadata={"name": self.name},
        )

    def identity(self) -> dict:
        import pymatching
        return {
            "name": self.name,
            "version": getattr(pymatching, "__version__", "unknown"),
            "uncorrelated": self.use_uncorrelated_detector_error_model,
        }
