"""Sliding-window MWPM decoder.

Approximates the streaming/real-time decoder pattern: the decoder window
slides over fixed-size chunks of syndrome rounds. In practice on a real
device this enables sub-µs incremental decoding (Riverlane Deltaflow,
Google's real-time decoder pipeline). Here it serves as a research-grade
emulation that records the same identity / latency fields the manifest
expects, even if execution is offline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from qmesh.ftmode.decoders.base import DecodeResult, Decoder


@dataclass
class SlidingWindowMWPMDecoder(Decoder):
    window_rounds: int = 4
    overlap_rounds: int = 1

    _matcher: object | None = None

    @property
    def name(self) -> str:
        return f"sliding_mwpm/window={self.window_rounds}+overlap={self.overlap_rounds}"

    def from_circuit(self, stim_circuit) -> "SlidingWindowMWPMDecoder":
        import pymatching

        dem = stim_circuit.detector_error_model(
            decompose_errors=True,
            ignore_decomposition_failures=True,
        )
        self._matcher = pymatching.Matching.from_detector_error_model(dem)
        return self

    def decode_batch(
        self,
        detection_events: np.ndarray,
        observable_flips: np.ndarray,
    ) -> DecodeResult:
        # Phase-2α stub: forward to the global MWPM, but record the sliding-
        # window identity so manifests are populated. A true sliding-window
        # decoder needs incremental graph construction tied to the device's
        # syndrome stream — Phase 2β work.
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
            metadata={
                "name": self.name,
                "window_rounds": self.window_rounds,
                "overlap_rounds": self.overlap_rounds,
                "implementation": "global_mwpm_emulation_for_2α",
            },
        )

    def identity(self) -> dict:
        return {
            "name": self.name,
            "window_rounds": self.window_rounds,
            "overlap_rounds": self.overlap_rounds,
            "implementation": "global_mwpm_emulation_for_2α",
        }
