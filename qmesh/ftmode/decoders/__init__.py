"""qmesh.ftmode.decoders — pluggable decoders for FT-mode execution.

A `Decoder` consumes detector events (from a `stim.DetectorErrorModel`) and
predicts logical-observable flips. The framework is decoder-agnostic so users
can swap PyMatching (Phase 2α default), Union-Find, sliding-window MWPM
(approximating real-time), or a learned neural decoder (skeleton today,
trainable in Phase 5) without touching the runner.

Each decoder records its identity in the FT manifest's `decoder` field so
runs are replayable and auditable.
"""

from __future__ import annotations

from qmesh.ftmode.decoders.base import Decoder, DecodeResult
from qmesh.ftmode.decoders.bp_osd import BpOsdDecoder
from qmesh.ftmode.decoders.pymatching_decoder import PyMatchingDecoder
from qmesh.ftmode.decoders.sliding_window import SlidingWindowMWPMDecoder
from qmesh.ftmode.decoders.streaming import StreamingMWPMDecoder

__all__ = [
    "Decoder",
    "DecodeResult",
    "BpOsdDecoder",
    "PyMatchingDecoder",
    "SlidingWindowMWPMDecoder",
    "StreamingMWPMDecoder",
    "get_decoder",
]


def get_decoder(name: str, **kwargs) -> Decoder:
    """Look up a decoder by short name."""
    name = name.lower()
    if name in ("pymatching", "mwpm"):
        return PyMatchingDecoder(**kwargs)
    if name in ("sliding_window", "sliding-mwpm", "sliding_mwpm"):
        return SlidingWindowMWPMDecoder(**kwargs)
    if name in ("streaming", "streaming_mwpm", "streaming-mwpm"):
        return StreamingMWPMDecoder(**kwargs)
    if name in ("bp_osd", "bposd", "bp-osd"):
        return BpOsdDecoder(**kwargs)
    if name in ("neural", "alphaqubit2"):
        from qmesh.ftmode.decoders.neural import NeuralDecoder
        return NeuralDecoder(**kwargs)
    raise ValueError(f"unknown decoder: {name!r}. "
                     f"Try: pymatching, sliding_window, streaming, bp_osd, neural")
