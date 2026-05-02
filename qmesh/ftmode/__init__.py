"""qmesh.ftmode — fault-tolerant mode (Phase 2α).

Public API:
    FTConfig                           — configure code, decoder, cultivation
    promote_and_run(module, ftconfig)  — execute under FT mode end-to-end
    memory_experiment(...)             — simple FT memory experiment
    threshold_sweep(...)               — sweep distance to find Λ
    estimate(module, code=...)         — Microsoft-RE-shaped resource estimate

Codes:    SurfaceCode, UnrotatedSurfaceCode, RepetitionCode, BBCode (stub)
Decoders: PyMatchingDecoder, SlidingWindowMWPMDecoder, NeuralDecoder (skeleton)
Factory:  InPlaceCultivation, DistillationFactory

The full architecture is documented in ARCHITECTURE.md §5. The thesis is
that FT mode should be a one-flag promotion — pick code/decoder/cultivation
per backend automatically, capture every choice in a signed manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from qmesh.ftmode.codes import (
    BBCode,
    Code,
    CodeMetadata,
    RepetitionCode,
    SurfaceCode,
    UnrotatedSurfaceCode,
)
from qmesh.ftmode.cultivation import (
    CultivationParams,
    DistillationFactory,
    InPlaceCultivation,
    MagicStateFactory,
)
from qmesh.ftmode.decoders import (
    BpOsdDecoder,
    Decoder,
    DecodeResult,
    PyMatchingDecoder,
    SlidingWindowMWPMDecoder,
    StreamingMWPMDecoder,
    get_decoder,
)
from qmesh.ftmode.estimate import (
    ResourceEstimate,
    estimate,
    estimate_for_distances,
)


class CodeChoice(str, Enum):
    AUTO = "auto"
    SURFACE = "surface"
    UNROTATED_SURFACE = "unrotated_surface"
    REPETITION = "repetition"
    BB_QLDPC = "bb"
    FLOQUET = "floquet"
    HYPERBOLIC_FLOQUET = "hyperbolic_floquet"


class DecoderChoice(str, Enum):
    PYMATCHING = "pymatching"
    SLIDING_WINDOW = "sliding_window"
    STREAMING = "streaming"
    BP_OSD = "bp_osd"
    UNION_FIND = "union_find"
    ALPHAQUBIT2 = "alphaqubit2"
    NEURAL_OPEN = "neural_open"


@dataclass(slots=True)
class FTConfig:
    code: CodeChoice = CodeChoice.SURFACE
    distance: int = 5
    rounds: int = 8
    decoder: DecoderChoice = DecoderChoice.PYMATCHING
    cultivation: MagicStateFactory | None = None
    target_logical_error: float = 1e-9
    physical_error_rate: float = 1e-3
    basis: str = "Z"

    def __post_init__(self) -> None:
        if self.cultivation is None:
            self.cultivation = InPlaceCultivation()

    def build_code(self) -> Code:
        ch = self.code
        if isinstance(ch, str):
            ch = CodeChoice(ch)
        if ch == CodeChoice.SURFACE or ch == CodeChoice.AUTO:
            return SurfaceCode(distance=self.distance, rounds=self.rounds)
        if ch == CodeChoice.UNROTATED_SURFACE:
            return UnrotatedSurfaceCode(distance=self.distance, rounds=self.rounds)
        if ch == CodeChoice.REPETITION:
            return RepetitionCode(distance=self.distance, rounds=self.rounds)
        if ch == CodeChoice.BB_QLDPC:
            return BBCode(distance=self.distance, rounds=self.rounds)
        raise NotImplementedError(f"FT code {ch.value} is Phase 2β work")

    def build_decoder(self) -> Decoder:
        return get_decoder(self.decoder.value if isinstance(self.decoder, DecoderChoice) else self.decoder)


from qmesh.ftmode.runner import (  # noqa: E402  (after FTConfig defined)
    memory_experiment,
    promote_and_run,
    threshold_sweep,
)

__all__ = [
    "FTConfig", "CodeChoice", "DecoderChoice",
    "Code", "CodeMetadata", "SurfaceCode", "UnrotatedSurfaceCode", "RepetitionCode", "BBCode",
    "Decoder", "DecodeResult", "BpOsdDecoder", "PyMatchingDecoder",
    "SlidingWindowMWPMDecoder", "StreamingMWPMDecoder", "get_decoder",
    "CultivationParams", "MagicStateFactory", "InPlaceCultivation", "DistillationFactory",
    "ResourceEstimate", "estimate", "estimate_for_distances",
    "promote_and_run", "memory_experiment", "threshold_sweep",
]
