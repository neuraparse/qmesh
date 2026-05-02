"""Abstract decoder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class DecodeResult:
    """Output of a decoder over a batch of shots."""

    predictions: np.ndarray              # shape (shots, observables), bool/int
    logical_error_count: int
    shots: int
    wall_seconds: float
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def logical_error_rate(self) -> float:
        return self.logical_error_count / self.shots if self.shots else 0.0


class Decoder(ABC):
    """A decoder consumes detector events and predicts observable flips."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def from_circuit(self, stim_circuit) -> "Decoder":
        """Configure this decoder for a given stim.Circuit (returns self)."""

    @abstractmethod
    def decode_batch(
        self,
        detection_events: np.ndarray,
        observable_flips: np.ndarray,
    ) -> DecodeResult:
        """Decode a batch and report logical-error count vs ground truth."""

    def identity(self) -> dict:
        """Stable identity metadata for the manifest."""
        return {"name": self.name, "version": "0.1.0"}
