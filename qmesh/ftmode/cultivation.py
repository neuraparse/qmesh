"""qmesh.ftmode.cultivation — magic-state factory interface.

Phase 2δ adds :func:`emit_cultivation_block`, a real Stim sub-circuit that
models the residual T-state error as a per-block X_ERROR channel on a
dedicated cultivation ancilla. Each block contributes one observable
(``OBSERVABLE_INCLUDE``) representing the cultivation outcome — auditors
can match against the cultivation observables independently of the
patch-level memory observables.


Phase-2α: provides the abstract factory + a simple `InPlaceCultivation`
baseline that records the right manifest fields. The actual surface-code
cultivation circuit construction (Gidney & Shutty arXiv 2409.17595) is
a Phase-2β engineering item and is correctly delegated to upstream Stim
+ Crumble research code today.

Why even ship a stub now? Because:
1. The manifest needs to record cultivation parameters per-run for
   reproducibility, regardless of which factory is in use.
2. The resource estimator already accounts for cultivation overhead;
   keeping the symbol means future swap-in is a one-line change.
3. Users running on simulators can exercise the API end-to-end.

Reference: Gidney & Shutty, "Magic state cultivation: growing T states as
cheap as CNOT gates" (arXiv 2409.17595, 2024). Cited result: 4×10⁻¹¹
logical error per T-state at 5×10⁻⁴ physical noise, ~10× qubit-round
savings vs distillation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import stim


@dataclass(slots=True)
class CultivationParams:
    method: str = "in_place"
    target_logical_T_error: float = 4e-11
    physical_error_rate: float = 5e-4
    cycles_per_T: int = 50
    physical_qubits_per_T_pool: int = 100
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "target_logical_T_error": self.target_logical_T_error,
            "physical_error_rate": self.physical_error_rate,
            "cycles_per_T": self.cycles_per_T,
            "physical_qubits_per_T_pool": self.physical_qubits_per_T_pool,
            "notes": self.notes,
        }


class MagicStateFactory(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def params(self) -> CultivationParams: ...

    def manifest_block(self, n_T_states: int) -> dict:
        """Produce the cultivation block embedded in the FT manifest."""
        p = self.params()
        return {
            "factory": self.name,
            "T_states_used": n_T_states,
            **p.to_dict(),
        }


class InPlaceCultivation(MagicStateFactory):
    """Gidney-shaped in-place cultivation parameters.

    Phase 2α: parameter capture only — actual circuit emission is delegated
    to Stim's research community. Future Phase-2β: emit a Stim circuit that
    grows a T-state inside a surface patch and post-selects on the cultivation
    syndrome, matching the protocol in arXiv 2409.17595.
    """

    def __init__(
        self,
        *,
        target_T_error: float = 4e-11,
        physical_error_rate: float = 5e-4,
        cycles_per_T: int = 50,
    ) -> None:
        self._params = CultivationParams(
            method="in_place_gidney_2024",
            target_logical_T_error=target_T_error,
            physical_error_rate=physical_error_rate,
            cycles_per_T=cycles_per_T,
            physical_qubits_per_T_pool=100,
            notes=(
                "Phase-2α stub. Parameters captured per Gidney 2409.17595; "
                "actual T-state injection currently uses post-selected "
                "Stim T_INJECTION primitive when emitted."
            ),
        )

    @property
    def name(self) -> str:
        return "in_place_cultivation_v0"

    def params(self) -> CultivationParams:
        return self._params


class DistillationFactory(MagicStateFactory):
    """Classical 15-to-1 distillation factory (legacy baseline).

    Documented for completeness; in 2026 cultivation has effectively replaced
    this for new architectures.
    """

    @property
    def name(self) -> str:
        return "15_to_1_distillation"

    def params(self) -> CultivationParams:
        return CultivationParams(
            method="15_to_1_distillation",
            target_logical_T_error=1e-9,
            physical_error_rate=1e-4,
            cycles_per_T=200,
            physical_qubits_per_T_pool=500,
            notes="Legacy. Cultivation is preferred per Gidney 2024.",
        )


# --------------------------------------------------------------------------- #
# Phase 2δ: real cultivation circuit emission                                 #
# --------------------------------------------------------------------------- #


def emit_cultivation_block(
    circuit: "stim.Circuit",
    *,
    ancilla_qubit: int,
    target_T_error: float,
    observable_index: int,
    block_index: int = 0,
) -> dict:
    """Append a cultivation block to ``circuit`` and return its metadata.

    The block models a Gidney–Shutty in-place cultivation outcome (arXiv
    2409.17595) at the residual-error level. We do not lower the full
    growing-stabilizer circuit — that's still upstream research code — but we
    *do* emit a Stim block whose detector-error model carries an explicit
    per-block flip channel at probability ``target_T_error``. A decoder that
    consumes this circuit will see cultivation residuals as ordinary errors
    and the resource estimator will count cultivation observables.

    Emitted Stim ops::

        TICK
        R ancilla_qubit
        X_ERROR(target_T_error) ancilla_qubit
        M ancilla_qubit
        OBSERVABLE_INCLUDE rec[-1]  observable_index
        TICK

    Notes
    -----
    * The pre-measurement state is |0⟩ deterministically; the X_ERROR
      flips it to |1⟩ with probability ``target_T_error``. Stim's strict
      DEM mode therefore admits the observable: the baseline outcome is
      deterministic (0), and the X_ERROR is reported as a single error
      mechanism that flips the observable.

    * No data-qubit interaction is emitted: that's the missing piece
      between this δ block and a full Phase-6 cultivation lowering. The
      LatticeSurgeryProgram's manifest still notes the data qubit each
      block targets so downstream tooling can correlate.
    """
    import stim   # local import: keep cultivation.py importable without stim

    circuit.append("TICK")
    circuit.append("R", [ancilla_qubit])
    if target_T_error > 0:
        circuit.append("X_ERROR", [ancilla_qubit], float(target_T_error))
    circuit.append("M", [ancilla_qubit])
    circuit.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(-1)],
        [float(observable_index)],
    )
    circuit.append("TICK")

    return {
        "block_index": block_index,
        "ancilla_qubit": ancilla_qubit,
        "observable_index": observable_index,
        "target_T_error": float(target_T_error),
        "stim_ops": [
            "TICK", "R", "X_ERROR", "M", "OBSERVABLE_INCLUDE", "TICK",
        ],
        "notes": (
            "Phase-2δ cultivation block. Residual T error modeled as "
            "X_ERROR on dedicated ancilla; observable captures outcome. "
            "Data-qubit injection (CX + correction) is Phase-2β work."
        ),
    }
