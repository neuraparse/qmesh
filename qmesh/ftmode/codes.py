"""qmesh.ftmode.codes — quantum error-correcting code primitives.

Phase 2α: real implementation of the surface code via Stim. BBCode and
FloquetCode are placeholders documenting their interface and Phase-2β plan
(BB needs custom Stim circuit construction; hyperbolic Floquet research
code lives in academic repos as of May 2026).

Each code object exposes:
- `generate_memory_circuit(rounds, basis, noise) -> stim.Circuit`
- `generate_logical_t_circuit(...) -> stim.Circuit`  (where supported)
- `physical_qubit_count() -> int`
- `name() -> str`
- `metadata() -> dict`            # for the manifest
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True, frozen=True)
class CodeMetadata:
    """Metadata recorded in every FT manifest."""
    name: str
    distance: int
    rounds: int
    physical_qubits: int
    logical_qubits: int
    extra: dict[str, object]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "distance": self.distance,
            "rounds": self.rounds,
            "physical_qubits": self.physical_qubits,
            "logical_qubits": self.logical_qubits,
            **self.extra,
        }


class Code(ABC):
    """Abstract code primitive."""

    @abstractmethod
    def metadata(self) -> CodeMetadata: ...

    @abstractmethod
    def generate_memory_circuit(
        self,
        *,
        basis: Literal["Z", "X"] = "Z",
        physical_error_rate: float = 1e-3,
    ):
        """Return a stim.Circuit performing a memory experiment."""

    def physical_qubit_count(self) -> int:
        return self.metadata().physical_qubits

    @property
    def name(self) -> str:
        return self.metadata().name


class SurfaceCode(Code):
    """Rotated surface code via Stim.

    Encodes a single logical qubit; supports memory experiments in Z or X
    basis, and inherits Stim's full noise model surface (depolarising on
    Cliffords, leakage flip on resets, measurement flip).

    Stim's `Circuit.generated('surface_code:rotated_memory_z', ...)` produces
    the standard d×d patch with d² data qubits + (d² - 1) ancilla qubits.
    """

    def __init__(self, *, distance: int, rounds: int) -> None:
        if distance < 3 or distance % 2 == 0:
            raise ValueError(f"distance must be odd ≥ 3, got {distance}")
        if rounds < 1:
            raise ValueError(f"rounds must be ≥ 1, got {rounds}")
        self.distance = distance
        self.rounds = rounds

    def metadata(self) -> CodeMetadata:
        d = self.distance
        # rotated surface: d² data + (d² - 1) syndrome ancillas
        physical = d * d + (d * d - 1)
        return CodeMetadata(
            name=f"rotated_surface_d{d}",
            distance=d,
            rounds=self.rounds,
            physical_qubits=physical,
            logical_qubits=1,
            extra={"layout": "rotated", "code_family": "surface"},
        )

    def generate_memory_circuit(
        self,
        *,
        basis: Literal["Z", "X"] = "Z",
        physical_error_rate: float = 1e-3,
    ):
        import stim

        kind = "rotated_memory_z" if basis == "Z" else "rotated_memory_x"
        return stim.Circuit.generated(
            f"surface_code:{kind}",
            rounds=self.rounds,
            distance=self.distance,
            after_clifford_depolarization=physical_error_rate,
            before_round_data_depolarization=physical_error_rate,
            before_measure_flip_probability=physical_error_rate,
            after_reset_flip_probability=physical_error_rate,
        )


class UnrotatedSurfaceCode(Code):
    """Standard (unrotated) surface code via Stim. Larger than rotated for
    the same distance but easier to reason about for didactic purposes."""

    def __init__(self, *, distance: int, rounds: int) -> None:
        self.distance = distance
        self.rounds = rounds

    def metadata(self) -> CodeMetadata:
        d = self.distance
        # unrotated: 2d² - 2d + 1 data + 2d² - 2d ancillas (approx)
        physical = 2 * d * d - 2 * d + 1 + (2 * d * d - 2 * d)
        return CodeMetadata(
            name=f"unrotated_surface_d{d}",
            distance=d, rounds=self.rounds,
            physical_qubits=physical, logical_qubits=1,
            extra={"layout": "unrotated", "code_family": "surface"},
        )

    def generate_memory_circuit(
        self,
        *,
        basis: Literal["Z", "X"] = "Z",
        physical_error_rate: float = 1e-3,
    ):
        import stim

        kind = "unrotated_memory_z" if basis == "Z" else "unrotated_memory_x"
        return stim.Circuit.generated(
            f"surface_code:{kind}",
            rounds=self.rounds,
            distance=self.distance,
            after_clifford_depolarization=physical_error_rate,
            before_round_data_depolarization=physical_error_rate,
            before_measure_flip_probability=physical_error_rate,
            after_reset_flip_probability=physical_error_rate,
        )


class RepetitionCode(Code):
    """Classical-error-correcting repetition code via Stim. Useful as a sanity
    baseline — corrects bit flips only, but trivially decodable."""

    def __init__(self, *, distance: int, rounds: int) -> None:
        self.distance = distance
        self.rounds = rounds

    def metadata(self) -> CodeMetadata:
        d = self.distance
        return CodeMetadata(
            name=f"repetition_d{d}",
            distance=d, rounds=self.rounds,
            physical_qubits=2 * d - 1, logical_qubits=1,
            extra={"code_family": "repetition", "corrects": "bit_flip"},
        )

    def generate_memory_circuit(
        self, *, basis: Literal["Z", "X"] = "Z", physical_error_rate: float = 1e-3,
    ):
        import stim

        kind = "memory" if basis == "Z" else "memory_x"
        return stim.Circuit.generated(
            f"repetition_code:{kind}",
            rounds=self.rounds, distance=self.distance,
            after_clifford_depolarization=physical_error_rate,
            before_round_data_depolarization=physical_error_rate,
            before_measure_flip_probability=physical_error_rate,
            after_reset_flip_probability=physical_error_rate,
        )


class BBCode(Code):
    """Bivariate-bicycle qLDPC code — IBM gross code [[144, 12, 12]].

    Phase 2β α-slice: emits a real stim.Circuit for the [[144, 12, 12]]
    "gross" code per Bravyi et al. 2024 (Nature 627, 778, also arXiv
    2308.07915). Construction parameters:

      l = 12, m = 6   (so n = 2 · l · m = 144)
      A = x³ + y + y²    (sparse polynomial in Z_l × Z_m)
      B = y³ + x + x²

    The parity-check matrix is::

        H_X = [ A   | B   ]    (66 X-checks)
        H_Z = [ B^T | A^T ]    (66 Z-checks)

    each row weight 6, each column weight 6. Distance d = 12 with
    ``distance = 12``; smaller (l, m) configurations give other code points
    (e.g. ``distance = 6`` ⇒ ``[[72, 12, 6]]`` — set ``distance=6`` to use it).

    α vs full β:
      * the syndrome circuit uses a generic 6-step CNOT schedule per round
        (one CNOT per stabilizer-data pair per substep); β work is to fold
        in Bravyi's depth-7 syndrome extraction with proper bridging,
      * we apply per-round depolarising noise on Cliffords + measurement-
        flip noise (matching Stim's surface_code conventions),
      * we use full per-detector references against the previous round so
        PyMatching can form a meaningful matching graph,
      * logical observable is one column of the logical-X operator matrix
        (a known weight-l string of data qubits); higher-fidelity logical
        operators per Bravyi 2024 §V are β work.

    Reference: IBM 2024 Nature paper.
    """

    def __init__(self, *, n: int | None = None, k: int = 12,
                 distance: int = 12, rounds: int = 8) -> None:
        # Standard configurations from Bravyi et al. 2024.
        # We key on `distance` to choose (l, m, A, B); n + k follow.
        if distance == 12:
            l, m = 12, 6
            self._A_terms = [("x", 3), ("y", 1), ("y", 2)]
            self._B_terms = [("y", 3), ("x", 1), ("x", 2)]
        elif distance == 6:
            l, m = 6, 6
            self._A_terms = [("x", 3), ("y", 1), ("y", 2)]
            self._B_terms = [("y", 3), ("x", 1), ("x", 2)]
        elif distance == 10:
            l, m = 9, 6
            self._A_terms = [("x", 3), ("y", 1), ("y", 2)]
            self._B_terms = [("y", 3), ("x", 1), ("x", 2)]
        else:
            # Fallback: l=12, m=6 (the gross code) — annotate distance.
            l, m = 12, 6
            self._A_terms = [("x", 3), ("y", 1), ("y", 2)]
            self._B_terms = [("y", 3), ("x", 1), ("x", 2)]
        self.l = l
        self.m = m
        self.k = k
        self.distance = distance
        self.rounds = rounds
        self.n = n if n is not None else 2 * l * m

    def metadata(self) -> CodeMetadata:
        return CodeMetadata(
            name=f"bb[[{self.n},{self.k},{self.distance}]]",
            distance=self.distance, rounds=self.rounds,
            physical_qubits=self.n, logical_qubits=self.k,
            extra={
                "code_family": "bivariate_bicycle",
                "l": self.l, "m": self.m,
                "A_terms": [list(t) for t in self._A_terms],
                "B_terms": [list(t) for t in self._B_terms],
                "implementation": "α-slice",
                "reference": "Bravyi et al. 2024 (Nature)",
            },
        )

    # ---- parity-check construction -------------------------------------

    def _logical_x_representatives(self, H_check, H_other) -> list[list[int]]:
        """Compute logical operator representatives over GF(2).

        For a CSS code with parity-check matrices H_X, H_Z:
          * a logical-Z operator is a vector in ker(H_X) \\ rowspan(H_Z),
          * a logical-X operator is a vector in ker(H_Z) \\ rowspan(H_X).

        Pass ``H_check`` = the check matrix L must commute with (i.e., the
        opposite-type checks), and ``H_other`` = the matrix whose rowspan
        we mod out (the same-type checks). This returns vectors that
        commute with H_check and are not in the rowspan of H_other.
        """
        import numpy as np
        n = self.n
        Hcheck = np.zeros((len(H_check), n), dtype=np.uint8)
        for i, row in enumerate(H_check):
            for q in row:
                Hcheck[i, q] ^= 1
        Hother = np.zeros((len(H_other), n), dtype=np.uint8)
        for i, row in enumerate(H_other):
            for q in row:
                Hother[i, q] ^= 1

        def row_reduce(M: np.ndarray) -> tuple[np.ndarray, list[int]]:
            M = M.copy()
            r = 0
            pivots: list[int] = []
            for c in range(M.shape[1]):
                # find a row with a 1 in column c, at row >= r
                rows = np.nonzero(M[r:, c])[0]
                if rows.size == 0:
                    continue
                p = r + rows[0]
                M[[r, p]] = M[[p, r]]
                # eliminate other rows
                for q in range(M.shape[0]):
                    if q != r and M[q, c]:
                        M[q] ^= M[r]
                pivots.append(c)
                r += 1
                if r == M.shape[0]:
                    break
            return M, pivots

        # Kernel of Hcheck: solve Hcheck v = 0
        Hc_rref, piv = row_reduce(Hcheck)
        free_cols = [c for c in range(n) if c not in piv]
        ker = np.zeros((len(free_cols), n), dtype=np.uint8)
        for fi, fc in enumerate(free_cols):
            ker[fi, fc] = 1
            for r_idx, p in enumerate(piv):
                if Hc_rref[r_idx, fc]:
                    ker[fi, p] = 1
        # Now reduce ker mod rowspace(Hother). Stack Hother + ker, row-reduce,
        # find ker rows with leading 1 not in rowspace of Hother.
        stacked = np.vstack([Hother, ker])
        rr, pivs = row_reduce(stacked)
        n_other = Hother.shape[0]
        logicals: list[list[int]] = []
        for r_idx in range(n_other, rr.shape[0]):
            row = rr[r_idx]
            if row.any():
                logicals.append([i for i in range(n) if row[i]])
                if len(logicals) >= self.k:
                    break
        return logicals

    def _build_check_matrices(self):
        """Return (H_X, H_Z) as lists-of-lists of data-qubit indices.

        Data qubits are labelled 0..n-1 with the convention:
            "left half"  data qubit (i, j) → index = i * m + j     in [0, l*m)
            "right half" data qubit (i, j) → index = l*m + i*m + j  in [l*m, 2*l*m)

        H_X has l*m rows ([0..l*m)); each X-check connects to:
          - data qubits (i', j') in left half such that A[(i,j), (i',j')] = 1
          - data qubits (i', j') in right half such that B[(i,j), (i',j')] = 1

        H_Z is built symmetrically with B^T and A^T.
        """
        l, m = self.l, self.m
        lm = l * m

        def shift_xy(i: int, j: int, term):
            kind, p = term
            if kind == "x":
                return (i + p) % l, j
            else:
                return i, (j + p) % m

        H_X: list[list[int]] = []
        for i in range(l):
            for j in range(m):
                row = []
                # left half: A
                for term in self._A_terms:
                    ip, jp = shift_xy(i, j, term)
                    row.append(ip * m + jp)
                # right half: B
                for term in self._B_terms:
                    ip, jp = shift_xy(i, j, term)
                    row.append(lm + ip * m + jp)
                H_X.append(row)

        # H_Z = [B^T | A^T] — A^T shifts by (-px, -py)
        H_Z: list[list[int]] = []
        for i in range(l):
            for j in range(m):
                row = []
                # left half: B^T → for term in B, neighbour is shifted by -term
                for term in self._B_terms:
                    kind, p = term
                    if kind == "x":
                        ip, jp = (i - p) % l, j
                    else:
                        ip, jp = i, (j - p) % m
                    row.append(ip * m + jp)
                # right half: A^T
                for term in self._A_terms:
                    kind, p = term
                    if kind == "x":
                        ip, jp = (i - p) % l, j
                    else:
                        ip, jp = i, (j - p) % m
                    row.append(lm + ip * m + jp)
                H_Z.append(row)

        return H_X, H_Z

    # ---- circuit emission ----------------------------------------------

    def generate_memory_circuit(
        self,
        *,
        basis: Literal["Z", "X"] = "Z",
        physical_error_rate: float = 1e-3,
    ):
        """Emit a Stim circuit for `rounds` rounds of BB-code memory.

        Z-basis memory: prepare all data qubits in |0⟩, repeatedly extract
        the X-syndrome (so X errors appear as syndrome flips), and finish by
        measuring all data qubits in Z.

        X-basis memory is symmetric.
        """
        import stim

        H_X, H_Z = self._build_check_matrices()
        nx = len(H_X)
        nz = len(H_Z)
        n = self.n

        # Qubit IDs:
        #   0 .. n-1               data qubits
        #   n .. n+nx-1            X-check ancillas
        #   n+nx .. n+nx+nz-1      Z-check ancillas
        x_anc = [n + i for i in range(nx)]
        z_anc = [n + nx + i for i in range(nz)]

        circuit = stim.Circuit()
        # QUBIT_COORDS for visualisation; embeds (i, j) → (j, i) in plane.
        for d in range(n):
            half = "L" if d < self.l * self.m else "R"
            local = d if half == "L" else d - self.l * self.m
            i = local // self.m
            j = local % self.m
            circuit.append("QUBIT_COORDS", [d],
                           [float(j) + (0 if half == "L" else self.m + 1.0),
                            float(i)])
        for i, q in enumerate(x_anc):
            circuit.append("QUBIT_COORDS", [q],
                           [float(i % self.m) + 0.3,
                            float(i // self.m) + 0.3])
        for i, q in enumerate(z_anc):
            circuit.append("QUBIT_COORDS", [q],
                           [float(i % self.m) + 0.6,
                            float(i // self.m) + 0.6])

        p = float(physical_error_rate)
        all_data = list(range(n))

        # Initial preparation in the chosen basis.
        if basis == "Z":
            circuit.append("R", all_data)
            if p > 0:
                circuit.append("X_ERROR", all_data, p)
        else:
            circuit.append("R", all_data)
            if p > 0:
                circuit.append("X_ERROR", all_data, p)
            circuit.append("H", all_data)
            if p > 0:
                circuit.append("DEPOLARIZE1", all_data, p)
        circuit.append("R", x_anc + z_anc)
        if p > 0:
            circuit.append("X_ERROR", x_anc + z_anc, p)
        circuit.append("TICK")

        def emit_syndrome_round(initial: bool) -> None:
            # X-checks: H ancilla; CNOTs ancilla→data; H ancilla; measure-reset
            circuit.append("H", x_anc)
            if p > 0:
                circuit.append("DEPOLARIZE1", x_anc, p)
            circuit.append("TICK")
            # 6 sub-rounds of CNOT (one per non-zero entry in H_X row)
            for sub in range(6):
                pairs = []
                for ai, row in enumerate(H_X):
                    pairs.append(x_anc[ai])
                    pairs.append(row[sub])
                circuit.append("CX", pairs)
                if p > 0:
                    circuit.append("DEPOLARIZE2", pairs, p)
                circuit.append("TICK")
            circuit.append("H", x_anc)
            if p > 0:
                circuit.append("DEPOLARIZE1", x_anc, p)
            circuit.append("TICK")
            # Z-checks: CNOTs data→ancilla
            for sub in range(6):
                pairs = []
                for ai, row in enumerate(H_Z):
                    pairs.append(row[sub])
                    pairs.append(z_anc[ai])
                circuit.append("CX", pairs)
                if p > 0:
                    circuit.append("DEPOLARIZE2", pairs, p)
                circuit.append("TICK")
            # Measure ancillas with reset for next round.
            if p > 0:
                circuit.append("X_ERROR", x_anc + z_anc, p)
            circuit.append("MR", x_anc + z_anc)

            n_anc = nx + nz
            # Detectors: in Z-basis memory the X-syndromes are the
            # informative ones (first nx records of this round).
            if basis == "Z":
                meaningful = list(range(nx))
                # Z-syndromes are deterministic in Z basis (data prepared in
                # |0⟩, Z-stabilizers are +1 eigenstates).
                if initial:
                    for i in meaningful:
                        # initial round: X-syndrome on a Z-prep state is
                        # random; skip to first round Z-anchor instead.
                        # We anchor X-syndromes against |0⟩-prep where
                        # Z-stabilizers are +1; only emit Z-detectors here.
                        pass
                    for i in range(nz):
                        rec = -(n_anc - nx - i)
                        circuit.append("DETECTOR", [stim.target_rec(rec)])
                else:
                    for i in range(n_anc):
                        # difference detector vs previous round
                        rec_now = -(n_anc - i)
                        rec_prev = -2 * n_anc + i
                        circuit.append("DETECTOR", [
                            stim.target_rec(rec_now),
                            stim.target_rec(rec_prev),
                        ])
            else:  # X basis
                if initial:
                    for i in range(nx):
                        rec = -(n_anc - i)
                        circuit.append("DETECTOR", [stim.target_rec(rec)])
                else:
                    for i in range(n_anc):
                        rec_now = -(n_anc - i)
                        rec_prev = -2 * n_anc + i
                        circuit.append("DETECTOR", [
                            stim.target_rec(rec_now),
                            stim.target_rec(rec_prev),
                        ])
            circuit.append("TICK")

        emit_syndrome_round(initial=True)
        for _ in range(self.rounds - 1):
            emit_syndrome_round(initial=False)

        # Final data measurement.
        if basis == "X":
            circuit.append("H", all_data)
            if p > 0:
                circuit.append("DEPOLARIZE1", all_data, p)
        if p > 0:
            circuit.append("X_ERROR", all_data, p)
        circuit.append("M", all_data)

        # Final detectors: comparing the final data measurement against the
        # last round's syndromes that should be reproducible.
        # For a Z-basis memory, the final Z-data measurement reconstructs
        # the Z-stabilizers; for each Z-check, the parity of its 6 data
        # qubits (modulo the previous Z-syndrome) should be 0 in the
        # absence of errors.
        n_anc = nx + nz
        # Compute a real logical operator representative for the
        # OBSERVABLE_INCLUDE.
        # Z-basis memory: we need a logical Z (commutes with H_X, not in
        # rowspan of H_Z). X-basis memory: a logical X (commutes with H_Z,
        # not in rowspan of H_X).
        if basis == "Z":
            logicals = self._logical_x_representatives(H_X, H_Z)
        else:
            logicals = self._logical_x_representatives(H_Z, H_X)
        # Pick the lowest-weight logical for stability.
        if logicals:
            logicals.sort(key=len)
            obs_qubits = logicals[0]
        else:
            obs_qubits = [0]   # degenerate fallback; manifest will note α

        if basis == "Z":
            for ai, row in enumerate(H_Z):
                refs = [stim.target_rec(-(n - q)) for q in row]
                # previous Z-ancilla measurement record: this ai sits at
                # offset (nz - ai) before the data measurements.
                prev_ref = stim.target_rec(-n - (nz - ai))
                circuit.append("DETECTOR", refs + [prev_ref])
            obs_refs = [stim.target_rec(-(n - q)) for q in obs_qubits]
            circuit.append("OBSERVABLE_INCLUDE", obs_refs, 0)
        else:
            for ai, row in enumerate(H_X):
                refs = [stim.target_rec(-(n - q)) for q in row]
                prev_ref = stim.target_rec(-n - n_anc + ai)
                circuit.append("DETECTOR", refs + [prev_ref])
            obs_refs = [stim.target_rec(-(n - q)) for q in obs_qubits]
            circuit.append("OBSERVABLE_INCLUDE", obs_refs, 0)

        return circuit


__all__ = [
    "Code", "CodeMetadata",
    "SurfaceCode", "UnrotatedSurfaceCode", "RepetitionCode",
    "BBCode",
]
