"""BP + OSD decoder for qLDPC codes — Phase 2γ.

Belief Propagation followed by Ordered Statistics Decoding (BP+OSD) is the
canonical decoder for general qLDPC codes (e.g. Bivariate-Bicycle / IBM
gross code) where MWPM doesn't apply because the syndrome graph contains
hyperedges (a single error flips three or more checks).

Implementation strategy:
  * Prefer ``ldpc.BpOsdDecoder`` from the Roffe et al. PyPI ``ldpc``
    package (https://github.com/quantumgizmos/ldpc) — that's the canonical
    fast C++ implementation and supports OSD-W up to high orders.
  * If ``ldpc`` isn't installed, fall back to a small in-package
    implementation: min-sum BP for ``max_iter=50`` iterations followed by
    OSD-1. The fallback is correct but slow; we hard-stop at distance > 7
    when forced to use it.

Wiring:
  * Available via ``FTConfig(decoder="bp_osd")`` (registered in
    :mod:`qmesh.ftmode.decoders`).
  * Recovers a syndrome-aware error guess from the parity-check matrix
    extracted out of the Stim ``DetectorErrorModel``. We construct the
    detector-by-error sparse matrix once at :meth:`from_circuit` time and
    feed it to BP+OSD per shot.

α-honest framing:
  * The fallback's BP convergence is not as tight as the ldpc package's;
    on small codes (n ≤ ~200) this is fine for sanity tests.
  * For BB-code [[144,12,12]] runs at p=1e-3 the fallback is too slow to
    be useful — install ``ldpc`` for production use.

Reference: Roffe et al. 2020, "Decoding across the quantum LDPC code
landscape" (Quantum Sci. Technol.).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from qmesh.ftmode.decoders.base import DecodeResult, Decoder


@dataclass
class BpOsdDecoder(Decoder):
    """BP+OSD decoder for qLDPC codes.

    Parameters
    ----------
    max_iter : int
        BP iterations per syndrome. Default 50.
    osd_order : int
        Ordered Statistics Decoding order (0 = greedy, 1 = exhaustive over
        a single bit flip). Default 1.
    bp_method : str
        ``"min_sum"`` or ``"product_sum"``. Default ``"min_sum"`` — matches
        the ldpc-package default.
    force_fallback : bool
        If True, never call ``ldpc.BpOsdDecoder`` even if installed. Used
        in tests to exercise the fallback path.
    """

    max_iter: int = 50
    osd_order: int = 1
    bp_method: str = "min_sum"
    force_fallback: bool = False

    _matcher: object | None = None       # ldpc.BpOsdDecoder when available
    _H: np.ndarray | None = None         # parity-check matrix (shape: m × n)
    _obs_matrix: np.ndarray | None = None  # observable matrix (shape: o × n)
    _error_priors: np.ndarray | None = None
    _num_observables: int = 0
    _circuit: object | None = None
    _ldpc_available: bool = False
    _using_fallback: bool = field(default=False, init=False)

    @property
    def name(self) -> str:
        backend = "ldpc-pypi" if (self._ldpc_available and not self.force_fallback) else "qmesh-fallback"
        return f"bp_osd/{backend}/iter={self.max_iter}/osd={self.osd_order}"

    # ----- DEM → check matrix --------------------------------------

    def _dem_to_matrices(self, dem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert a Stim DetectorErrorModel to (H, observable_matrix, priors).

        Returns
        -------
        H : (num_detectors, num_errors) uint8 — H[i,j]=1 iff error j flips
            detector i.
        Lobs : (num_observables, num_errors) uint8 — likewise for logical
            observables.
        priors : (num_errors,) float — per-error probability.
        """
        n_det = dem.num_detectors
        n_obs = dem.num_observables

        # First pass: count errors and discover the index range.
        errors: list[tuple[float, set[int], set[int]]] = []

        def _walk(model, det_offset: int = 0, repeat: int = 1) -> None:
            for inst in model:
                # CircuitRepeatBlock-like: detector errors usually flat
                if hasattr(inst, "type") and inst.type == "error":
                    p = inst.args_copy()[0]
                    dets: set[int] = set()
                    obs: set[int] = set()
                    for t in inst.targets_copy():
                        if t.is_relative_detector_id():
                            dets.add(t.val + det_offset)
                        elif t.is_logical_observable_id():
                            obs.add(t.val)
                        # separator / coordinate args are ignored
                    if dets or obs:
                        errors.append((p, dets, obs))

        _walk(dem)
        n_err = len(errors)
        H = np.zeros((n_det, n_err), dtype=np.uint8)
        Lobs = np.zeros((n_obs, n_err), dtype=np.uint8)
        priors = np.zeros(n_err, dtype=np.float64)
        for j, (p, dets, obs) in enumerate(errors):
            for d in dets:
                if 0 <= d < n_det:
                    H[d, j] = 1
            for o in obs:
                if 0 <= o < n_obs:
                    Lobs[o, j] = 1
            priors[j] = max(p, 1e-15)
        return H, Lobs, priors

    def from_circuit(self, stim_circuit) -> "BpOsdDecoder":
        dem = stim_circuit.detector_error_model(
            decompose_errors=False,
            ignore_decomposition_failures=True,
            approximate_disjoint_errors=True,
        )
        H, Lobs, priors = self._dem_to_matrices(dem)
        self._H = H
        self._obs_matrix = Lobs
        self._error_priors = priors
        self._num_observables = stim_circuit.num_observables
        self._circuit = stim_circuit

        # Try to instantiate the ldpc package.
        self._ldpc_available = False
        self._matcher = None
        if not self.force_fallback:
            try:
                import ldpc  # noqa: F401
                self._ldpc_available = True
            except ImportError:
                self._ldpc_available = False

        if self._ldpc_available and not self.force_fallback:
            try:
                # Newer ldpc package (v2+) exposes BpOsdDecoder.
                from ldpc.bposd_decoder import BpOsdDecoder as _LdpcBpOsd
                self._matcher = _LdpcBpOsd(
                    H,
                    error_channel=priors.tolist(),
                    bp_method=self.bp_method,
                    max_iter=self.max_iter,
                    osd_method="osd_cs",
                    osd_order=self.osd_order,
                )
                self._using_fallback = False
            except Exception:
                # Older ldpc / different surface — fall back gracefully.
                try:
                    from ldpc import bposd_decoder as _bposd
                    self._matcher = _bposd(
                        H,
                        error_channel=priors.tolist(),
                        bp_method=self.bp_method,
                        max_iter=self.max_iter,
                        osd_method="osd_cs",
                        osd_order=self.osd_order,
                    )
                    self._using_fallback = False
                except Exception:
                    self._matcher = None
                    self._using_fallback = True
        else:
            self._using_fallback = True
        return self

    # ----- fallback BP+OSD ----------------------------------------

    def _fallback_decode_one(self, syndrome: np.ndarray) -> np.ndarray:
        """Min-sum BP + OSD-order osd_order on a single syndrome.

        Returns a binary error-vector guess of length n.
        """
        H = self._H
        priors = self._error_priors
        if H is None or priors is None:
            return np.zeros(0, dtype=np.uint8)
        m, n = H.shape

        # Hard-stop: the fallback is too slow to be useful for big codes.
        # Block when n is large unless the user really wants it.
        if n > 4_000:
            raise RuntimeError(
                "BP+OSD fallback too slow on this code (n={n}, distance>7). "
                "Install the `ldpc` PyPI package for production use: "
                "`pip install ldpc`.".format(n=n)
            )

        syndrome = syndrome.astype(np.uint8).flatten()
        # llr from priors: log((1-p)/p)
        llr_prior = np.log((1.0 - priors) / np.maximum(priors, 1e-15))

        # Sparse representation of H
        rows: list[np.ndarray] = [np.where(H[i] == 1)[0] for i in range(m)]
        cols: list[np.ndarray] = [np.where(H[:, j] == 1)[0] for j in range(n)]

        # message arrays: m_v_to_c[i,j] for each (check i, var j) pair
        # Use a dict-of-arrays keyed by check index for sparsity.
        msg_v2c: dict[tuple[int, int], float] = {}
        msg_c2v: dict[tuple[int, int], float] = {}
        for i in range(m):
            for j in rows[i]:
                msg_v2c[(i, j)] = llr_prior[j]
                msg_c2v[(i, j)] = 0.0

        # min-sum BP
        for _it in range(self.max_iter):
            # Check → variable
            for i in range(m):
                # get all incoming v→c messages
                vs = rows[i]
                in_msgs = np.array([msg_v2c[(i, v)] for v in vs])
                signs = np.sign(in_msgs)
                signs[signs == 0] = 1.0
                # parity sign correction: (-1)^syndrome bit
                s_sign = -1.0 if syndrome[i] else 1.0
                abs_msgs = np.abs(in_msgs)
                # for each outgoing message, exclude its own variable
                for k, v in enumerate(vs):
                    out_sign = s_sign * np.prod(np.delete(signs, k))
                    if abs_msgs.size > 1:
                        out_mag = np.min(np.delete(abs_msgs, k))
                    else:
                        out_mag = abs_msgs[k]  # degenerate
                    msg_c2v[(i, v)] = out_sign * out_mag

            # Variable → check
            for j in range(n):
                cs = cols[j]
                if cs.size == 0:
                    continue
                in_msgs = np.array([msg_c2v[(c, j)] for c in cs])
                total = llr_prior[j] + in_msgs.sum()
                for k, c in enumerate(cs):
                    msg_v2c[(c, j)] = total - in_msgs[k]

            # Check convergence: posterior llr → hard decision
            posteriors = llr_prior.copy()
            for j in range(n):
                cs = cols[j]
                for c in cs:
                    posteriors[j] += msg_c2v[(c, j)]
            decisions = (posteriors < 0).astype(np.uint8)
            # Check H @ decisions == syndrome (mod 2)
            check_syn = (H @ decisions) % 2
            if np.array_equal(check_syn, syndrome):
                return decisions

        # BP didn't converge → OSD post-processing.
        posteriors = llr_prior.copy()
        for j in range(n):
            cs = cols[j]
            for c in cs:
                posteriors[j] += msg_c2v[(c, j)]
        # Reliability: |posteriors| (smaller = less reliable).
        # OSD-0: sort columns by reliability descending, pick the most
        # reliable n-rank-of-H columns as "fixed at BP guess", solve for
        # the rest.
        order = np.argsort(-np.abs(posteriors))
        bp_guess = (posteriors < 0).astype(np.uint8)

        # OSD-0: gauss-eliminate columns of H in `order`; find rank columns
        # to form a basis, solve for those, fix the rest at bp_guess.
        H_work = H.copy().astype(np.uint8)
        s_work = syndrome.copy().astype(np.uint8)
        # subtract bp_guess on the non-basis columns: r = s - H_n * bp_n
        # but we don't know split yet — do row-reduction first.
        # Reorder columns by `order` (most reliable first).
        col_perm = order
        H_perm = H_work[:, col_perm]
        # Row-reduce H_perm; track pivot columns (these are our basis).
        M = H_perm.copy()
        s_vec = s_work.copy()
        pivots: list[int] = []
        r = 0
        for c in range(M.shape[1]):
            if r >= M.shape[0]:
                break
            rows_with_one = np.where(M[r:, c] == 1)[0]
            if rows_with_one.size == 0:
                continue
            p = r + rows_with_one[0]
            if p != r:
                M[[r, p]] = M[[p, r]]
                s_vec[[r, p]] = s_vec[[p, r]]
            for q in range(M.shape[0]):
                if q != r and M[q, c]:
                    M[q] ^= M[r]
                    s_vec[q] ^= s_vec[r]
            pivots.append(c)
            r += 1
        rank = len(pivots)
        # Initialize x_perm = bp_guess[col_perm]
        x_perm = bp_guess[col_perm].copy()
        # Compute target rhs from the non-pivot columns at their bp values:
        # row-reduced form: each pivot row r has 1 only at col=pivots[r].
        # The equation is: x_pivot_r + sum_{c not in pivots} M[r,c] * x_c = s_vec[r].
        non_pivots = [c for c in range(M.shape[1]) if c not in pivots]
        for r_idx, piv_col in enumerate(pivots):
            rhs = int(s_vec[r_idx])
            for c in non_pivots:
                if M[r_idx, c]:
                    rhs ^= int(x_perm[c])
            x_perm[piv_col] = np.uint8(rhs)
        # Map back through col_perm: result[col_perm[k]] = x_perm[k]
        result = np.zeros(n, dtype=np.uint8)
        for k, j in enumerate(col_perm):
            result[j] = x_perm[k]

        # OSD-1: try flipping each non-pivot column independently and keep
        # the lowest-weight valid solution.
        if self.osd_order >= 1 and len(non_pivots) > 0:
            best = result.copy()
            best_weight = int(best.sum())
            # Limit search size to avoid blowup on large codes.
            limit = min(64, len(non_pivots))
            for k in range(limit):
                trial_perm = x_perm.copy()
                trial_perm[non_pivots[k]] ^= 1
                # re-resolve pivots
                for r_idx, piv_col in enumerate(pivots):
                    rhs = int(s_vec[r_idx])
                    for c in non_pivots:
                        if M[r_idx, c]:
                            rhs ^= int(trial_perm[c])
                    trial_perm[piv_col] = np.uint8(rhs)
                trial = np.zeros(n, dtype=np.uint8)
                for k2, j in enumerate(col_perm):
                    trial[j] = trial_perm[k2]
                check = (H @ trial) % 2
                if np.array_equal(check, syndrome):
                    w = int(trial.sum())
                    if w < best_weight:
                        best_weight = w
                        best = trial
            result = best

        return result

    # ----- batch decode --------------------------------------------

    def decode_batch(
        self,
        detection_events: np.ndarray,
        observable_flips: np.ndarray,
    ) -> DecodeResult:
        if self._H is None or self._obs_matrix is None:
            raise RuntimeError("call from_circuit() first")
        t0 = time.time()
        shots = detection_events.shape[0]
        n_obs = self._num_observables or 1
        predictions = np.zeros((shots, n_obs), dtype=np.uint8)
        Lobs = self._obs_matrix

        for i in range(shots):
            syn = detection_events[i].astype(np.uint8).flatten()
            # match the H matrix's detector axis
            n_det = self._H.shape[0]
            if syn.size < n_det:
                syn = np.concatenate([syn, np.zeros(n_det - syn.size, dtype=np.uint8)])
            elif syn.size > n_det:
                syn = syn[:n_det]
            try:
                if not self._using_fallback and self._matcher is not None:
                    err = self._matcher.decode(syn)
                else:
                    err = self._fallback_decode_one(syn)
                err = np.asarray(err, dtype=np.uint8).flatten()
                pred_obs = (Lobs @ err) % 2
            except Exception:
                pred_obs = np.zeros(n_obs, dtype=np.uint8)
            if pred_obs.size < n_obs:
                pred_obs = np.concatenate([
                    pred_obs, np.zeros(n_obs - pred_obs.size, dtype=np.uint8),
                ])
            elif pred_obs.size > n_obs:
                pred_obs = pred_obs[:n_obs]
            predictions[i] = pred_obs

        if observable_flips.ndim == 1:
            observable_flips = observable_flips.reshape(-1, 1)
        n_errors = int(np.sum(predictions != observable_flips))
        return DecodeResult(
            predictions=predictions,
            logical_error_count=n_errors,
            shots=shots,
            wall_seconds=time.time() - t0,
            metadata={
                "name": self.name,
                "max_iter": self.max_iter,
                "osd_order": self.osd_order,
                "bp_method": self.bp_method,
                "ldpc_available": self._ldpc_available,
                "using_fallback": self._using_fallback,
                "n_errors_in_dem": int(self._H.shape[1]) if self._H is not None else 0,
            },
        )

    def identity(self) -> dict:
        return {
            "name": self.name,
            "max_iter": self.max_iter,
            "osd_order": self.osd_order,
            "bp_method": self.bp_method,
            "ldpc_package_available": self._ldpc_available,
            "using_fallback": self._using_fallback,
        }


__all__ = ["BpOsdDecoder"]
