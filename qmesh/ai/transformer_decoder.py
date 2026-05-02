"""qmesh.ai.transformer_decoder — small transformer over detector-round tokens.

Inspired by AlphaQubit-2 (arXiv 2512.07737, March 2026): the published model is
~100M+ params and trained on hundreds of millions of shots from real Sycamore
hardware. This is the open-source α-quality skeleton — 2 layers, 4 heads,
d_model=64 (~50K params), trained on Stim-generated synthetic data. It is
faithful to the architecture pattern (sequence-of-rounds → transformer encoder
→ observable head) so once larger weights or real-hardware datasets are
available, the only thing that changes is the training scale.

Pipeline mirrors `qmesh.ai.train_decoder` so the resulting weights drop into
the same `qmesh.ftmode.decoders.NeuralDecoder` slot — it is the *model class*
that's registered into `decoder._model`, not a separate decoder type. The
manifest captures `architecture: "transformer"` so audits can tell which
model produced which logical-error rate.

Tokens:
  - The Stim circuit's detector coordinates have z = round index.
  - We bucket detectors by round → a (R, d_per_round) tensor; rounds with
    fewer detectors are zero-padded and masked.
  - Each round vector projects to d_model, plus a learned positional
    embedding per round; standard nn.TransformerEncoder; mean-pool +
    Linear → observable logits.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from qmesh.ai.train_decoder import TrainingResult, generate_dataset
from qmesh.ftmode.codes import Code
from qmesh.ftmode.decoders.neural import NeuralDecoder


def _round_buckets(stim_circuit) -> tuple[list[list[int]], int, int]:
    """Bucket detector indices by their round (z-coord).

    Returns (buckets, n_rounds, d_per_round). Pads missing rounds with [].
    """
    coords = stim_circuit.get_detector_coordinates()
    rounds: dict[int, list[int]] = {}
    for det_id, c in coords.items():
        # Stim coords are [x, y, round]. Some codes use 2-tuple — fall back.
        r = int(c[2]) if len(c) >= 3 else 0
        rounds.setdefault(r, []).append(det_id)
    if not rounds:
        return [], 0, 0
    n_rounds = max(rounds.keys()) + 1
    buckets = [sorted(rounds.get(r, [])) for r in range(n_rounds)]
    d_per_round = max((len(b) for b in buckets), default=0)
    return buckets, n_rounds, d_per_round


def _events_to_tokens(
    detection_events: np.ndarray, buckets: list[list[int]], d_per_round: int,
) -> np.ndarray:
    """Reshape (N, n_detectors) → (N, n_rounds, d_per_round) zero-padded."""
    n = detection_events.shape[0]
    n_rounds = len(buckets)
    tokens = np.zeros((n, n_rounds, d_per_round), dtype=np.float32)
    for r, dets in enumerate(buckets):
        if not dets:
            continue
        for k, d in enumerate(dets):
            tokens[:, r, k] = detection_events[:, d]
    return tokens


def _build_model(d_per_round: int, n_rounds: int, n_obs: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
    """Build the transformer encoder model."""
    import torch
    import torch.nn as nn

    class _RoundTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Linear(d_per_round, d_model)
            self.pos_embed = nn.Embedding(n_rounds, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=4 * d_model, dropout=0.1,
                activation="gelu", batch_first=True, norm_first=True,
            )
            # enable_nested_tensor=False suppresses a noisy upstream warning
            # when norm_first=True; behaviour is unchanged.
            self.encoder = nn.TransformerEncoder(
                enc_layer, num_layers=n_layers, enable_nested_tensor=False,
            )
            self.head = nn.Linear(d_model, n_obs)
            # remember architectural shape for round-trip token reconstruction
            self.d_per_round = d_per_round
            self.n_rounds = n_rounds
            self.n_obs = n_obs
            self.d_model = d_model

        def forward(self, tokens: "torch.Tensor") -> "torch.Tensor":
            # tokens: (N, n_rounds, d_per_round) of {0.,1.}
            x = self.input_proj(tokens)
            r_idx = torch.arange(tokens.shape[1], device=tokens.device)
            x = x + self.pos_embed(r_idx)
            x = self.encoder(x)
            x = x.mean(dim=1)  # mean-pool over rounds
            return self.head(x)

    return _RoundTransformer()


class _NeuralAdapter:
    """Wraps the transformer so it can replace the MLP inside a NeuralDecoder.

    NeuralDecoder._model.forward receives a flat (N, n_detectors) tensor.
    This adapter reshapes that into round-tokens before forwarding.
    """

    def __init__(self, model, buckets: list[list[int]], d_per_round: int) -> None:
        self._model = model
        self._buckets = buckets
        self._d_per_round = d_per_round
        self._n_rounds = len(buckets)
        # detector-id → (round, slot) index
        self._det_to_slot: dict[int, tuple[int, int]] = {}
        for r, dets in enumerate(buckets):
            for k, d in enumerate(dets):
                self._det_to_slot[d] = (r, k)

    def __call__(self, x):
        return self.forward(x)

    def eval(self):
        self._model.eval()
        return self

    def train(self, mode: bool = True):
        self._model.train(mode)
        return self

    def parameters(self):
        return self._model.parameters()

    def state_dict(self):
        return self._model.state_dict()

    def load_state_dict(self, sd):
        return self._model.load_state_dict(sd)

    def forward(self, flat_events):
        import torch
        # flat_events: (N, n_detectors)
        n = flat_events.shape[0]
        n_det = flat_events.shape[1]
        tokens = torch.zeros(
            (n, self._n_rounds, self._d_per_round),
            dtype=flat_events.dtype, device=flat_events.device,
        )
        for d in range(n_det):
            slot = self._det_to_slot.get(d)
            if slot is None:
                continue
            r, k = slot
            tokens[:, r, k] = flat_events[:, d]
        return self._model(tokens)


def train_transformer_decoder(
    code: Code,
    *,
    physical_error_rate: float = 1e-3,
    n_train: int = 5_000,
    n_val: int = 1_000,
    epochs: int = 4,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    weights_dir: Path | str = "ai_weights",
    seed: int | None = None,
) -> tuple[NeuralDecoder, TrainingResult]:
    """Train a transformer syndrome decoder for the given code.

    Returns (decoder, training_result). The decoder is a NeuralDecoder slot
    with its `_model` swapped for the transformer + token-reshape adapter.
    Saved weights are sha256-signed and replayable.
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
    stim_circuit, X_train_flat, y_train = generate_dataset(
        code, physical_error_rate=physical_error_rate, n_samples=n_train,
    )
    _, X_val_flat, y_val = generate_dataset(
        code, physical_error_rate=physical_error_rate, n_samples=n_val,
    )

    n_detectors = X_train_flat.shape[1]
    n_obs = y_train.shape[1] if y_train.ndim > 1 else 1
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

    buckets, n_rounds, d_per_round = _round_buckets(stim_circuit)
    if d_per_round == 0:
        raise RuntimeError("transformer decoder needs detector coordinates "
                           "(z=round); circuit has none.")

    X_train = _events_to_tokens(X_train_flat, buckets, d_per_round)
    X_val = _events_to_tokens(X_val_flat, buckets, d_per_round)

    model = _build_model(d_per_round, n_rounds, n_obs,
                         d_model=d_model, n_heads=n_heads, n_layers=n_layers)
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
        running, n_batches = 0.0, 0
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
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_pred = (val_logits > 0).float()
            val_acc = (val_pred == y_val_t).all(dim=1).float().mean().item()
        history.append({"epoch": epoch, "loss": avg_loss, "val_accuracy": val_acc})

    # Persist signed weights
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    code_md = code.metadata()
    fname = weights_dir / f"{code_md.name}_p{physical_error_rate:.0e}_xfmr.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "architecture": "transformer",
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "n_detectors": n_detectors,
            "n_observables": n_obs,
            "n_rounds": n_rounds,
            "d_per_round": d_per_round,
            "buckets": buckets,
            "code": code_md.to_dict(),
            "physical_error_rate": physical_error_rate,
        },
    }, fname)
    sha256 = hashlib.sha256(fname.read_bytes()).hexdigest()

    final_acc = history[-1]["val_accuracy"] if history else 0.0
    duration = time.time() - t0

    decoder = load_transformer_decoder(fname)
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


def load_transformer_decoder(weights_path: Path | str) -> NeuralDecoder:
    """Load a transformer-architecture weights file into a NeuralDecoder slot.

    Returns a NeuralDecoder whose `_model` is the transformer wrapped in the
    adapter; its `from_circuit()` is a no-op apart from caching dimensions.
    """
    import torch

    weights_path = Path(weights_path)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if cfg.get("architecture") != "transformer":
        raise ValueError(
            f"weights at {weights_path} are not a transformer "
            f"(architecture={cfg.get('architecture')!r})"
        )

    model = _build_model(
        cfg["d_per_round"], cfg["n_rounds"], cfg["n_observables"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
    )
    model.load_state_dict(state["state_dict"])
    model.eval()

    buckets = cfg["buckets"]
    adapter = _NeuralAdapter(model, buckets, cfg["d_per_round"])

    decoder = NeuralDecoder(weights_path=weights_path, hidden_dim=cfg["d_model"])
    # Pre-fill the slots that NeuralDecoder.from_circuit() would set;
    # also override with the transformer adapter so .decode_batch() routes through.
    decoder._input_dim = cfg["n_detectors"]
    decoder._output_dim = cfg["n_observables"]
    decoder._model = adapter
    decoder._trained = True

    # Replace from_circuit with a sanity-checking no-op
    def _from_circuit(stim_circuit, *, _decoder=decoder, _cfg=cfg):
        dem = stim_circuit.detector_error_model()
        if dem.num_detectors != _cfg["n_detectors"]:
            raise ValueError(
                f"transformer decoder mismatch: weights for "
                f"{_cfg['n_detectors']} detectors, circuit has {dem.num_detectors}"
            )
        return _decoder
    decoder.from_circuit = _from_circuit  # type: ignore[method-assign]

    # Override identity to flag architecture
    base_identity = decoder.identity()
    def _identity(*, _base=dict(base_identity), _cfg=cfg):
        out = dict(_base)
        out["name"] = "neural-transformer-v0"
        out["architecture"] = "transformer"
        out["d_model"] = _cfg["d_model"]
        out["n_layers"] = _cfg["n_layers"]
        out["n_heads"] = _cfg["n_heads"]
        out["trained"] = True
        return out
    decoder.identity = _identity  # type: ignore[method-assign]

    return decoder


__all__ = [
    "train_transformer_decoder",
    "load_transformer_decoder",
]
