"""qmesh.backends.registry — backend lookup."""

from __future__ import annotations

from qmesh.backends.base import Backend

_REGISTRY: dict[str, Backend] = {}


def register(backend: Backend) -> None:
    _REGISTRY[backend.capabilities.name] = backend


def get(name: str) -> Backend:
    if name not in _REGISTRY:
        raise KeyError(f"backend '{name}' not registered. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def all_backends() -> dict[str, Backend]:
    return dict(_REGISTRY)
