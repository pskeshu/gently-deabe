"""Named-model registry — maps short logical names to .hdf5 weight paths.

Loaded from a YAML file. Models are loaded lazily (first request triggers
`convert()` which reads the .hdf5 and builds the PyTorch model).

See `configs/highNA_diSPIM_Nuclear.example.yaml` for the YAML schema.
"""
from __future__ import annotations

import dataclasses
import pathlib
import threading
from typing import Iterable

import torch
import yaml

from .convert import convert
from .model import RCAN


@dataclasses.dataclass
class ModelEntry:
    name: str
    path: pathlib.Path
    kind: str = "rcan"
    description: str = ""

    def resolve_checkpoint(self) -> pathlib.Path:
        """If `path` is a directory of .hdf5 checkpoints, pick the lowest-loss one.
        If it's a single file, use it directly."""
        p = self.path
        if p.is_file():
            return p
        if not p.is_dir():
            raise FileNotFoundError(p)
        hdf5s = list(p.glob("*.hdf5"))
        if not hdf5s:
            raise FileNotFoundError(f"no .hdf5 in {p}")

        def _val_loss(q: pathlib.Path) -> float:
            try:
                return float(q.stem.split("_")[-1])
            except (ValueError, IndexError):
                return float("inf")

        return min(hdf5s, key=_val_loss)


class ModelRegistry:
    """In-memory registry of named models. Models load on first use."""

    def __init__(self, entries: Iterable[ModelEntry], device: str = "cuda") -> None:
        self._entries: dict[str, ModelEntry] = {e.name: e for e in entries}
        self._cache: dict[str, RCAN] = {}
        self._lock = threading.Lock()
        self._device = device

    @classmethod
    def from_yaml(cls, path: pathlib.Path | str, device: str = "cuda") -> "ModelRegistry":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        entries = [
            ModelEntry(
                name=name,
                path=pathlib.Path(spec["path"]),
                kind=spec.get("kind", "rcan"),
                description=spec.get("description", ""),
            )
            for name, spec in (cfg.get("models") or {}).items()
        ]
        return cls(entries, device=device)

    def names(self) -> list[str]:
        return list(self._entries.keys())

    def get_entry(self, name: str) -> ModelEntry:
        if name not in self._entries:
            raise KeyError(f"unknown model {name!r}; available: {self.names()}")
        return self._entries[name]

    def get(self, name: str) -> RCAN:
        """Return the (loaded, on-device) model for `name`. Loads on first call."""
        with self._lock:
            if name not in self._cache:
                entry = self.get_entry(name)
                checkpoint = entry.resolve_checkpoint()
                model, _ = convert(checkpoint)
                model.to(self._device).eval()
                self._cache[name] = model
            return self._cache[name]

    def loaded(self) -> list[str]:
        return list(self._cache.keys())

    def evict(self, name: str) -> bool:
        """Drop a loaded model from cache to free GPU memory. Returns True if evicted."""
        with self._lock:
            if name in self._cache:
                del self._cache[name]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
