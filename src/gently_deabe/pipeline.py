"""Multi-step pipeline orchestration (Step 1 DeAbe -> Step 2 Decon -> Step 3 Expan).

A pipeline is a list of `PipelineStep`s. Each step:
  - Looks up its model by name in the registry
  - Optionally Z-interpolates the input first (for the isotropic-grid step
    between DeAbe and Decon)
  - Runs tiled inference
  - Clamps negatives to zero (matches the reference behaviour)

Outputs are float32 [0, 1]. Use `to_uint16(arr, scale_value)` to write a
TIFF compatible with the reference pipeline.
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Iterable

import numpy as np
import torch
from scipy.ndimage import zoom

from .inference import apply, normalize
from .registry import ModelRegistry


@dataclasses.dataclass
class PipelineStep:
    """One stage of the multi-step pipeline."""

    name: str                              # human-friendly label, e.g. "DeAbe"
    model: str                             # registry model name
    block_shape: tuple[int, int, int] = (32, 128, 128)
    overlap_shape: tuple[int, int, int] | None = None
    interp_ratio: tuple[float, float, float] | None = None
    """If set, zoom the input by this ratio before running this step
    (the typical use is `(6.154, 1, 1)` between DeAbe and Decon to make
    the volume isotropic before deconvolution)."""


class Pipeline:
    def __init__(self, registry: ModelRegistry, device: str = "cuda") -> None:
        self.registry = registry
        self.device = device

    def run(
        self,
        image: np.ndarray,
        steps: Iterable[PipelineStep],
        *,
        normalize_each_step: bool = True,
        verbose: bool = False,
    ) -> np.ndarray:
        """Run a sequence of steps on a 3D float32/uint16 input. Returns float32 in [0, 1]."""
        if image.ndim != 3:
            raise ValueError(f"expected 3D input, got shape {image.shape}")

        x = normalize(image)
        for step in steps:
            if step.interp_ratio is not None:
                x = zoom(x, step.interp_ratio, order=1)
                if normalize_each_step:
                    x = normalize(x)

            model = self.registry.get(step.model)
            x = apply(
                model, x,
                step.block_shape,
                step.overlap_shape,
                device=self.device,
                verbose=verbose,
            )
            np.clip(x, 0, None, out=x)

        return x


def to_uint16(arr: np.ndarray, scale_value: float = 2000.0) -> np.ndarray:
    """Convert a float32 [0, 1] prediction to a uint16 TIFF-friendly array.

    Matches `DeAbePlus/apply_multi_steps.py`: out = clip(scale_value * pred, 0, 65535).
    Scale 2000 is the published default; the lab's CND-1 reference uses something
    larger (mean output ~3700 ADU) — set this to match if you want byte-identical
    intensity scaling to a specific reference.
    """
    return np.clip(scale_value * arr, 0, 65535).astype(np.uint16)


def standard_pipeline_steps(
    *,
    spim_view: str = "spimb",
    z_interp_ratio: float = 6.154,
    block_shape_step1: tuple[int, int, int] = (32, 128, 128),
    block_shape_step2: tuple[int, int, int] = (128, 128, 128),
    block_shape_step3: tuple[int, int, int] = (64, 128, 128),
    include_step3: bool = False,
) -> list[PipelineStep]:
    """The canonical DeAbe -> Decon (-> Expan) pipeline as used in the Gently /
    CND-1 lineaging workflow. Pass to `Pipeline.run()`."""
    steps = [
        PipelineStep(
            name="DeAbe (Step 1)",
            model=f"step1_{spim_view}",
            block_shape=block_shape_step1,
        ),
        PipelineStep(
            name="Decon (Step 2)",
            model="step2_decon",
            interp_ratio=(z_interp_ratio, 1.0, 1.0),
            block_shape=block_shape_step2,
        ),
    ]
    if include_step3:
        steps.append(
            PipelineStep(
                name="Expan (Step 3)",
                model="step3_expan",
                block_shape=block_shape_step3,
            )
        )
    return steps
