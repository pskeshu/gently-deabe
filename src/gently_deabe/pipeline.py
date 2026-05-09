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
import time
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


def lab_exact_diSPIM(
    image: np.ndarray,
    model_step1: torch.nn.Module,
    model_step2: torch.nn.Module,
    *,
    zoom_z: int = 6,
    overlap_shape: tuple[int, int, int] = (2, 32, 32),
    device: str = "cuda",
    return_uint16: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """The lab's exact 2-step diSPIM DL pipeline.

    Mirrors `Worm_Untwisting_DataProcessing/Image_processing/Deep_Learning/
    RCAN_Apply_diSPIM_TwoSteps0.py` line-for-line:

        x  = image[::zoom_z, :, :]                                # decimate Z
        s1 = apply(m1, normalize(x), block=(z1, 128, 128), overlap=(2,32,32))
        s1[s1 < 0] = 0
        s1 = zoom(s1, (zoom_z, 1, 1))                             # re-upsample Z
        s2 = apply(m2, normalize(s1), block=(z2, 128, 128), overlap=(2,32,32))
        out = (s2 - s2.min()) / (s2.max() - s2.min()) * 65535     # min-max stretch
        return out.astype(uint16)

    With the lab's own Model_step1 / Model_step2 weights, this matches the
    lab's TF 1.15 production output to correlation **0.998** on real
    volumes (see PROOF_ARTIFACT.md).

    Returns the result and a dict of per-step timings: keys `step1`,
    `zoom`, `step2`, `total`, all in seconds.
    """
    if image.ndim != 3:
        raise ValueError(f"expected 3D image, got shape {image.shape}")

    timings: dict[str, float] = {}
    t_total = time.time()

    # Step 1: decimate Z, run, clip
    t0 = time.time()
    x = image[::zoom_z, :, :].astype(np.float32)
    n = normalize(x)
    z1 = x.shape[0]
    s1 = apply(model_step1, n,
               block_shape=(z1, 128, 128),
               overlap_shape=overlap_shape,
               device=device)
    np.clip(s1, 0, None, out=s1)
    timings["step1"] = time.time() - t0

    # Re-upsample Z by the same factor
    t0 = time.time()
    s1z = zoom(s1, (zoom_z, 1, 1))
    timings["zoom"] = time.time() - t0

    # Step 2: full-Z single tile
    t0 = time.time()
    n = normalize(s1z)
    s2 = apply(model_step2, n,
               block_shape=(s1z.shape[0], 128, 128),
               overlap_shape=overlap_shape,
               device=device)
    timings["step2"] = time.time() - t0

    timings["total"] = time.time() - t_total

    if not return_uint16:
        return s2, timings

    # Min-max stretch to uint16 — matches the lab's `(s2 - s2.min()) /
    # (s2.max() - s2.min()) * 65535` final scaling, NOT the constant
    # scale_value used by DeAbePlus/apply_multi_steps.py.
    inmin, inmax = float(s2.min()), float(s2.max())
    out = (s2 - inmin) / (inmax - inmin + 1e-9) * 65535.0
    return out.astype(np.uint16), timings
