"""Tiled 3D inference matching `3D-RCAN/rcan/utils.apply`.

The reference algorithm:
  1. Slide a fixed-size block over the 3D input with `overlap` voxels between
     adjacent blocks per dimension.
  2. Pad the trailing partial block with zeros up to block size.
  3. Pre-compute a per-block linear-ramp weight mask whose centre is 1 and
     edges fade linearly toward 0 inside the overlap band.
  4. Run the model on each block; accumulate `pred * weight` and `weight` in
     two float32 buffers; final output is `accum / sum_weight`.

We reproduce that exactly so output matches the TF 1.15 reference within
single-precision roundoff.
"""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import torch


def normalize(image: np.ndarray, p_min: float = 2.0, p_max: float = 99.9) -> np.ndarray:
    """Percentile contrast stretch (matches `rcan.utils.normalize`)."""
    low, high = np.percentile(image, (p_min, p_max))
    return ((image.astype(np.float32) - low) / (high - low + 1e-6)).astype(np.float32)


def _build_block_weight(block_shape: Sequence[int], overlap: Sequence[int]) -> np.ndarray:
    """Linear-ramp blending mask matching the reference `block_weight` build."""
    if any(2 * o >= m for m, o in zip(block_shape, overlap)):
        raise ValueError(f"overlap {overlap} too large for block_shape {block_shape}")
    core = np.ones(
        [m - 2 * o for m, o in zip(block_shape, overlap)], dtype=np.float32
    )
    bw = np.pad(core, [(o + 1, o + 1) for o in overlap], mode="linear_ramp")
    return bw[tuple(slice(1, -1) for _ in overlap)]


def apply(
    model: torch.nn.Module,
    image: np.ndarray,
    block_shape: Sequence[int],
    overlap_shape: Sequence[int] | None = None,
    device: torch.device | str = "cuda",
    verbose: bool = False,
) -> np.ndarray:
    """Apply `model` tile-by-tile to a 3D float32 image, return float32 prediction.

    `image` is expected pre-normalized (use `normalize` above) and 3D
    `(Z, Y, X)`. The output has the same shape and dtype.
    """
    if image.ndim != 3:
        raise ValueError(f"expected 3D image, got shape {image.shape}")
    block_shape = tuple(int(b) for b in block_shape)
    if overlap_shape is None:
        overlap_shape = (2, 32, 32)
    overlap_shape = tuple(int(o) for o in overlap_shape)
    if len(block_shape) != 3 or len(overlap_shape) != 3:
        raise ValueError("block_shape and overlap_shape must be 3D")

    step_shape = tuple(m - o for m, o in zip(block_shape, overlap_shape))
    if any(s <= 0 for s in step_shape):
        raise ValueError(
            f"block_shape {block_shape} must exceed overlap_shape {overlap_shape}"
        )

    block_weight = _build_block_weight(block_shape, overlap_shape)

    out = np.zeros(image.shape, dtype=np.float32)
    sum_weight = np.zeros(image.shape, dtype=np.float32)

    n_steps = tuple(int(np.ceil(i / s)) for i, s in zip(image.shape, step_shape))
    block_starts = list(itertools.product(*(np.arange(n) * s for n, s in zip(n_steps, step_shape))))

    was_training = model.training
    model.eval()
    iterator = block_starts
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(block_starts, dynamic_ncols=True)
        except ImportError:
            pass

    with torch.no_grad():
        for tl in iterator:
            br = tuple(min(t + b, i) for t, b, i in zip(tl, block_shape, image.shape))
            src = tuple(slice(s, e) for s, e in zip(tl, br))
            actual = tuple(e - s for s, e in zip(tl, br))

            patch = image[src]
            if actual != block_shape:
                pad_width = [(0, b - a) for b, a in zip(block_shape, actual)]
                patch = np.pad(patch, pad_width, mode="constant")

            x = torch.from_numpy(patch).to(device).unsqueeze(0).unsqueeze(0)
            y = model(x).squeeze(0).squeeze(0).detach().cpu().numpy()

            crop = tuple(slice(0, a) for a in actual)
            pred = y[crop]
            w = block_weight[crop]

            out[src] += pred * w
            sum_weight[src] += w

    if was_training:
        model.train()

    return out / np.maximum(sum_weight, 1e-6)
