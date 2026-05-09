"""Convert Keras / TF 1.15 .hdf5 weights to a PyTorch state_dict.

The 3D-RCAN architecture is built deterministically in both Keras
(`3D-RCAN/rcan/model.py`) and our port (`gently_deabe.model`), so we pair
Conv3D layers by position rather than by name. Keras weight ordering is
permuted to PyTorch's via (4, 3, 0, 1, 2).

Auto-detects num_residual_blocks from the Conv3D count in the .hdf5
(108 = 5 RG x 5 RB, 68 = 5 RG x 3 RB, etc.).
"""
from __future__ import annotations

import argparse
import pathlib

import h5py
import numpy as np
import torch

from .model import RCAN


def _list_keras_conv_layers(weights_path: pathlib.Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk a Keras .hdf5 in build order, yield (kernel, bias) per Conv layer."""
    convs: list[tuple[np.ndarray, np.ndarray]] = []
    with h5py.File(str(weights_path), "r") as f:
        root = f["model_weights"] if "model_weights" in f else f
        layer_names = [n.decode("utf-8") if isinstance(n, bytes) else n
                       for n in root.attrs["layer_names"]]
        for name in layer_names:
            grp = root[name]
            weight_names = [n.decode("utf-8") if isinstance(n, bytes) else n
                            for n in grp.attrs.get("weight_names", [])]
            if not weight_names:
                continue
            datasets = {wn.split("/")[-1].split(":")[0]: grp[wn][...]
                        for wn in weight_names}
            kernel = datasets.get("kernel")
            bias = datasets.get("bias")
            if kernel is None or bias is None:
                continue
            convs.append((kernel, bias))
    return convs


def _ordered_torch_conv_modules(model: RCAN) -> list[tuple[str, torch.nn.Conv3d]]:
    """Return PyTorch Conv3d modules in the same order Keras builds them."""
    out: list[tuple[str, torch.nn.Conv3d]] = []
    out.append(("head_conv", model.head_conv))
    for gi, grp in enumerate(model.groups):
        for bi, blk in enumerate(grp.blocks):
            out.append((f"groups.{gi}.blocks.{bi}.conv1", blk.conv1))
            out.append((f"groups.{gi}.blocks.{bi}.conv2", blk.conv2))
            out.append((f"groups.{gi}.blocks.{bi}.attn.conv1", blk.attn.conv1))
            out.append((f"groups.{gi}.blocks.{bi}.attn.conv2", blk.attn.conv2))
        if not grp.is_only_group:
            out.append((f"groups.{gi}.tail_conv", grp.tail_conv))
    out.append(("body_conv", model.body_conv))
    out.append(("tail_conv", model.tail_conv))
    return out


def _infer_block_count(num_convs: int, num_residual_groups: int = 5) -> int:
    """Back-solve num_residual_blocks (M) from the total Conv3D count.

    For num_residual_groups > 1: total = 3 + N * (4M + 1)
    For num_residual_groups == 1: total = 3 + 4M
    """
    if num_residual_groups == 1:
        m, rem = divmod(num_convs - 3, 4)
        if rem != 0 or m < 1:
            raise ValueError(f"can't infer num_residual_blocks from {num_convs} convs (N=1)")
        return m
    inner = (num_convs - 3) // num_residual_groups
    if 3 + num_residual_groups * inner != num_convs:
        raise ValueError(
            f"conv count {num_convs} not consistent with num_residual_groups={num_residual_groups}"
        )
    m, rem = divmod(inner - 1, 4)
    if rem != 0 or m < 1:
        raise ValueError(
            f"can't infer num_residual_blocks: inner={inner}, expected 4*M+1"
        )
    return m


def convert(
    hdf5_path: pathlib.Path | str,
    in_channels: int = 1,
    out_channels: int = 1,
    num_channels: int = 32,
    num_residual_blocks: int | None = None,
    num_residual_groups: int = 5,
    channel_reduction: int = 8,
    residual_scaling: float = 1.0,
    verbose: bool = True,
) -> tuple[RCAN, dict]:
    """Build a PyTorch RCAN matching the saved arch and load Keras weights into it.

    If num_residual_blocks is None it is auto-detected from the Conv3D count.
    """
    hdf5_path = pathlib.Path(hdf5_path)
    keras_convs = _list_keras_conv_layers(hdf5_path)

    if num_residual_blocks is None:
        num_residual_blocks = _infer_block_count(len(keras_convs), num_residual_groups)
        if verbose:
            print(f"  auto-detected num_residual_blocks={num_residual_blocks} from "
                  f"{len(keras_convs)} convs")

    model = RCAN(
        in_channels=in_channels,
        out_channels=out_channels,
        num_channels=num_channels,
        num_residual_blocks=num_residual_blocks,
        num_residual_groups=num_residual_groups,
        channel_reduction=channel_reduction,
        residual_scaling=residual_scaling,
    )

    torch_convs = _ordered_torch_conv_modules(model)
    if len(keras_convs) != len(torch_convs):
        raise ValueError(
            f"layer count mismatch: keras={len(keras_convs)} pytorch={len(torch_convs)}"
        )

    state: dict = {}
    for (k_kernel, k_bias), (name, t_mod) in zip(keras_convs, torch_convs):
        if k_kernel.ndim != 5:
            raise ValueError(f"{name}: expected 5D kernel, got shape {k_kernel.shape}")
        t_kernel = np.transpose(k_kernel, (4, 3, 0, 1, 2))
        if t_kernel.shape != tuple(t_mod.weight.shape):
            raise ValueError(
                f"{name}: shape mismatch keras={t_kernel.shape} torch={tuple(t_mod.weight.shape)}"
            )
        state[name + ".weight"] = torch.from_numpy(np.ascontiguousarray(t_kernel))
        state[name + ".bias"] = torch.from_numpy(np.ascontiguousarray(k_bias))

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict load mismatch — missing={missing}, unexpected={unexpected}"
        )
    return model, state


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Keras .hdf5 to PyTorch state_dict")
    ap.add_argument("hdf5", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    args = ap.parse_args()

    model, state = convert(args.hdf5)
    torch.save(state, str(args.out))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"saved {args.out} ({n_params:,} params)")


if __name__ == "__main__":
    main()
