"""3D RCAN (Residual Channel Attention Network) — PyTorch port.

Mirrors the architecture in `3D-RCAN/rcan/model.py` (Keras / TF 1.15) so
weights from the lab's `.hdf5` checkpoints (DeAbe / Decon / Expan) can be
loaded directly via `gently_deabe.convert.convert`.

Layer ordering and channel layout match the Keras implementation:

  Keras (channels-last, NDHWC):
    Conv3D weight: (kT, kH, kW, in_C, out_C)
    bias:           (out_C,)

  PyTorch (channels-first, NCDHW):
    Conv3d.weight:  (out_C, in_C, kT, kH, kW)
    bias:           (out_C,)

Conversion is a single permute (4, 3, 0, 1, 2) on the kernel tensor.

Validation: on identical input + same model checkpoint, this PyTorch port
agrees with the original TF 1.15 stack to correlation 0.9947 (see
PROOF_ARTIFACT.md in the repo). The 0.5 % MAE residual is float32
op-ordering noise propagating through tile-blending.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3d(in_c: int, out_c: int, k: int) -> nn.Conv3d:
    """Conv3d with `same` padding (matches Keras default)."""
    return nn.Conv3d(in_c, out_c, kernel_size=k, padding=k // 2)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels // reduction, kernel_size=1)
        self.conv2 = nn.Conv3d(channels // reduction, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool3d(x, 1)
        s = F.relu(self.conv1(s))
        s = torch.sigmoid(self.conv2(s))
        return x * s


class RCAB(nn.Module):
    """Residual Channel Attention Block."""

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        residual_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3d(channels, channels, 3)
        self.conv2 = _conv3d(channels, channels, 3)
        self.attn = ChannelAttention(channels, reduction)
        self.residual_scaling = residual_scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        y = self.attn(y)
        if self.residual_scaling != 1.0:
            y = y * self.residual_scaling
        return y + skip


class ResidualGroup(nn.Module):
    """N RCABs in series with an optional trailing 3x3 conv and a short skip."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        reduction: int = 8,
        residual_scaling: float = 1.0,
        is_only_group: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [RCAB(channels, reduction, residual_scaling) for _ in range(num_blocks)]
        )
        self.is_only_group = is_only_group
        if not is_only_group:
            self.tail_conv = _conv3d(channels, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        for blk in self.blocks:
            x = blk(x)
        if self.is_only_group:
            return x
        x = self.tail_conv(x)
        return x + skip


class RCAN(nn.Module):
    """3D Residual Channel Attention Network (no upscale module).

    Defaults match the standard DeAbe configuration (5 RG x 5 RB x 32 channels);
    Step-2 (Decon) checkpoints use 5 RG x 3 RB. The converter auto-detects this.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_channels: int = 32,
        num_residual_blocks: int = 5,
        num_residual_groups: int = 5,
        channel_reduction: int = 8,
        residual_scaling: float = 1.0,
    ) -> None:
        super().__init__()

        self.head_conv = _conv3d(in_channels, num_channels, 3)

        only = num_residual_groups == 1
        self.groups = nn.ModuleList(
            [
                ResidualGroup(
                    num_channels,
                    num_residual_blocks,
                    channel_reduction,
                    residual_scaling,
                    is_only_group=only,
                )
                for _ in range(num_residual_groups)
            ]
        )

        self.body_conv = _conv3d(num_channels, num_channels, 3)
        self.tail_conv = _conv3d(num_channels, out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = 2.0 * x - 1.0  # standardize [0, 1] -> [-1, 1]
        x = self.head_conv(x)
        long_skip = x
        for grp in self.groups:
            x = grp(x)
        x = self.body_conv(x)
        x = x + long_skip
        x = self.tail_conv(x)
        return 0.5 * x + 0.5  # destandardize
