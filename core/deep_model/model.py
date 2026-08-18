"""3D U-Net architecture with FiLM conditioning (``UNet3D``).

Input: ``(7, R, R, R)`` channel grid (same channel order as
``core.surrogate.fields.CHANNELS``) plus a global conditioning vector
(normalized alloy + config, size >= 20).

Encoder: 4 levels, each ``Conv3D + GroupNorm + SiLU`` with stride-2
downsampling; base channels 32 doubling per level (32 -> 64 -> 128 -> 256).
Bottleneck: conv block at the lowest resolution.  Decoder: 4 levels with
transposed-conv upsample, skip concatenation, ``Conv3D + GroupNorm + SiLU``,
and a FiLM block per level that modulates features by the conditioning
vector.  Head: 1x1x1 conv producing ``(3, R, R, R)`` = start_time, end_time
(log space), local_time.

The number of levels adapts to the resolution: the network needs R divisible
by 8 (R=8 trains with 3 levels; R in {16, 32, 64, 128} trains with 4).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

END_TIME_FLOOR = 0.1
SUPPORTED_RESOLUTIONS = (8, 16, 32, 64, 128)
SETTINGS_RESOLUTIONS = (16, 32, 64, 128)

DEFAULT_CONDITIONING_SIZE = 23  # 15 alloy + 8 config features


def levels_for_resolution(resolution: int) -> int:
    """Number of encoder/decoder levels for a resolution.

    The 4-level U-Net needs R / 2^4 >= 1 (R divisible by 16).  R = 8 gets 3
    levels so the tiny-overfit tests can train quickly; larger resolutions
    are capped at 4 levels per the architecture plan.
    """
    return max(1, min(4, int(math.log2(max(resolution, 2)))))


class ConvBlock(nn.Module):
    """Two Conv3D + GroupNorm + SiLU blocks."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class Downsample(nn.Module):
    """Stride-2 conv for encoder downsampling (optional channel doubling)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Transposed-conv upsample by factor 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FiLM(nn.Module):
    """Feature-wise linear modulation driven by the conditioning vector.

    The conditioning vector goes through an MLP producing per-channel scale
    (gamma) and shift (beta) for the incoming feature map, letting the model
    modulate physics per alloy/config.
    """

    def __init__(self, channels: int, conditioning_size: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(conditioning_size, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2 * channels),
        )
        self.channels = channels

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        params = self.mlp(conditioning)  # (B, 2C) or (2C,)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.view(*gamma.shape, 1, 1, 1)
        beta = beta.view(*beta.shape, 1, 1, 1)
        return gamma * x + beta


class UNet3D(nn.Module):
    """3D U-Net with FiLM conditioning for solidification schedules."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 3,
        base_channels: int = 32,
        levels: int = 4,
        conditioning_size: int = DEFAULT_CONDITIONING_SIZE,
        groups: int = 8,
    ):
        super().__init__()
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        channels = [base_channels * (2 ** i) for i in range(levels)]

        self.encoder = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, c in enumerate(channels):
            block_in = in_channels if i == 0 else c
            self.encoder.append(ConvBlock(block_in, c, groups))
            next_c = channels[i + 1] if i + 1 < len(channels) else c
            self.downs.append(Downsample(c, next_c))

        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, groups)

        self.decoder_ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.films = nn.ModuleList()
        prev = channels[-1] * 2
        for i in range(levels):
            c = channels[levels - 1 - i]
            self.decoder_ups.append(Upsample(prev, c))
            self.decoder_blocks.append(ConvBlock(2 * c, c, groups))
            self.films.append(FiLM(c, conditioning_size))
            prev = c

        self.head = nn.Conv3d(channels[0], out_channels, 1)

    def forward(
        self, channels: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """Predict schedule fields from channels + conditioning.

        Returns:
            Shape ``(B, 3, R, R, R)``: start_time, log(end_time),
            local_time.
        """
        x = channels
        skips: List[torch.Tensor] = []
        for block, down in zip(self.encoder, self.downs):
            x = block(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for up, block, film, skip in zip(
            self.decoder_ups, self.decoder_blocks, self.films, reversed(skips)
        ):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
            x = film(x, conditioning)
        return self.head(x)


# ---------------------------------------------------------------------------
# Deep artifact manifest
# ---------------------------------------------------------------------------

DEEP_MANIFEST_FILENAME = "deep_manifest.json"


@dataclass
class DeepModelManifest:
    """Metadata for a trained deep model artifact.

    The *standard* ``manifest.json`` (kind="torch_unet", backend="torch") is
    written alongside so the shared ``load_model`` dispatch reads it
    unchanged; this file carries the deep-only details (export targets,
    quantization, training device).
    """

    kind: str = "torch_unet"
    targets: List[str] = field(default_factory=list)
    quantization: Dict[str, str] = field(default_factory=dict)
    accuracy: Dict[str, Any] = field(default_factory=dict)
    alloy_coverage: List[str] = field(default_factory=list)
    trained_on: str = ""
    training_device: str = ""
    conditioning_size: int = DEFAULT_CONDITIONING_SIZE
    epochs: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeepModelManifest":
        allowed = {f for f in asdict(cls()).keys()}
        clean = {k: v for k, v in (data or {}).items() if k in allowed}
        return cls(**clean)


def save_deep_manifest(manifest: DeepModelManifest, directory: str) -> str:
    """Persist ``deep_manifest.json`` into a model directory."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, DEEP_MANIFEST_FILENAME)
    with open(path, "w") as fh:
        json.dump(manifest.to_dict(), fh, indent=2)
    return path


def read_deep_manifest(directory: str) -> Optional[DeepModelManifest]:
    """Load ``deep_manifest.json`` from a model directory, or None."""
    path = os.path.join(directory, DEEP_MANIFEST_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return DeepModelManifest.from_dict(json.load(fh))
    except Exception:
        return None
