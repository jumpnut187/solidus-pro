from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

END_TIME_FLOOR = 0.1
SUPPORTED_RESOLUTIONS = (8, 16, 32, 64, 128)
SETTINGS_RESOLUTIONS = (16, 32, 64, 128)

DEFAULT_CONDITIONING_SIZE = 23
DEFAULT_BASE_CHANNELS = 16
DEFAULT_TRAINING_RESOLUTION = 64

TARGET_NAMES = ("log1p_start", "log1p_local")
LOG_TARGET_CEILING = 11.0

def levels_for_resolution(resolution: int) -> int:
    return max(1, min(4, int(math.log2(max(resolution, 2)))))

if TORCH_AVAILABLE:
    _ModuleBase = nn.Module
else:
    class _ModuleBase:

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is not installed. Install the deep-learning extras with "
                "`pip install -r requirements.txt` (DEEP LEARNING MODULE section)."
            )

class ConvBlock(_ModuleBase):

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

class Downsample(_ModuleBase):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(_ModuleBase):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class FiLM(_ModuleBase):

    def __init__(self, channels: int, conditioning_size: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(conditioning_size, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2 * channels),
        )
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        self.channels = channels

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        params = self.mlp(conditioning)
        delta_gamma, beta = params.chunk(2, dim=-1)
        gamma = 1.0 + delta_gamma.view(*delta_gamma.shape, 1, 1, 1)
        beta = beta.view(*beta.shape, 1, 1, 1)
        return gamma * x + beta

class UNet3D(_ModuleBase):

    def __init__(
        self,
        in_channels: int = 7,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        levels: int = 4,
        conditioning_size: int = DEFAULT_CONDITIONING_SIZE,
        groups: int = 8,
    ):
        super().__init__()
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        if base_channels % groups != 0:
            raise ValueError(
                f"base_channels ({base_channels}) must be divisible by groups ({groups})"
            )
        self.base_channels = int(base_channels)
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

        self.head = nn.Conv3d(channels[0], len(TARGET_NAMES), 1)
        self.register_buffer("target_mean", torch.zeros(len(TARGET_NAMES)))
        self.register_buffer("target_std", torch.ones(len(TARGET_NAMES)))

    def set_target_normalization(self, mean, std) -> None:
        mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std_t = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        if mean_t.shape != self.target_mean.shape or std_t.shape != self.target_std.shape:
            raise ValueError(
                f"Expected {len(TARGET_NAMES)} normalization values, "
                f"got mean {tuple(mean_t.shape)} std {tuple(std_t.shape)}"
            )
        if bool((std_t <= 0).any()):
            raise ValueError(f"target_std must be positive, got {std_t.tolist()}")
        with torch.no_grad():
            self.target_mean.copy_(mean_t.to(self.target_mean.device))
            self.target_std.copy_(std_t.to(self.target_std.device))

    def forward_normalized(
        self, channels: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
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

    def forward(
        self, channels: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.forward_normalized(channels, conditioning)
        mean = self.target_mean.view(1, -1, 1, 1, 1)
        std = self.target_std.view(1, -1, 1, 1, 1)
        log_fields = torch.clamp(normalized * std + mean, min=0.0, max=LOG_TARGET_CEILING)
        fields = torch.expm1(log_fields)
        start = fields[:, 0:1]
        local = fields[:, 1:2]
        return torch.cat([start, start + local, local], dim=1)

DEEP_MANIFEST_FILENAME = "deep_manifest.json"

@dataclass
class DeepModelManifest:

    kind: str = "torch_unet"
    targets: List[str] = field(default_factory=list)
    quantization: Dict[str, str] = field(default_factory=dict)
    accuracy: Dict[str, Any] = field(default_factory=dict)
    alloy_coverage: List[str] = field(default_factory=list)
    trained_on: str = ""
    training_device: str = ""
    conditioning_size: int = DEFAULT_CONDITIONING_SIZE
    base_channels: int = DEFAULT_BASE_CHANNELS
    epochs: int = 0
    target_names: List[str] = field(default_factory=lambda: list(TARGET_NAMES))
    target_mean: List[float] = field(default_factory=list)
    target_std: List[float] = field(default_factory=list)
    training: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeepModelManifest":
        allowed = {f for f in asdict(cls()).keys()}
        clean = {k: v for k, v in (data or {}).items() if k in allowed}
        return cls(**clean)

def save_deep_manifest(manifest: DeepModelManifest, directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, DEEP_MANIFEST_FILENAME)
    with open(path, "w") as fh:
        json.dump(manifest.to_dict(), fh, indent=2)
    return path

def read_deep_manifest(directory: str) -> Optional[DeepModelManifest]:
    path = os.path.join(directory, DEEP_MANIFEST_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return DeepModelManifest.from_dict(json.load(fh))
    except Exception:
        return None
