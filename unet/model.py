"""Stage 3 segmentation models: SMP U-Net / U-Net++ plus a custom Attention U-Net.

All variants use a pretrained (ImageNet) encoder — BUSI is too small to learn
strong low-level features from scratch. ``build_unet`` is the single factory
every training/evaluation script should call; the concrete architecture and
encoder are config-driven (see ``config.yaml`` / ``configs/unet_*.yaml``).

- ``unet`` / ``unetpp``: thin passthroughs to ``segmentation_models_pytorch``.
- ``attention_unet``: a from-scratch decoder with Oktay-style attention gates
  on every skip connection, built on top of an SMP pretrained encoder. SMP
  itself has no literal "Attention U-Net"; its closest built-in is SCSE
  decoder attention (channel+spatial squeeze-excite), which is a different
  mechanism from the gated skip connections this prompt asks for, so the
  gates are implemented explicitly here.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn


def build_unet(
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """Factory for every Stage 3 architecture variant used in this project."""
    if architecture == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    if architecture == "unetpp":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    if architecture == "attention_unet":
        return AttentionUNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    raise ValueError(f"Unknown U-Net architecture: {architecture!r}")


class ConvBlock(nn.Module):
    """Two 3x3 conv + BN + ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Oktay et al. (2018) attention gate: suppresses irrelevant skip-connection regions.

    ``g`` (gating signal, from the coarser decoder stage) and ``x`` (the
    encoder skip feature) are projected to a shared intermediate space,
    summed, passed through ReLU then a 1x1 conv + sigmoid to produce a
    per-pixel attention coefficient, which rescales ``x`` before it is
    concatenated into the decoder path.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        combined = self.relu(self.gate_proj(gate) + self.skip_proj(skip))
        alpha = self.attention(combined)
        return skip * alpha


class AttentionDecoderBlock(nn.Module):
    """Upsample -> attention-gate the skip connection -> concat -> ConvBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.has_skip = skip_channels > 0
        if self.has_skip:
            self.attention_gate = AttentionGate(
                gate_channels=in_channels, skip_channels=skip_channels, inter_channels=out_channels // 2
            )
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.upsample(x)
        if self.has_skip and skip is not None:
            gated_skip = self.attention_gate(gate=x, skip=skip)
            x = torch.cat([x, gated_skip], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """Attention U-Net: pretrained SMP encoder + custom attention-gated decoder."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
    ) -> None:
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights
        )
        # encoder.out_channels[0] is the input-resolution "identity" stage;
        # stages 1..5 are the real skip connections, deepest last.
        encoder_channels = list(self.encoder.out_channels)
        skip_channels = encoder_channels[1:][::-1]  # deep -> shallow, e.g. [512,256,128,64,64]
        decoder_in_channels = [skip_channels[0]] + list(decoder_channels[:-1])
        # Deepest block has no skip (already the bottleneck); remaining 4 use
        # the 4 encoder stages; the final block upsamples once more with no
        # skip available (matches encoder_channels[0], the identity stage).
        skips_per_block = skip_channels[1:] + [0]

        self.blocks = nn.ModuleList(
            [
                AttentionDecoderBlock(in_ch, skip_ch, out_ch)
                for in_ch, skip_ch, out_ch in zip(decoder_in_channels, skips_per_block, decoder_channels)
            ]
        )
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)   # list: [identity, stage1, stage2, stage3, stage4, stage5]
        skips = features[1:-1][::-1]  # [stage4, stage3, stage2, stage1], deep -> shallow
        skips = skips + [None]        # last block has no skip

        decoder_input = features[-1]
        for block, skip in zip(self.blocks, skips):
            decoder_input = block(decoder_input, skip)

        return self.segmentation_head(decoder_input)
