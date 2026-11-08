"""Classification backbones for a controlled comparison that has not been run.

ResNet50 remains the only backbone with a saved BUSI checkpoint. DenseNet121,
EfficientNet-B0, and ConvNeXt-Tiny can be constructed for a later experiment.
This module does not declare a winner.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    convnext_tiny,
    densenet121,
    efficientnet_b0,
    resnet50,
)

BACKBONES = ("resnet50", "densenet121", "efficientnet_b0", "convnext_tiny")
TRAINED_CHECKPOINT = {
    "resnet50": "checkpoints/resnet/best_resnet.pth",
}
UNTRAINED = [name for name in BACKBONES if name not in TRAINED_CHECKPOINT]


def _adapt_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if in_channels == conv.in_channels:
        return conv
    import torch

    updated = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        shared = min(in_channels, conv.in_channels)
        updated.weight[:, :shared] = conv.weight[:, :shared]
        if in_channels > conv.in_channels:
            filler = conv.weight.mean(dim=1, keepdim=True)
            updated.weight[:, conv.in_channels :] = filler
        if conv.bias is not None and updated.bias is not None:
            updated.bias.copy_(conv.bias)
    return updated


def _replace_first(model: nn.Module, name: str, in_channels: int) -> None:
    if name == "resnet50":
        model.conv1 = _adapt_conv(model.conv1, in_channels)
    elif name == "densenet121":
        model.features.conv0 = _adapt_conv(model.features.conv0, in_channels)
    elif name == "efficientnet_b0":
        model.features[0][0] = _adapt_conv(model.features[0][0], in_channels)
    elif name == "convnext_tiny":
        model.features[0][0] = _adapt_conv(model.features[0][0], in_channels)
    else:
        raise ValueError(name)


def build_classifier(
    backbone: str = "resnet50",
    *,
    pretrained: bool = True,
    num_classes: int = 2,
    in_channels: int = 3,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a binary classifier. ``pretrained=False`` avoids a weight download."""
    if backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of {BACKBONES}")
    if backbone == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif backbone == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    if in_channels != 3:
        if pretrained and in_channels > 3:
            _replace_first(model, backbone, in_channels)
        else:
            _replace_first(model, backbone, in_channels)
    if freeze_backbone:
        head_names = ("fc", "classifier")
        for param_name, param in model.named_parameters():
            if not param_name.startswith(head_names):
                param.requires_grad = False
    return model
